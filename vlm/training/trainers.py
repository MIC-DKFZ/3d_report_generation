import contextlib
import functools
import gc
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from typing import Optional

import torch.distributed as dist
from accelerate import skip_first_batches
from accelerate.utils import is_torch_xla_available
from transformers import TrainerState
from transformers.debug_utils import DebugOption
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import speed_metrics, TrainOutput, SaveStrategy
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import check_torch_load_is_safe
from trl import SFTTrainer

from vlm.training.schedulers.base import SchedulerBase
from vlm.utils import logger
from vlm.utils.training.checkpoint_state import load_model_checkpoint, save_model_checkpoint


class VLMSFTTrainer(SFTTrainer):

    def __init__(
        self,
        scheduler: SchedulerBase | None = None,
        optimizer_scheduler_checkpoint: str | Path | None = None,
        model_weights_preloaded: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Our custom scheduler
        self.scheduler = scheduler
        self.optimizer_scheduler_checkpoint = (
            str(optimizer_scheduler_checkpoint)
            if optimizer_scheduler_checkpoint is not None
            else None
        )
        self.model_weights_preloaded = model_weights_preloaded

    def get_batch_samples(self, epoch_iterator, num_batches: int, device: torch.device):
        """Count causal-LM targets across the actual accumulation window."""
        batches = []
        for _ in range(num_batches):
            try:
                batches.append(next(epoch_iterator))
            except StopIteration:
                break
        if not batches:
            return batches, None

        # The loss predicts labels[..., 1:]; labels[..., 0] is never a target.
        # Count only supervised targets, excluding prompt and padding labels.
        num_items_in_batch = sum(
            batch["labels"][..., 1:].ne(-100).sum() for batch in batches
        ).to(device)
        if self.args.average_tokens_across_devices:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
        if num_items_in_batch.item() == 0:
            raise ValueError("The accumulation window contains no supervised tokens.")
        return batches, num_items_in_batch

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self.model_weights_preloaded:
            logger.info_main(
                "Skipping Trainer model-weight loading because checkpoint weights were preloaded."
            )
            return

        load_model_checkpoint(model if model is not None else self.model, resume_from_checkpoint)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        model = self.accelerator.unwrap_model(self.model)
        save_model_checkpoint(model, output_dir, state_dict=state_dict)

    def _load_best_model(self):
        checkpoint = self.state.best_model_checkpoint
        if checkpoint is not None:
            load_model_checkpoint(self.accelerator.unwrap_model(self.model), checkpoint)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Disable this function, as we only support our custom schedulers.
        """
        pass

    def create_optimizer(self):
        pass

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        pass


    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            # Comment this out so we only use our custom LR logging (WandbGroupedLRCallback)
            # if learning_rate is not None:
            #     logs["learning_rate"] = learning_rate
            # else:
            #     logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overwrite the inner training loop to accomplish two main things:
        1. Log to our log files.
        2. Allow integration of our own callback to support our custom schedulers.
        """
        self.already_complete_on_resume = False
        optimizer_scheduler_ckpt_dir = self.optimizer_scheduler_checkpoint
        scheduler_is_initialized = False
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

            self.state.train_batch_size = self._train_batch_size
        logger.debug_main(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                DebugUnderflowOverflow(self.model)

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Ensure scheduler metadata is initialized and an optimizer exists before
        # the first accelerator prepare call.
        self.scheduler.initialize(
            model=self.model,
            max_steps=max_steps
        )
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler, _ = self.scheduler.prepare_training_step(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                global_step=self.state.global_step,
                max_steps=max_steps,
            )

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # Already wrapped models (e.g. DataParallel) only need optimizer preparation.
        use_accelerator_prepare = model is self.model

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                    if self.is_tp_enabled:
                        self.optimizer = self.accelerator.prepare(self.optimizer)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # Prepare schedulers that do not expose a step method together with the model.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        # NOTE:
        # For continuing training with our custom trainer we have to delay the lr_scheduler creation
        # therefore we skip the function here and load them in the SchedulerBase's update fct.
        # self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is the wrapped model (e.g. DDP).

        # Train!
        logger.info_main("")
        logger.info_main("***** Running training *****")
        logger.info_main(f"  Num examples = {num_examples:,}")
        logger.info_main(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_main(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info_main(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info_main(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info_main(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info_main(f"  Total optimization steps = {max_steps:,}")
        logger.info_main(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
        logger.info_main("")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.state.train_batch_size = self._train_batch_size
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info_main("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info_main(f"  Continuing training from epoch {epochs_trained}")
            logger.info_main(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info_main(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        if resume_from_checkpoint is not None and self.state.global_step >= max_steps:
            self.already_complete_on_resume = True
            logger.info_main(
                "Training is already complete at global step "
                f"{self.state.global_step}/{max_steps}; no additional updates will run."
            )
            epochs_trained = num_train_epochs
            steps_trained_in_current_epoch = 0

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            rng_to_sync = False

            # Handle resumption from checkpoint
            if epoch == epochs_trained and resume_from_checkpoint is not None:
                if steps_trained_in_current_epoch > 0 and not args.ignore_data_skip:
                    epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                    step = steps_trained_in_current_epoch - 1
                    rng_to_sync = True
                elif steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                # Store the number of batches for current gradient accumulation
                # This is used to correctly scale the loss when the last accumulation step has fewer batches
                self.current_gradient_accumulation_steps = len(batch_samples)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen not in ["no", False]:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            if self.args.include_num_input_tokens_seen == "non_padding":
                                if "attention_mask" in inputs:
                                    input_tokens = inputs["attention_mask"].sum()
                                elif (
                                    self.processing_class is not None
                                    and hasattr(self.processing_class, "pad_token_id")
                                    and self.processing_class.pad_token_id is not None
                                ):
                                    input_tokens = (
                                        inputs[main_input_name] != self.processing_class.pad_token_id
                                    ).sum()
                                else:
                                    logger.warning(
                                        "Could not determine method to count non-padding tokens, falling back to counting all tokens."
                                    )
                                    input_tokens = inputs[main_input_name].numel()
                            else:
                                input_tokens = inputs[main_input_name].numel()

                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()

                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    if step % args.gradient_accumulation_steps == 0:
                        # Call our own scheduler
                        if optimizer_scheduler_ckpt_dir and not scheduler_is_initialized:
                            scheduler_is_initialized = True
                            check_torch_load_is_safe()
                            self.optimizer, self.lr_scheduler, changed_requires_grad_state = self.scheduler.resume(
                                model=self.model,
                                global_step=self.state.global_step,
                                max_steps=self.state.max_steps,
                                checkpoint_dir=optimizer_scheduler_ckpt_dir,
                            )
                        else:
                            self.optimizer, self.lr_scheduler, changed_requires_grad_state = (
                                self.scheduler.prepare_training_step(
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    lr_scheduler=self.lr_scheduler,
                                    global_step=self.state.global_step,
                                    max_steps=self.state.max_steps,
                                )
                            )

                        if changed_requires_grad_state:
                            del self.model_wrapped
                            gc.collect()
                            torch.cuda.empty_cache()

                            self.accelerator.wait_for_everyone()
                            model = self.accelerator.prepare(self.model)
                            self.model_wrapped = model
                            self.accelerator.wait_for_everyone()

                        if not getattr(self.optimizer, "_is_accelerate_prepared", False):
                            # logger.warning_main("RE-WRAPPING OPTIMIZER")
                            self.optimizer = self.accelerator.prepare(self.optimizer)

                        if not getattr(self.lr_scheduler, "_is_accelerate_prepared", False):
                            # logger.warning_main("RE-WRAPPING LR-SCHEDULER")
                            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
                            self.lr_scheduler.step(self.state.global_step)  # Step once to set correct LR for
                                                                            # offset LR schedulers using 'global_step'

                        self.callback_handler.optimizer = self.optimizer
                        self.callback_handler.lr_scheduler = self.lr_scheduler

                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if self.use_apex:
                                from apex import amp

                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                grad_norm_context = contextlib.nullcontext
                                if self.is_tp_enabled:
                                    from torch.distributed._tensor.experimental import implicit_replication

                                    grad_norm_context = implicit_replication
                                with grad_norm_context():
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                            grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        context = contextlib.nullcontext
                        if self.is_tp_enabled:
                            from torch.distributed._tensor.experimental import implicit_replication

                            context = implicit_replication

                        with context():
                            self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        # learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step(self.state.global_step)

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            # learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        # logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
