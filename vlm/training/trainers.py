# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified to support VLM's 3D medical image conditioning and custom training workflow.

import contextlib
import functools
from pathlib import Path

import torch
from accelerate import skip_first_batches
from accelerate.utils import DistributedType
from transformers import Trainer
from transformers.utils import check_torch_load_is_safe, is_torch_xla_available

from vlm.training.schedulers.base import SchedulerBase
from vlm.utils import logger
from vlm.training.checkpoint_state import load_model_checkpoint, save_model_checkpoint


class VLMTrainer(Trainer):
    """
    Trainer whose optimizer and LR scheduler are owned by a staged scheduler.

    At the start of every optimizer step, the staged scheduler may change which parameters are
    trainable and rebuild the optimizer and LR scheduler. The copied '_run_epoch' applies these
    changes, so re-sync it with the pinned transformers version on upgrades.
    """

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
        self.already_complete_on_resume = False

    def _get_num_items_in_batch(self, batch_samples, device):
        num_items_in_batch = super()._get_num_items_in_batch(batch_samples, device)
        if num_items_in_batch is not None and num_items_in_batch.item() == 0:
            raise ValueError("The accumulation window contains no supervised tokens.")
        return num_items_in_batch

    def log(self, logs, start_time=None):
        # WandbGroupedLRCallback logs the learning rate of every parameter group instead.
        logs.pop("learning_rate", None)
        super().log(logs, start_time)

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

    # ---- Staged scheduler ----

    def _init_training_state(
        self, max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial
    ):
        epochs_trained, steps_trained_in_current_epoch = super()._init_training_state(
            max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial
        )
        self.state.train_batch_size = self._train_batch_size

        self.already_complete_on_resume = (
            resume_from_checkpoint is not None and self.state.global_step >= max_steps
        )
        if self.already_complete_on_resume:
            logger.info_main(
                "Training is already complete at global step "
                f"{self.state.global_step}/{max_steps}; no additional updates will run."
            )
            return num_train_epochs, 0
        return epochs_trained, steps_trained_in_current_epoch

    def create_optimizer(self, model=None):
        # Runs after the trainer state is initialized, so resuming knows its global step.
        self.scheduler.initialize(model=self.model, max_steps=self.state.max_steps)
        step = dict(
            model=self.model, global_step=self.state.global_step, max_steps=self.state.max_steps
        )
        if self.optimizer_scheduler_checkpoint is None:
            self.optimizer, self.lr_scheduler, _ = self.scheduler.prepare_training_step(
                optimizer=None, lr_scheduler=None, **step
            )
        else:
            check_torch_load_is_safe()
            self.optimizer, self.lr_scheduler, _ = self.scheduler.resume(
                checkpoint_dir=self.optimizer_scheduler_checkpoint, **step
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        # Built together with the optimizer; prepared at the first training step.
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        # Restored by the staged scheduler in 'create_optimizer'.
        pass

    def _prepare_training_step(self, model):
        """Apply the staged scheduler for the next optimizer step and return the training model."""
        self.optimizer, self.lr_scheduler, requires_grad_changed = (
            self.scheduler.prepare_training_step(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                global_step=self.state.global_step,
                max_steps=self.state.max_steps,
            )
        )
        if requires_grad_changed:
            # DDP only reduces gradients of parameters that were trainable when it was built.
            model = self.model_wrapped = self.accelerator.prepare(self.model)
        if not getattr(self.optimizer, "_is_accelerate_prepared", False):
            self.optimizer = self.accelerator.prepare(self.optimizer)
        if not getattr(self.lr_scheduler, "_is_accelerate_prepared", False):
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
            # Offset schedulers derive their learning rate from the global step.
            self.lr_scheduler.step(self.state.global_step)
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        return model

    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        """Copy of transformers 5.19.0 'Trainer._run_epoch'; changes are marked with 'VLM:'."""

        # VLM: '_inner_training_loop' passes the model it wrapped first, before any re-wrapping.
        model = self.model_wrapped

        step = -1
        grad_norm = None
        learning_rate = None
        rng_to_sync = False

        # Handle resumption from checkpoint: skip already-trained batches in the resumed epoch
        num_update_steps_trained = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
                train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                step = steps_trained_in_current_epoch - 1
                num_update_steps_trained = steps_trained_in_current_epoch // self.args.gradient_accumulation_steps
                rng_to_sync = True
            elif steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        # `DataLoaderShard.set_epoch` (accelerate) does not forward to a batch sampler wrapped
        # in `BatchSamplerShard`, so reach it explicitly.
        shard_wrapper = getattr(train_dataloader, "batch_sampler", None)
        underlying_sampler = getattr(shard_wrapper, "batch_sampler", None)
        if hasattr(underlying_sampler, "set_epoch"):
            underlying_sampler.set_epoch(epoch)
        epoch_iterator = iter(train_dataloader)

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % self.args.gradient_accumulation_steps
        if remainder == 0:
            remainder = self.args.gradient_accumulation_steps

        # Outer loop: one iteration per optimizer step. Each iteration prefetches
        # `gradient_accumulation_steps` batches (fewer for the last step if the epoch
        # doesn't divide evenly).
        for update_step in range(num_update_steps_trained, num_update_steps_per_epoch):
            num_batches = (
                self.args.gradient_accumulation_steps if update_step != (num_update_steps_per_epoch - 1) else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, self.args.device)

            # This is used to correctly scale the loss when the last accumulation step has fewer batches.
            # Not used if `num_items_in_batch` is not None.
            self.current_gradient_accumulation_steps = len(batch_samples)

            # need to sync after if we skipped the batches in `get_batch_samples` for shuffle order reason
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Inner loop: forward + backward for each micro-batch. Gradients are
            # accumulated without syncing until the last micro-batch, then we clip,
            # step the optimizer, and log/save/evaluate.
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if step % self.args.gradient_accumulation_steps == 0:
                    # VLM: let the staged scheduler configure this optimizer step.
                    model = self._prepare_training_step(model)
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # We sync the gradients in the following cases: 1. sync_each_batch set to True 2. Using deepspeed 3. when we are at the last batch sample
                if (
                    self.accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False)
                    or self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    or i == len(batch_samples) - 1
                ):
                    sync_context = contextlib.nullcontext
                else:
                    sync_context = functools.partial(self.accelerator.no_sync, model=model)
                with sync_context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (
                    self.args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    self._tr_loss += self._tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if self._tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {self._tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    self._tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self._track_num_input_tokens(inputs)

                if do_sync_step:
                    grad_norm = None
                    if self.args.max_grad_norm > 0:
                        grad_norm = self._clip_grad_norm(model)
                    grad_norm = self._get_grad_norm(model, grad_norm=grad_norm)

                    self.control = self.callback_handler.on_pre_optimizer_step(self.args, self.state, self.control)
                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(self.args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # VLM: offset schedulers derive their learning rate from the global step.
                        self.lr_scheduler.step(self.state.global_step)

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        self._tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(self.args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        # VLM: removed the PyTorch/XLA graph flush.

        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({self.state.max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(
            self._tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )
