import torch
import torch.nn as nn
from torch.optim import Adam, Muon, Optimizer
from torch.optim._muon import muon
from torch.optim.adam import adam

DEFAULT_ADAMW_NAME_PATTERNS = ["embed", "lm_head"]


class MuonWithAdamW(Optimizer):
    """
    Muon for the 2D weight matrices, AdamW for everything else, in a single optimizer.

    ``torch.optim.Muon`` rejects anything that is not 2D, and even among the 2D parameters the
    embedding tables and the output head are conventionally left to AdamW (their rows are
    looked up / scored independently, so orthogonalising the whole matrix makes no sense).
    Each param group therefore carries a ``use_muon`` flag and ``step`` dispatches it to torch's
    own Muon or AdamW kernel -- the maths are torch's, not a re-implementation.

    The groups must already be split when they reach ``__init__``: use ``split_param_groups``
    (``SchedulerBase._get_param_groups`` does). Splitting in here instead would leave
    ``optimizer.param_groups`` longer than the list the LR schedulers were built from, and they
    zip ``max_lrs`` against ``optimizer.param_groups`` by position -- every group after the
    first split would silently get its neighbour's learning rate.

    One ``Optimizer`` rather than a wrapper around a ``Muon`` and an ``AdamW``: the trainer,
    accelerate, checkpoint save/load and the schedulers' ``isinstance`` checks all see a single
    ``param_groups`` / ``state``, and the inherited ``state_dict`` / ``load_state_dict`` work as is.

    ``adjust_lr_fn="match_rms_adamw"`` (torch's default is ``None``) scales each Muon update to
    AdamW's RMS, so the learning rates and weight decay tuned for AdamW carry over unchanged.
    With ``None`` / ``"original"`` Muon wants an lr around 0.02.

    Not compatible with DeepSpeed ZeRO: it hands the optimizer flat 1D shards of the parameters,
    which cannot be orthogonalised. DeepSpeed already refuses the class under ZeRO ("untested
    ZeRO Optimizer"); ``step`` raises if that check was switched off.
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            # Muon
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
            ns_steps: int = 5,
            eps: float = 1e-7,
            adjust_lr_fn: str | None = "match_rms_adamw",
            # AdamW
            betas: tuple[float, float] = (0.9, 0.999),
            adamw_eps: float = 1e-8,
            amsgrad: bool = False,
            foreach: bool | None = None,
            fused: bool | None = None,
            # Only read by split_param_groups; accepted here so it can live in the optimizer's
            # class_kwargs, which the config validates against this signature.
            adamw_name_patterns: list[str] = None,
    ):
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError(f"adjust_lr_fn={adjust_lr_fn!r} is not supported by torch.optim.Muon")
        if fused and foreach:
            raise ValueError("`fused` and `foreach` cannot be `True` together.")

        muon_defaults = dict(
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=tuple(ns_coefficients),
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        adamw_defaults = dict(
            betas=tuple(betas),
            eps=adamw_eps,
            amsgrad=amsgrad,
            maximize=False,
            foreach=foreach,
            capturable=False,
            differentiable=False,
            fused=fused,
            decoupled_weight_decay=True,
        )

        param_groups = list(params)
        if param_groups and not isinstance(param_groups[0], dict):
            raise TypeError(
                "MuonWithAdamW needs param groups carrying a 'use_muon' flag; build them with "
                "MuonWithAdamW.split_param_groups."
            )
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError(
                    f"param group {group.get('name', '<unnamed>')!r} has no 'use_muon' flag; "
                    "build the groups with MuonWithAdamW.split_param_groups."
                )
            # Only the defaults of the group's own algorithm: both have an 'eps', with
            # different meanings and magnitudes.
            for key, value in (muon_defaults if group["use_muon"] else adamw_defaults).items():
                group.setdefault(key, value)

        super().__init__(param_groups, dict(lr=lr, weight_decay=weight_decay))

    @classmethod
    def split_param_groups(
            cls,
            param_groups: list[dict],
            model: nn.Module,
            adamw_name_patterns: list[str] | None = None,
    ) -> list[dict]:
        """
        Split every group into a ``<name>_MUON`` group (2D, name matching none of
        ``adamw_name_patterns``) and a ``<name>_ADAMW`` group (the rest), keeping each group's
        other keys (lr, weight_decay). Empty halves are dropped. Deterministic, so the same
        trainable set always gives the same group count -- which checkpoint resumes depend on.
        """
        if adamw_name_patterns is None:
            adamw_name_patterns = DEFAULT_ADAMW_NAME_PATTERNS
        param_names = {id(p): n for n, p in model.named_parameters()}

        split_groups = []
        for group in param_groups:
            muon_params, adamw_params = [], []
            for p in group["params"]:
                name = param_names.get(id(p), "")
                if p.ndim == 2 and not any(pattern in name for pattern in adamw_name_patterns):
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            base_name = group.get("name", f"group{len(split_groups)}")
            for suffix, use_muon, params in (("MUON", True, muon_params), ("ADAMW", False, adamw_params)):
                if params:
                    split_groups.append({
                        **group,
                        "name": f"{base_name}_{suffix}",
                        "params": params,
                        "use_muon": use_muon,
                    })
        return split_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict):
        params_with_grad, grads, momentum_bufs = [], [], []
        for p in group["params"]:
            if p.ndim != 2:
                raise RuntimeError(
                    f"MuonWithAdamW got a {p.ndim}D tensor of shape {tuple(p.shape)} in Muon group "
                    f"{group.get('name')!r}. Under DeepSpeed ZeRO the optimizer only sees flattened "
                    "shards, which Muon cannot orthogonalise -- run without --trainer.deepspeed."
                )
        # Muon._init_group only touches self.state and the group.
        has_complex = Muon._init_group(self, group, params_with_grad, grads, momentum_bufs)
        muon(
            params_with_grad,
            grads,
            momentum_bufs,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
            adjust_lr_fn=group["adjust_lr_fn"],
            has_complex=has_complex,
        )

    def _adamw_step(self, group: dict):
        params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], [], [], []
        # Same as Muon above: Adam._init_group only touches self.state and the group.
        has_complex = Adam._init_group(
            self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps
        )
        beta1, beta2 = group["betas"]
        adam(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=group["amsgrad"],
            has_complex=has_complex,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )
