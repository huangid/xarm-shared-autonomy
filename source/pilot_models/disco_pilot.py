"""DiSCo copilot: BC_Pilot + DiSCo sampling (seed/inpaint) + beta blend."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from torch import Tensor
from residual_copilot.pilot_models.bc_pilot import BC_Pilot
from residual_copilot.pilot_models.knn_pilot import _slerp



class DiSCo_Pilot(BC_Pilot):
    def __init__(self, model_id: str, device: Optional[str] = None,
                gamma: float = 0.3, rho: float = 1.0, beta: float = 0.6):
        super().__init__(model_id, device=device)
        self.gamma, self.rho, self.beta = gamma, rho, beta

    def act(self, obs: Dict[str, Any], ref_action: Tensor | None = None, is_idle: bool = False):
        processed = self.preprocessor(obs)
        ref_norm = self.ref_action_processor(ref_action) if ref_action is not None else None
        with torch.inference_mode():
            # DiSCo (Wang et al., HRI '26): seed the reverse diffusion process by
            # forward-diffusing the pilot's current action, then inpaint the known
            # part of the sequence during the first k_ip of the k_sw reverse steps
            # that get run — see DiffusionModel.conditional_sample for the mechanics.
            action_tensor = self.policy.select_action(
                processed,
                disco_seed=ref_norm,
                disco_gamma=self.gamma,
                disco_rho=self.rho,
            )
        out = self.postprocessor(action_tensor).to(self.device)
        if ref_action is not None and self.beta > 0.0:
            u = ref_action.to(out.device, out.dtype)
            beta_eff = 1.0 if is_idle else self.beta   # idle -> trust user fully -> hold still
            if u.shape != out.shape:
                raise ValueError(f"DiSCo blend shape mismatch: user {u.shape} vs policy {out.shape}")
            # Blend pos [0:3] and gripper [7:8] linearly; SLERP the quat [3:7].
            blended = beta_eff * u + (1.0 - beta_eff) * out
            t = out.new_full(out.shape[:-1], beta_eff)              # weight toward user as beta -> 1
            blended[..., 3:7] = _slerp(out[..., 3:7], u[..., 3:7], t)  # q0=policy, q1=user
            out = blended
        return out


__all__ = ["DiSCo_Pilot"]