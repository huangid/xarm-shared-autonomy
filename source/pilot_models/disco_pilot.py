"""DiSCo copilot: BC_Pilot + DiSCo sampling (seed/inpaint) + beta blend."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from torch import Tensor
from residual_copilot.pilot_models.bc_pilot import BC_Pilot



class DiSCo_Pilot(BC_Pilot):
    def __init__(self, model_id: str, device: Optional[str] = None,
                gamma: float = 0.3, rho: float = 1.0, beta: float = 0.6):
        super().__init__(model_id, device=device)
        self.gamma, self.rho, self.beta = gamma, rho, beta
        dm = self.policy.diffusion
        dm._use_disco = True
        dm._disco_gamma = gamma
        dm._disco_rho = rho
        self._prev_ref = None

    def act(self, obs: Dict[str, Any], ref_action: Tensor | None = None, is_idle: bool = False):
        processed = self.preprocessor(obs)
        ref_norm = self.ref_action_processor(ref_action) if ref_action is not None else None
        with torch.inference_mode():
            action_tensor = self.policy.select_action(processed, ref_action=ref_norm)
        out = self.postprocessor(action_tensor).to(self.device)
        if ref_action is not None and self.beta > 0.0:
            u = ref_action.to(out.device, out.dtype)
            beta_eff = 1.0 if is_idle else self.beta   # idle -> trust user fully -> hold still
            if u.shape == out.shape:
                out = beta_eff * u + (1.0 - beta_eff) * out
        return out


__all__ = ["DiSCo_Pilot"]