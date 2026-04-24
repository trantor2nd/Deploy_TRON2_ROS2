#!/usr/bin/env python3
"""GR00T inference + a deterministic stub for control-path verification.

Two public surfaces:

  - ``GR00TRunner(checkpoint, device, task_text, base_model_path).infer(
        left_wrist_bgr, cam_high_bgr, right_wrist_bgr, state16) -> (K, 16) np.ndarray``
    Loads the LeRobot GR00T checkpoint exactly the way it was saved and runs
    one forward pass per observation.

  - ``build_stub_chunk(base, K, joint_idx, amplitude, period, t_now, step_dt)
        -> (K, 16) np.ndarray``
    Pure function that produces a synthetic chunk anchored at ``base`` with one
    joint oscillating sinusoidally. No GR00T, no LeRobot — used to isolate the
    ROS → buffer → WebSocket → arm path from the model.

All values in the returned chunk match the dataset convention: arm joints in
rad, gripper columns in 0–1 (the downstream controller scales ×100 before
talking to the robot).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# LeRobot / GR00T imports are deferred so the stub path remains importable
# in environments without the lerobot_py310 stack installed.
# ---------------------------------------------------------------------------

def _import_groot():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition
    return {
        "FeatureType": FeatureType,
        "NormalizationMode": NormalizationMode,
        "PolicyFeature": PolicyFeature,
        "GrootConfig": GrootConfig,
        "GrootPolicy": GrootPolicy,
        "PolicyProcessorPipeline": PolicyProcessorPipeline,
        "policy_action_to_transition": policy_action_to_transition,
    }


# ---------------------------------------------------------------------------
# GR00T inference
# ---------------------------------------------------------------------------

class GR00TRunner:
    """Loads the policy + pre/post pipelines and runs a single-step forward pass."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        task_text: str,
        base_model_path: Optional[str],
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.task_text = task_text
        self.base_model_path = base_model_path
        self.log = logging.getLogger("groot")

        mods = _import_groot()
        self._FeatureType = mods["FeatureType"]
        self._NormalizationMode = mods["NormalizationMode"]
        self._PolicyFeature = mods["PolicyFeature"]
        self._GrootConfig = mods["GrootConfig"]
        self._GrootPolicy = mods["GrootPolicy"]
        self._PolicyProcessorPipeline = mods["PolicyProcessorPipeline"]
        self._policy_action_to_transition = mods["policy_action_to_transition"]

        torch.set_grad_enabled(False)
        self.cfg, self.pre, self.post, self.model = self._load()
        self.model.eval().to(self.device)
        self.log.info(f"policy ready on {self.device}: {self.checkpoint}")

    # ------------------------------- loading -------------------------------
    def _resolve_base_model(self) -> str:
        if self.base_model_path:
            return self.base_model_path
        cache = Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots"
        if cache.exists():
            snaps = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                return str(snaps[0])
        raise RuntimeError(
            "No base model found. Set --base-model-path or BASE_MODEL_PATH to a local "
            "snapshot of nvidia/GR00T-N1.5-3B."
        )

    def _load(self):
        ckpt = self.checkpoint
        if (ckpt / "pretrained_model").is_dir():
            ckpt = ckpt / "pretrained_model"

        cfg_path = ckpt / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        cfg_dict = json.loads(cfg_path.read_text())
        cfg_dict.pop("type", None)

        def _to_feature(d):
            return self._PolicyFeature(
                type=self._FeatureType[d["type"]],
                shape=tuple(d.get("shape") or ()),
            )

        if "input_features" in cfg_dict:
            cfg_dict["input_features"] = {k: _to_feature(v) for k, v in cfg_dict["input_features"].items()}
        if "output_features" in cfg_dict:
            cfg_dict["output_features"] = {k: _to_feature(v) for k, v in cfg_dict["output_features"].items()}
        if "normalization_mapping" in cfg_dict:
            cfg_dict["normalization_mapping"] = {
                k: self._NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()
            }

        cfg_dict["base_model_path"] = self._resolve_base_model()
        cfg = self._GrootConfig(**cfg_dict)
        cfg.device = str(self.device)

        overrides = {"device_processor": {"device": str(self.device)}}
        pre = self._PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
        )
        post = self._PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_postprocessor.json",
            overrides=overrides,
            to_transition=self._policy_action_to_transition,
        )
        policy = self._GrootPolicy.from_pretrained(pretrained_name_or_path=ckpt, config=cfg)
        return cfg, pre, post, policy

    # --------------------------- preprocessing -----------------------------
    def _to_img_tensor(self, bgr: np.ndarray, out_hw: Tuple[int, int] = (480, 640)) -> torch.Tensor:
        """BGR HWC uint8 -> (1, 3, out_hw[0], out_hw[1]) float32 in [0, 1] on device."""
        out_h, out_w = out_hw
        if bgr.shape[0] != out_h or bgr.shape[1] != out_w:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    # --------------------------- inference ---------------------------------
    def infer(
        self,
        left_wrist_bgr: np.ndarray,
        cam_high_bgr: np.ndarray,
        right_wrist_bgr: np.ndarray,
        state16: np.ndarray,
    ) -> np.ndarray:
        """Run one forward pass and return a (K, 16) action chunk in dataset units."""
        batch = {
            "observation.images.cam_left_wrist": self._to_img_tensor(left_wrist_bgr),
            "observation.images.cam_high": self._to_img_tensor(cam_high_bgr),
            "observation.images.cam_right_wrist": self._to_img_tensor(right_wrist_bgr),
            "observation.state": torch.from_numpy(state16.astype(np.float32)).unsqueeze(0).to(self.device),
            "task": self.task_text,
        }
        model_in = self.pre(batch)
        model_in = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in model_in.items()}
        with torch.no_grad():
            raw = self.model.predict_action_chunk(model_in)          # (1, K, max_action_dim=32)
        if raw.dim() != 3:
            raise RuntimeError(f"Unexpected action chunk shape: {tuple(raw.shape)}")
        K = raw.shape[1]

        # Per-step postprocessing is REQUIRED with lerobot 0.4.4 +
        # groot_action_unpack_unnormalize_v1: when fed a (1, K, D) tensor, the
        # unpack step collapses the chunk dim and returns (1, env_action_dim).
        # Feeding each step separately preserves the chunk. Verified against
        # dataset GT: MAE/range ≈ 1–3% per joint (test_infer_on_dataset.py).
        steps = []
        for k in range(K):
            single = raw[:, k, :]                                    # (1, max_action_dim)
            out = self.post(single)["action"]                        # (1, env_action_dim=16)
            steps.append(out.reshape(-1))
        action = torch.stack(steps, dim=0)                           # (K, 16)
        return action.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic stub chunk (no GR00T, no LeRobot)
# ---------------------------------------------------------------------------

def build_stub_chunk(
    base: np.ndarray,
    chunk_length: int,
    joint_idx: int,
    amplitude: float,
    period: float,
    t_now: float,
    step_dt: float,
) -> np.ndarray:
    """Tile ``base`` across ``chunk_length`` rows, sinusoidally wiggle one joint.

    ``base`` is the (16,) dataset-order state (gripper already in 0–1). Every
    row equals ``base`` except column ``joint_idx``, which oscillates by
    ``±amplitude`` rad at ``period`` seconds per cycle. Intended to be played
    back via ``controller.WSController.play_chunk``.
    """
    chunk = np.tile(base, (chunk_length, 1)).astype(np.float32)
    for k in range(chunk_length):
        t = t_now + k * step_dt
        chunk[k, joint_idx] = base[joint_idx] + amplitude * float(np.sin(2 * np.pi * t / period))
    return chunk
