#!/usr/bin/env python3
"""Smoke-test GR00T inference against the LeRobot training dataset.

Pulls N samples directly from ``/home/hsb/TRON2_data/pick_stones``, runs them
through the exact same load / preprocess / model / postprocess path that
``deploy.py`` uses, and compares the predicted K-step action chunk against
the ground-truth action[t : t+K] stored in the dataset.

If MAE is small -> inference pipeline is correct; residual deploy-time failure
is on the ROS observation side (wrong units, scrambled joint order, wrong
topics, stale frames).

If MAE is large (especially if predictions hug the dataset mean while GT
swings) -> the loading / pre / post / normalization is still broken.

Run inside the lerobot_py310 conda env. Example:
    python test_infer_on_dataset.py --indices 100,500,2000,10000 --episode 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_policy(checkpoint: Path, device: str, base_model_path: Optional[str]):
    """Replica of ``deploy.py :: GR00TRunner._load`` so this script tests the
    same loading path the deploy loop takes."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition

    ckpt = checkpoint
    if (ckpt / "pretrained_model").is_dir():
        ckpt = ckpt / "pretrained_model"

    cfg_dict = json.loads((ckpt / "config.json").read_text())
    cfg_dict.pop("type", None)

    def _to_feature(d):
        return PolicyFeature(
            type=FeatureType[d["type"]],
            shape=tuple(d.get("shape") or ()),
        )

    cfg_dict["input_features"] = {k: _to_feature(v) for k, v in cfg_dict["input_features"].items()}
    cfg_dict["output_features"] = {k: _to_feature(v) for k, v in cfg_dict["output_features"].items()}
    cfg_dict["normalization_mapping"] = {
        k: NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()
    }

    # Base-model resolution mirrors deploy.py.
    if not base_model_path:
        cache = Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots"
        if cache.exists():
            snaps = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                base_model_path = str(snaps[0])
    if not base_model_path:
        raise RuntimeError(
            "No GR00T base model found. Set BASE_MODEL_PATH or --base-model-path."
        )
    cfg_dict["base_model_path"] = base_model_path

    cfg = GrootConfig(**cfg_dict)
    cfg.device = device

    overrides = {"device_processor": {"device": device}}
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt,
        config_filename="policy_postprocessor.json",
        overrides=overrides,
        to_transition=policy_action_to_transition,
    )
    model = GrootPolicy.from_pretrained(pretrained_name_or_path=ckpt, config=cfg)
    return cfg, pre, post, model


def run_one(pre, post, model, sample, task_text: str, device: torch.device, K: int, verbose: bool = False):
    """Run inference and return BOTH candidate chunk shapes so we can decide
    empirically which unpacking path agrees with GT.

    Returns a dict:
      - 'raw_shape'      : tuple (shape of model.predict_action_chunk output)
      - 'bulk_shape'     : tuple (shape of post(raw) out)
      - 'bulk_pred'      : (k_bulk, 16) np.float32 from one-shot post
      - 'perstep_pred'   : (K, 16) np.float32 from per-step post loop
    """
    batch = {
        "observation.images.cam_left_wrist": sample["observation.images.cam_left_wrist"].unsqueeze(0).to(device),
        "observation.images.cam_high":       sample["observation.images.cam_high"].unsqueeze(0).to(device),
        "observation.images.cam_right_wrist":sample["observation.images.cam_right_wrist"].unsqueeze(0).to(device),
        "observation.state":                 sample["observation.state"].unsqueeze(0).to(device),
        "task": task_text,
    }
    model_in = pre(batch)
    model_in = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in model_in.items()}
    with torch.no_grad():
        raw = model.predict_action_chunk(model_in)        # expected (1, K, action_dim)

    # Path A: feed the whole chunk to postprocessor once.
    bulk_out = post(raw)["action"]

    # Path B: feed each step separately (what the original deploy.py did).
    # Only try if raw actually has a time dim of size K.
    perstep = None
    if raw.dim() == 3 and raw.shape[1] >= 1:
        steps = []
        Ksteps = raw.shape[1]
        for k in range(Ksteps):
            single = raw[:, k, :]                         # (1, action_dim)
            out = post(single)["action"]                  # expected (1, 16)
            steps.append(out.reshape(-1))                 # flatten to (16,) robustly
        perstep = torch.stack(steps, dim=0)               # (Ksteps, 16)

    # Path A -> normalized to (k_bulk, 16)
    if bulk_out.dim() == 3:
        bulk_pred = bulk_out[0]
    elif bulk_out.dim() == 2:
        bulk_pred = bulk_out
    elif bulk_out.dim() == 1:
        bulk_pred = bulk_out.unsqueeze(0)
    else:
        raise RuntimeError(f"unexpected post shape: {tuple(bulk_out.shape)}")

    if verbose:
        print(f"    raw shape      : {tuple(raw.shape)}")
        print(f"    post(raw) shape: {tuple(bulk_out.shape)}   -> bulk_pred {tuple(bulk_pred.shape)}")
        if perstep is not None:
            print(f"    per-step shape : {tuple(perstep.shape)}")
        print(f"    raw[0,0,:4]    : {raw[0, 0, :4].detach().cpu().numpy()}")
        if raw.shape[1] > 1:
            print(f"    raw[0,K-1,:4]  : {raw[0, -1, :4].detach().cpu().numpy()}")

    return {
        "raw_shape": tuple(raw.shape),
        "bulk_shape": tuple(bulk_out.shape),
        "bulk_pred": bulk_pred.detach().cpu().numpy().astype(np.float32),
        "perstep_pred": None if perstep is None else perstep.detach().cpu().numpy().astype(np.float32),
    }


def _mae_report(label: str, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, act_range: np.ndarray):
    """Print MAE stats for one candidate path. Handles pred shorter than gt."""
    k_pred = pred.shape[0]
    k_gt = gt.shape[0]
    k = min(k_pred, k_gt, int(valid.sum()))
    if k == 0:
        print(f"  [{label}] no comparable steps")
        return
    v = valid[:min(k_pred, k_gt)]
    p = pred[:min(k_pred, k_gt)][v]
    g = gt[:min(k_pred, k_gt)][v]
    err = p - g
    mae_per_joint = np.abs(err).mean(axis=0)
    nmae = mae_per_joint / np.maximum(act_range, 1e-6)
    print(f"  [{label}] compared {p.shape[0]} steps | overall MAE = {float(np.abs(err).mean()):.5f}")
    print(f"  [{label}] MAE/joint: {np.array2string(mae_per_joint, precision=4, separator=',')}")
    print(f"  [{label}] MAE/range: {np.array2string(nmae, precision=3, separator=',')}")
    k0 = 0
    print(f"  [{label}] k=0   pred: {np.array2string(p[k0], precision=3, separator=',')}")
    print(f"  [{label}] k=0   gt  : {np.array2string(g[k0], precision=3, separator=',')}")
    if p.shape[0] > 10:
        print(f"  [{label}] k=10  pred: {np.array2string(p[10], precision=3, separator=',')}")
        print(f"  [{label}] k=10  gt  : {np.array2string(g[10], precision=3, separator=',')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path(os.environ.get(
                       "TRON2_CKPT",
                       "/home/data/hf/hub/models--trantor2nd--tron2-pickup-gr00t/"
                       "snapshots/007a769f46218f62e1d5a7e3fb3eb176716b03ce",
                   )))
    p.add_argument("--dataset-root", type=Path, default=Path("/home/hsb/TRON2_data/pick_stones"))
    p.add_argument("--dataset-repo-id", type=str, default="/home/hsb/TRON2_data/pick_stones",
                   help="Passed as repo_id to LeRobotDataset; the path works for local datasets.")
    p.add_argument("--episode", type=int, default=0,
                   help="Only load this episode (fast; avoids fetching all videos).")
    p.add_argument("--indices", type=str, default="30,200,600,1200",
                   help="Comma-separated frame indices WITHIN the chosen episode.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--task-text", type=str,
                   default="pick_up_stones_and_place_them_into_the_container")
    p.add_argument("--base-model-path", type=str, default=os.environ.get("BASE_MODEL_PATH"))
    p.add_argument("--chunk-size", type=int, default=50,
                   help="Action chunk length K; must match cfg.chunk_size in the checkpoint.")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    torch.set_grad_enabled(False)

    # delta_timestamps lets us fetch a K-step action window aligned to each frame.
    # fps=30 per meta/info.json; spacing the deltas at 1/fps gives one chunk per step.
    fps = 30.0
    K = args.chunk_size
    delta_ts = {"action": [i / fps for i in range(K)]}

    print(f"[dataset] loading episode {args.episode} from {args.dataset_root} …")
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        delta_timestamps=delta_ts,
        download_videos=False,   # local dataset, nothing to download
    )
    print(f"[dataset] len={len(ds)} frames, fps={ds.fps}")

    print(f"[policy] loading checkpoint {args.checkpoint} …")
    cfg, pre, post, model = load_policy(args.checkpoint, str(device), args.base_model_path)
    model.eval().to(device)
    print(f"[policy] chunk_size={cfg.chunk_size}, max_action_dim={cfg.max_action_dim}, device={device}")

    if cfg.chunk_size != K:
        print(f"[warn] cfg.chunk_size={cfg.chunk_size} but --chunk-size={K}; continuing with {K}")

    # Stats for contextualizing MAE (per-joint std of the dataset action).
    stats_path = args.dataset_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    act_std = np.asarray(stats.get("action", {}).get("std", [0] * 16), dtype=np.float32)
    act_range = np.asarray(stats.get("action", {}).get("max", [0] * 16), dtype=np.float32) - \
                np.asarray(stats.get("action", {}).get("min", [0] * 16), dtype=np.float32)

    indices = [int(s) for s in args.indices.split(",") if s.strip()]

    for i, t in enumerate(indices):
        if t >= len(ds):
            print(f"[skip] frame {t} >= episode length {len(ds)}")
            continue

        sample = ds[t]
        gt = sample["action"].cpu().numpy().astype(np.float32)      # (K, 16)
        pad_flag = sample.get("action_is_pad")
        if pad_flag is not None and bool(pad_flag.any()):
            n_pad = int(pad_flag.sum())
            print(f"[note] t={t}: {n_pad}/{K} GT steps are padded; metrics on valid only")
            valid = (~pad_flag.bool().cpu().numpy())
        else:
            valid = np.ones(K, dtype=bool)

        # Print shapes for the first sample so we can diagnose.
        result = run_one(pre, post, model, sample, args.task_text, device, K, verbose=(i == 0))

        print(f"\n=== frame t={t} (episode {args.episode}, {int(valid.sum())}/{K} valid steps in GT) ===")
        print(f"  obs.state       : {np.array2string(sample['observation.state'].cpu().numpy(), precision=3, separator=',')}")
        print(f"  gt[0]           : {np.array2string(gt[0], precision=3, separator=',')}")
        if valid.sum() > 10:
            print(f"  gt[10]          : {np.array2string(gt[10], precision=3, separator=',')}")

        _mae_report("bulk   ", result["bulk_pred"], gt, valid, act_range)
        if result["perstep_pred"] is not None:
            _mae_report("perstep", result["perstep_pred"], gt, valid, act_range)

    print(f"\n[ref] action std per joint  : {np.array2string(act_std, precision=3, separator=',')}")
    print(f"[ref] mean(action std)       : {float(act_std.mean()):.4f}")
    print(f"[ref] action range per joint : {np.array2string(act_range, precision=3, separator=',')}")


if __name__ == "__main__":
    # Match deploy.py / oracle_realtime_deploy env so attention kernel and
    # offline mode are identical.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_USE_FLASH_ATTENTION_2", "0")
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "1")
    os.environ.setdefault("ATTENTION_IMPLEMENTATION", "eager")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
