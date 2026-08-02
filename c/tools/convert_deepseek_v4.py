#!/usr/bin/env python3
"""Convert a DeepSeek-V4-Flash Hugging Face snapshot to a Colibri CPU layout.

The converter deliberately keeps the original QAT FP4 routed experts.  The
official checkpoint stores those weights as packed E2M1 nibbles with one
UE8M0 scale per 32 input values.  Large non-expert matrices are converted to
the existing Colibri grouped-int4 container; small control tensors stay in
float32 and embeddings/head use per-row int8.

Only the conversion step needs torch/safetensors.  The resulting runtime
directory contains U8/I8/F32/BF16 tensors understood by the dependency-free C
engine.  Conversion is shard-by-shard and resumable; no model artifact is
created in the repository by this tool.

Usage:
    python tools/convert_deepseek_v4.py \
        --indir /models/DeepSeek-V4-Flash-0731 \
        --outdir /nvme/dsv4_colibri

The optional --include-dspark flag keeps MTP/DSpark tensors in a separate
sidecar directory.  The base model never loads that sidecar by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ARCH = "deepseek_v4"
DEFAULT_GROUP = 64
DEFAULT_MAX_GB = 170.0


def _require_deps():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        from safetensors.torch import safe_open, save_file  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise SystemExit(
            "DeepSeek-V4 conversion requires numpy, torch and safetensors; "
            "install them in the conversion environment only (runtime is pure C)."
        ) from exc


def _source_files(indir: Path) -> List[Path]:
    files = sorted(indir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors shards found in {indir}")
    return files


def _layer_index(name: str) -> int:
    m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else -1


def _is_expert(name: str) -> bool:
    return ".experts." in name and any(
        x in name for x in ("gate_proj", "up_proj", "down_proj", ".w1", ".w2", ".w3")
    )


def _is_dspark(name: str) -> bool:
    low = name.lower()
    return low.startswith("mtp.") or low.startswith("model.mtp.") or "dspark" in low


def _normalise_name(name: str) -> str:
    """Map official names to the compact names used by the C engine."""
    if name.startswith("model."):
        name = name[len("model.") :]
    name = name.replace("self_attn", "attn").replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")

    # Keep the layer index and expert index visible to the streaming loader.
    replacements = (
        ("embed_tokens", "embed"),
        ("input_layernorm", "attn_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("q_a_proj", "wq_a"),
        ("q_a_layernorm", "q_norm"),
        ("q_b_proj", "wq_b"),
        ("q_proj", "wq"),
        ("kv_a_proj_with_mqa", "wkv_a"),
        ("kv_a_layernorm", "kv_norm"),
        ("kv_b_proj", "wkv_b"),
        ("o_a_proj", "wo_a"),
        ("o_b_proj", "wo_b"),
        ("o_proj", "wo"),
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
        ("lm_head", "head"),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def _key_without_suffix(name: str) -> str:
    for suffix in (".weight_scale_inv", ".weight_scale", ".scale", ".weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _find_scale(keys: set[str], weight_name: str) -> Optional[str]:
    """Accept the naming variants used by official and preview snapshots."""
    candidates = [
        weight_name.replace(".weight", ".weight_scale_inv"),
        weight_name.replace(".weight", ".weight_scale"),
        weight_name.replace(".weight", ".scale"),
    ]
    for candidate in candidates:
        if candidate in keys:
            return candidate
    # Some converted snapshots expose a bare ``scale`` next to ``w1`` etc.
    stem = _key_without_suffix(weight_name)
    for candidate in (stem + ".scale", stem + ".weight_scale"):
        if candidate in keys:
            return candidate
    return None


def _dtype_name(tensor) -> str:
    return str(tensor.dtype).lower()


def _raw_u8(tensor):
    import torch

    # ``view`` is important for torch.float8_e8m0fnu: converting that dtype to
    # uint8 converts the numeric value, while the official FP4 scale is the
    # raw exponent byte and must be preserved bit-for-bit.
    if tensor.element_size() == 1:
        return tensor.view(torch.uint8).contiguous()
    return tensor.to(torch.uint8).contiguous()


def decode_ue8m0(tensor):
    """Decode an unsigned E8M0 power-of-two scale to float32."""
    import torch

    raw = _raw_u8(tensor).to(torch.int16)
    # E8M0 uses exponent bias 127.  Zero is kept as zero for the rare padded
    # or explicitly disabled block; normal model scales are non-zero.
    out = torch.where(raw == 0, torch.zeros_like(raw, dtype=torch.float32),
                      torch.pow(torch.tensor(2.0, dtype=torch.float32), raw.float() - 127.0))
    return out


def _to_f32(tensor, scale=None):
    """Return a source tensor as float32, applying block scale if present."""
    import torch

    kind = _dtype_name(tensor)
    if "float8_e4m3" in kind:
        value = tensor.float()
    elif tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        value = tensor.float()
    elif tensor.dtype in (torch.int8, torch.uint8) and scale is not None:
        # This is the official packed FP4 path only; callers that pass an
        # unscaled integer tensor should use the raw expert path instead.
        value = tensor.float()
    else:
        value = tensor.float()
    if scale is None:
        return value
    if "float8_e8m0" in _dtype_name(scale) or scale.dtype in (torch.uint8, torch.int8):
        scale_f = decode_ue8m0(scale)
    else:
        scale_f = scale.float()
    return _broadcast_block_scale(value, scale_f)


def _broadcast_block_scale(value, scale):
    """Broadcast [ceil(O/128),ceil(I/128)] scales over a 2-D matrix."""
    import torch

    if value.ndim != 2 or scale.numel() == 1:
        return value * scale.reshape(-1)[0]
    out_dim, in_dim = value.shape
    so, si = scale.shape[-2], scale.shape[-1]
    expanded = scale.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    return value * expanded[:out_dim, :in_dim]


def _dequant_source(f, name: str, keys: set[str]):
    tensor = f.get_tensor(name)
    scale_name = _find_scale(keys, name)
    scale = f.get_tensor(scale_name) if scale_name else None
    dtype = _dtype_name(tensor)
    # Experts are handled separately and must remain byte-identical FP4.
    if "float8_e4m3" in dtype:
        if scale is None:
            raise ValueError(f"{name}: FP8 tensor has no UE8M0/F32 scale sidecar")
        return _to_f32(tensor, scale)
    # ``wo_a`` arrives as block-shaped FP8 with a scale; official conversion
    # first dequantizes it and then flattens it to BF16.
    if scale is not None and tensor.ndim == 2 and tensor.shape != scale.shape:
        return _to_f32(tensor, scale)
    return tensor.float()


def _pad_last_dim(array, multiple: int):
    import numpy as np

    n, d = array.shape
    padded = ((d + multiple - 1) // multiple) * multiple
    if padded == d:
        return array, d
    result = np.zeros((n, padded), dtype=np.float32)
    result[:, :d] = array
    return result, d


def quant_int4_grouped(w, group: int = DEFAULT_GROUP):
    """Pack signed symmetric int4, one float scale per input group."""
    import numpy as np

    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"int4 expects a matrix, got {w.shape}")
    rows, original_i = w.shape
    padded, _ = _pad_last_dim(w, group)
    groups = padded.shape[1] // group
    g = padded.reshape(rows, groups, group)
    scale = np.maximum(np.abs(g).max(axis=2) / np.float32(7.0), np.float32(1e-20))
    q = np.clip(np.rint(g / scale[:, :, None]), -8, 7).astype(np.int16) + 8
    q = q.reshape(rows, padded.shape[1])[:, :original_i]
    packed = np.zeros((rows, (original_i + 1) // 2), dtype=np.uint8)
    packed[:, : q[:, 0::2].shape[1]] |= q[:, 0::2].astype(np.uint8)
    if original_i > 1:
        packed[:, : q[:, 1::2].shape[1]] |= q[:, 1::2].astype(np.uint8) << 4
    return packed, scale.astype(np.float32)


def quant_int8(w):
    import numpy as np

    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"int8 expects a matrix, got {w.shape}")
    scale = np.maximum(np.abs(w).max(axis=1) / np.float32(127.0), np.float32(1e-20))
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q.view(np.uint8).copy(), scale.astype(np.float32)


def _torch_u8(array):
    import torch

    return torch.from_numpy(array.astype("uint8", copy=False)).contiguous()


def _torch_f32(array):
    import torch
    import numpy as np

    return torch.from_numpy(np.asarray(array, dtype=np.float32)).contiguous()


def _classify(name: str, config: dict) -> str:
    low = name.lower()
    if _is_dspark(name):
        return "dspark"
    if _is_expert(name):
        return "expert"
    if low.endswith("e_score_correction_bias") or low.endswith("bias"):
        return "f32"
    if any(x in low for x in ("norm", "hc_", ".hc", "tie2eid", "ape", "attn_sink")):
        return "f32"
    # The reference implementation deliberately keeps the learned CSA
    # compressor (and the indexer used to choose ratio-4 blocks) in BF16.  A
    # second int4 pass here noticeably changes the compressed-memory routing;
    # retaining these relatively small matrices is worthwhile for the quality
    # target and still fits the 15 GB dense working set.
    if any(x in low for x in (".compressor.", ".indexer.", "weights_proj")):
        return "f32"
    # The official converter materialises wo_a from its 128x128 FP8 block
    # layout before saving it.  Keep that projection in float32 here too;
    # otherwise a second int4 quantisation would compound the QAT error on the
    # grouped output path.
    if low.endswith("wo_a.weight") or low.endswith("wo_a"):
        return "f32"
    if low.endswith("embed_tokens.weight") or low.endswith("lm_head.weight"):
        return "io"
    if low.endswith(".weight") or low.endswith(".weight_scale_inv") or low.endswith(".weight_scale"):
        return "matrix"
    return "f32"


def _mapped_pair(name: str) -> str:
    return _normalise_name(name)


def _set_tensor(out: Dict[str, object], name: str, value):
    if name in out:
        raise ValueError(f"duplicate output tensor {name}")
    out[name] = value


def convert_shard(src: Path, include_dspark: bool, group: int):
    """Convert one input shard into an in-memory output dictionary."""
    _require_deps()
    import torch
    from safetensors.torch import safe_open

    output: Dict[str, object] = {}
    dspark: Dict[str, object] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for original in sorted(keys):
            if original.endswith(("weight_scale_inv", "weight_scale")):
                # Sidecars are consumed with their weight below.
                continue
            kind = _classify(original, {})
            if kind == "dspark":
                if not include_dspark:
                    continue
                target = dspark
            else:
                target = output
            if not original.endswith(".weight") and kind not in ("f32",):
                # Keep unusual V4 metadata/control tensors, but don't turn a
                # scale sidecar into an independent runtime tensor.
                if original.endswith(".scale"):
                    continue
            mapped = _mapped_pair(original)
            tensor = f.get_tensor(original)

            if kind == "expert":
                scale_name = _find_scale(keys, original)
                if scale_name is None:
                    raise ValueError(f"{original}: missing FP4 UE8M0 scale")
                scale = f.get_tensor(scale_name)
                if tensor.ndim != 2 or tensor.dtype not in (torch.int8, torch.uint8):
                    raise ValueError(f"{original}: expected packed FP4 int8/U8 matrix")
                # Official layout is [O, I/2] bytes and [O, I/32] UE8M0.
                _set_tensor(target, mapped.replace(".weight", ".weight"), _raw_u8(tensor))
                _set_tensor(target, mapped.replace(".weight", ".scale"), _raw_u8(scale))
                continue

            if original.endswith(".weight"):
                f32 = _dequant_source(f, original, keys)
                if f32.ndim == 4 and original.endswith("wo_a.weight"):
                    f32 = f32.reshape(f32.shape[0] * f32.shape[1], f32.shape[2] * f32.shape[3])
                if f32.ndim == 2 and kind == "io":
                    q, scale = quant_int8(f32.cpu().numpy())
                    _set_tensor(target, mapped.replace(".weight", ".weight"), _torch_u8(q))
                    _set_tensor(target, mapped.replace(".weight", ".scale"), _torch_f32(scale))
                elif f32.ndim == 2 and kind == "matrix":
                    q, scale = quant_int4_grouped(f32.cpu().numpy(), group)
                    _set_tensor(target, mapped.replace(".weight", ".weight"), _torch_u8(q))
                    _set_tensor(target, mapped.replace(".weight", ".scale"), _torch_f32(scale))
                else:
                    _set_tensor(target, mapped, f32.float().contiguous())
            else:
                # Biases, routing tables and scalar metadata are retained as
                # float32.  The runtime safetensors reader intentionally has a
                # small dtype surface, so integer control tables are converted
                # to F32; token/expert IDs in this model are far below the exact
                # integer range of float32 and the C loader can validate them.
                if tensor.dtype in (torch.int64, torch.int32, torch.int16,
                                    torch.uint32, torch.uint16):
                    _set_tensor(target, mapped, tensor.float().contiguous())
                else:
                    _set_tensor(target, mapped, tensor.contiguous())
    return output, dspark


def _save_shard(path: Path, tensors: Dict[str, object], metadata: dict):
    _require_deps()
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(tmp), metadata={k: str(v) for k, v in metadata.items()})
    os.replace(tmp, path)


def _build_index(outdir: Path, prefix: str = "model-"):
    _require_deps()
    from safetensors import safe_open

    files = sorted(outdir.glob(f"{prefix}*.safetensors"))
    weight_map: Dict[str, str] = {}
    total_size = 0
    for path in files:
        total_size += path.stat().st_size
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in weight_map:
                    raise ValueError(f"duplicate tensor across output shards: {name}")
                weight_map[name] = path.name
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with (outdir / "model.safetensors.index.json").open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return total_size, weight_map


def _copy_metadata(indir: Path, outdir: Path):
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        src = indir / name
        if src.exists():
            shutil.copy2(src, outdir / name)


def convert(indir: Path, outdir: Path, include_dspark: bool, group: int, max_gb: float):
    _require_deps()
    indir = indir.resolve()
    outdir = outdir.resolve()
    if not indir.is_dir():
        raise SystemExit(f"input directory does not exist: {indir}")
    config_path = indir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "deepseek_v4":
        raise SystemExit(f"expected model_type=deepseek_v4, got {config.get('model_type')!r}")
    if group <= 0 or group % 32:
        raise SystemExit("--group must be a positive multiple of 32")
    outdir.mkdir(parents=True, exist_ok=True)
    _copy_metadata(indir, outdir)

    sources = _source_files(indir)
    total = len(sources)
    manifest = {
        "format_version": 1,
        "architecture": ARCH,
        "expert_format": "mxfp4-e2m1-ue8m0-g32",
        "dense_format": f"int4-g{group}",
        "io_format": "int8-row",
        "context_limit": 50000,
        "dspark": "sidecar" if include_dspark else "omitted",
        "source_shards": len(sources),
    }
    (outdir / "colibri_v4_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    dspark_dir = outdir / "dspark"
    for num, src in enumerate(sources):
        dst = outdir / f"model-{num:05d}-of-{total:05d}.safetensors"
        if dst.exists():
            print(f"[{num + 1}/{total}] {dst.name} exists; skipping")
            continue
        print(f"[{num + 1}/{total}] converting {src.name}", flush=True)
        tensors, dspark = convert_shard(src, include_dspark, group)
        if tensors:
            _save_shard(dst, tensors, {
                "colibri.architecture": ARCH,
                "colibri.expert_format": "mxfp4-e2m1-ue8m0-g32",
                "colibri.dense_format": f"int4-g{group}",
            })
        if include_dspark and dspark:
            dspark_dir.mkdir(parents=True, exist_ok=True)
            _save_shard(dspark_dir / dst.name, dspark, {
                "colibri.architecture": ARCH,
                "colibri.component": "dspark",
            })

    total_size, weight_map = _build_index(outdir)
    if include_dspark and dspark_dir.exists():
        _build_index(dspark_dir)
    size_gb = total_size / 1e9
    print(f"converted {len(weight_map)} tensors: {size_gb:.2f} GB")
    if size_gb > max_gb:
        raise SystemExit(
            f"converted model is {size_gb:.2f} GB, above --max-gb {max_gb:.2f}; "
            "use a larger quality budget or add an explicit lower-bit profile"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", required=True, type=Path, help="official HF snapshot directory")
    ap.add_argument("--outdir", required=True, type=Path, help="Colibri output directory")
    ap.add_argument("--group", type=int, default=DEFAULT_GROUP, help="non-expert int4 group size")
    ap.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB,
                    help="maximum base output size in decimal GB")
    ap.add_argument("--include-dspark", action="store_true",
                    help="write MTP/DSpark tensors into an optional sidecar")
    args = ap.parse_args(argv)
    convert(args.indir, args.outdir, args.include_dspark, args.group, args.max_gb)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
