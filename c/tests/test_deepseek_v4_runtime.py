"""Run one deterministic token through a tiny handcrafted V4 container."""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / ("deepseek_v4.exe" if os.name == "nt" else "deepseek_v4")


def _safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _f32(values):
    return struct.pack("<" + "f" * len(values), *values)


class DeepSeekV4RuntimeTest(unittest.TestCase):
    @unittest.skipUnless(
        ENGINE.exists() and os.environ.get("COLIBRI_RUN_FIXTURE_TEST") == "1",
        "opt-in tiny runtime test (set COLIBRI_RUN_FIXTURE_TEST=1)",
    )
    def test_tiny_container(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = {
                "model_type": "deepseek_v4", "vocab_size": 8, "dim": 4,
                "moe_inter_dim": 2, "n_layers": 1, "n_hash_layers": 0,
                "n_heads": 1, "n_routed_experts": 1, "n_shared_experts": 1,
                "n_activated_experts": 1, "score_func": "sqrtsoftplus",
                "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 2,
                "head_dim": 4, "rope_head_dim": 2, "o_groups": 1,
                "o_lora_rank": 2, "window_size": 2, "compress_ratios": [0],
                "hc_mult": 1, "hc_sinkhorn_iters": 2, "hc_eps": 1e-6,
                "norm_eps": 1e-6, "eos_token_id": 7,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "tokenizer.json").write_text(json.dumps({
                "model": {"type": "BPE", "vocab": {chr(97 + i): i for i in range(8)}, "merges": []},
                "added_tokens": [],
            }), encoding="utf-8")

            tensors = []

            def f(name, shape):
                tensors.append((name, "F32", shape, _f32([0.0] * (shape[0] if len(shape) == 1 else shape[0] * shape[1]))))

            def q(name, rows, cols, fmt="int4"):
                if fmt == "int8":
                    tensors.append((name, "U8", (rows, cols), bytes(rows * cols)))
                    tensors.append((name + ".scale", "F32", (rows,), _f32([1.0] * rows)))
                else:
                    tensors.append((name, "U8", (rows, (cols + 1) // 2), bytes(rows * ((cols + 1) // 2))))
                    tensors.append((name + ".scale", "F32", (rows, 1), _f32([1.0] * rows)))

            q("embed.weight", 8, 4, "int8")
            f("norm.weight", (4,))
            q("head.weight", 8, 4, "int8")
            f("layers.0.attn_norm.weight", (4,)); f("layers.0.ffn_norm.weight", (4,))
            q("layers.0.attn.wq_a.weight", 2, 4); f("layers.0.attn.q_norm.weight", (2,))
            q("layers.0.attn.wq_b.weight", 4, 2); q("layers.0.attn.wkv.weight", 4, 4)
            f("layers.0.attn.kv_norm.weight", (4,)); q("layers.0.attn.wo_a.weight", 2, 4)
            q("layers.0.attn.wo_b.weight", 4, 2); f("layers.0.attn.attn_sink", (1,))
            q("layers.0.ffn.gate.weight", 1, 4)
            q("layers.0.ffn.shared_experts.w1.weight", 2, 4)
            q("layers.0.ffn.shared_experts.w2.weight", 4, 2)
            q("layers.0.ffn.shared_experts.w3.weight", 2, 4)
            # One zero FP4 expert: packed E2M1 bytes plus one UE8M0 scale/row.
            for name, rows, cols in (("w1", 2, 4), ("w2", 4, 2), ("w3", 2, 4)):
                tensors.append((f"layers.0.ffn.experts.0.{name}.weight", "U8",
                                (rows, (cols + 1) // 2), bytes(rows * ((cols + 1) // 2))))
                tensors.append((f"layers.0.ffn.experts.0.{name}.scale", "U8",
                                (rows, 1), bytes([127] * rows)))
            _safetensors(root / "model-00000-of-00001.safetensors", tensors)

            env = dict(os.environ, SNAP=str(root), PROMPT="", NGEN="1", CTX="16")
            result = subprocess.run([str(ENGINE), "0"], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("a", result.stdout)


if __name__ == "__main__":
    unittest.main()
