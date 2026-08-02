"""Converter smoke tests which remain dependency-free on the runtime path."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_deepseek_v4", ROOT / "tools" / "convert_deepseek_v4.py"
)
CONVERTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONVERTER)


class DeepSeekV4ConverterTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is conversion-only")
    def test_int4_packing_layout(self):
        import numpy as np

        weights = np.array([[0.0, 1.0, -1.0, 0.5]], dtype=np.float32)
        packed, scale = CONVERTER.quant_int4_grouped(weights, group=32)
        self.assertEqual(packed.shape, (1, 2))
        self.assertEqual(scale.shape, (1, 1))
        self.assertEqual(int(packed[0, 0] & 0x0F), 8)
        self.assertEqual(int(packed[0, 0] >> 4), 15)

    def test_official_name_mapping(self):
        self.assertEqual(
            CONVERTER._normalise_name("model.layers.2.self_attn.q_proj.weight"),
            "layers.2.attn.wq.weight",
        )
        self.assertEqual(
            CONVERTER._normalise_name("model.layers.2.mlp.experts.7.gate_proj.weight"),
            "layers.2.ffn.experts.7.w1.weight",
        )
        self.assertEqual(
            CONVERTER._normalise_name("model.layers.2.self_attn.o_a_proj.weight"),
            "layers.2.attn.wo_a.weight",
        )


if __name__ == "__main__":
    unittest.main()
