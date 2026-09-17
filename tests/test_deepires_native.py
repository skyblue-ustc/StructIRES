from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_deepires_native.py"
    spec = importlib.util.spec_from_file_location("train_deepires_native", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepIRESNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_encode_sequences_normalizes_and_crops(self) -> None:
        tokens, lengths = self.module.encode_sequences(["ACTGN", "UUUUUU"], max_length=4)
        self.assertEqual(tokens.tolist(), [[1, 2, 4, 3], [4, 4, 4, 4]])
        self.assertEqual(lengths.tolist(), [4, 4])

    def test_official_pre_padding_and_right_crop(self) -> None:
        tokens, lengths = self.module.encode_sequences(
            ["AC", "ACGUUA"], max_length=4, crop="right", padding="pre"
        )
        self.assertEqual(tokens.tolist(), [[0, 0, 1, 2], [3, 4, 4, 1]])
        self.assertEqual(lengths.tolist(), [2, 4])

    def test_unmasked_fixed_length_forward(self) -> None:
        model = self.module.DeepIRES(dropout=0.0, unmasked_fixed_length=True).eval()
        tokens, lengths = self.module.encode_sequences(
            ["AC", "GGUU"], max_length=8, padding="pre"
        )
        with torch.no_grad():
            logits = model(tokens, lengths)
        self.assertEqual(logits.shape, (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_deepires_forward_shape_and_finite_values(self) -> None:
        model = self.module.DeepIRES(dropout=0.0).eval()
        tokens, lengths = self.module.encode_sequences(["ACGUAC", "GGUU"], max_length=8)
        with torch.no_grad():
            representation = model.encode(tokens, lengths)
            logits = model(tokens, lengths)
        self.assertEqual(representation.shape, (2, 16))
        self.assertEqual(logits.shape, (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_focal_loss_is_finite(self) -> None:
        logits = torch.tensor([-1.0, 1.0])
        labels = torch.tensor([0.0, 1.0])
        loss = self.module.binary_focal_loss(logits, labels, alpha=0.25, gamma=2.0)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
