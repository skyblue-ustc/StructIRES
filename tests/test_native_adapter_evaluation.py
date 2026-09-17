from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_native_adapter_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("native_adapter_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeAdapterEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.run_dir = Path("/example/external-runs/example-dev-run")
        cls.manifest = {
            "variant": "pretrained_deep_adapter",
            "test_evaluation_skipped": True,
            "test_labels_used_for_selection": False,
        }

    def valid_lock(self) -> dict[str, object]:
        return {
            "status": "selected_for_formal_evaluation",
            "selection_complete_before_test": True,
            "selected": {"trainer_variant": "pretrained_deep_adapter"},
            "source_dev_run_dirs": [str(self.run_dir)],
        }

    def test_accepts_frozen_matching_development_run(self) -> None:
        self.module.validate_lock(self.valid_lock(), self.run_dir, self.manifest)

    def test_formal_evaluator_supports_internal_fusion_candidates(self) -> None:
        expected = {
            "pretrained_separated_deep_adapter",
            "pretrained_refined_deep_adapter",
            "dual_pretrained_penultimate_warmstart",
            "cross_gated_penultimate_adapter",
        }
        self.assertTrue(expected <= set(self.module.SUPPORTED_VARIANTS))

    def test_rejects_unfrozen_lock(self) -> None:
        lock = self.valid_lock()
        lock["status"] = "development_validation_only"
        with self.assertRaisesRegex(ValueError, "not frozen"):
            self.module.validate_lock(lock, self.run_dir, self.manifest)

    def test_rejects_run_not_declared_before_test(self) -> None:
        lock = self.valid_lock()
        lock["source_dev_run_dirs"] = []
        with self.assertRaisesRegex(ValueError, "not included"):
            self.module.validate_lock(lock, self.run_dir, self.manifest)

    def test_rejects_source_that_already_scored_test(self) -> None:
        manifest = dict(self.manifest)
        manifest["test_evaluation_skipped"] = False
        with self.assertRaisesRegex(ValueError, "development-only"):
            self.module.validate_lock(self.valid_lock(), self.run_dir, manifest)

    def test_accepts_predeclared_sequence_comparator(self) -> None:
        lock = self.valid_lock()
        lock["formal_comparators"] = [{"trainer_variant": "sequence"}]
        manifest = dict(self.manifest)
        manifest["variant"] = "sequence"
        self.module.validate_lock(lock, self.run_dir, manifest)

    def test_rejects_unlisted_comparator(self) -> None:
        manifest = dict(self.manifest)
        manifest["variant"] = "sequence"
        with self.assertRaisesRegex(ValueError, "not locked"):
            self.module.validate_lock(self.valid_lock(), self.run_dir, manifest)


if __name__ == "__main__":
    unittest.main()
