from __future__ import annotations

import unittest
from pathlib import Path

from ires_design.baselines import DEFAULT_REGISTRY, load_registry, registry_by_id
from ires_design.experiments import load_and_validate_experiment


ROOT = Path(__file__).resolve().parents[1]


class BaselineRegistryTests(unittest.TestCase):
    def test_registry_has_unique_expected_ids(self) -> None:
        specs = load_registry(DEFAULT_REGISTRY)
        ids = [spec.id for spec in specs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(
            {"official_rfamllama", "generna", "ires_ea", "ires_dm", "nsga2"} <= set(ids)
        )

    def test_external_reference_role_is_explicit(self) -> None:
        registry = registry_by_id()
        self.assertEqual(registry["ires_dm"].role, "external_reference")
        self.assertEqual(registry["local_rfam_ar_reproduction"].role, "supplementary_reproduction")

    def test_experiment_contracts_validate(self) -> None:
        config_dir = ROOT / "configs" / "experiments"
        for path in sorted(config_dir.glob("*.json")):
            with self.subTest(path=path.name):
                payload = load_and_validate_experiment(path)
                self.assertTrue(payload["experiment_id"])


if __name__ == "__main__":
    unittest.main()

