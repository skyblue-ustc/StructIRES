from __future__ import annotations

import json
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
            {
                "ires_ea",
                "ires_dm",
                "ires_dm_retrained",
                "structure_only_mutation",
                "score_only_ga",
                "random_screen",
                "nsga2",
            }
            <= set(ids)
        )

    def test_external_reference_role_is_explicit(self) -> None:
        registry = registry_by_id()
        self.assertEqual(registry["ires_dm"].role, "external_reference")
        self.assertEqual(registry["ires_dm_retrained"].role, "strict_task_baseline")
        self.assertEqual(registry["official_rfamllama"].role, "proposal_backbone")
        self.assertEqual(registry["local_rfam_ar_reproduction"].role, "supplementary_reproduction")

    def test_experiment_contracts_validate(self) -> None:
        config_dir = ROOT / "configs" / "experiments"
        for path in sorted(config_dir.glob("*.json")):
            with self.subTest(path=path.name):
                payload = load_and_validate_experiment(path)
                self.assertTrue(payload["experiment_id"])

    def test_evaluators_preserve_assay_context(self) -> None:
        path = ROOT / "configs" / "evaluators.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluators = {row["id"]: row for row in payload["evaluators"]}
        self.assertIn("DNA/lentiviral", evaluators["ires_lm_legacy_dna"]["assay_context"])
        self.assertIn("direct RNA", evaluators["ires_trappr_rna"]["assay_context"])
        self.assertTrue(all(row.get("assay_context") for row in evaluators.values()))

    def test_research_manifests_are_valid_json(self) -> None:
        manifest_dir = ROOT / "assets" / "manifests"
        for name in ("literature_sources.json", "ires_trappr_supplement.json"):
            with self.subTest(name=name):
                payload = json.loads((manifest_dir / name).read_text(encoding="utf-8"))
                self.assertEqual(payload["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
