from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_structires_native_contact_fusion.py"
    )
    spec = importlib.util.spec_from_file_location("native_contact_fusion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyRNAFM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 640)
        self.lm = nn.Linear(640, 8)

    def forward(self, tokens: torch.Tensor, layers: list[int]):
        representation = self.embedding(tokens)
        return {
            "representations": {layer: representation for layer in layers},
            "logits": self.lm(representation),
        }


class DummyLayeredRNAFM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(4, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.emb_layer_norm_after = nn.LayerNorm(4)


class DummyEncoder(nn.Module):
    def encode(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        pooled = values.float().reshape(len(values), -1).mean(dim=1, keepdim=True)
        return pooled.repeat(1, 16)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encode(values, lengths)


class DummyPretrainedAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence = DummyEncoder()
        self.sequence.classifier = nn.Sequential(
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(32, 1),
        )
        self.structure = DummyEncoder()
        self.gate = nn.Linear(32, 16)
        self.structure_delta = nn.Linear(16, 16)


class NativeDeepIRESHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_profile_permutation_is_split_local_length_matched_and_deranged(self) -> None:
        sequences = ["A" * length for length in (4, 4, 5, 5, 4, 4, 5, 5, 6)]
        fit = np.asarray([0, 1, 2, 3])
        validation = np.asarray([4, 5, 6, 7])
        test = np.asarray([8])
        mapping = self.module.split_local_profile_permutation(
            sequences, (fit, validation, test), max_length=10, seed=1337
        )
        self.assertTrue(np.array_equal(
            mapping,
            self.module.split_local_profile_permutation(
                sequences, (fit, validation, test), max_length=10, seed=1337
            ),
        ))
        for partition in (fit, validation, test):
            self.assertTrue(set(mapping[partition]).issubset(set(partition)))
        for index, source in enumerate(mapping.tolist()):
            self.assertEqual(len(sequences[index]), len(sequences[source]))
        self.assertTrue(np.all(mapping[np.concatenate((fit, validation))] != np.concatenate((fit, validation))))
        self.assertEqual(mapping[8], 8)

    def test_layerwise_backbone_groups_decay_toward_lower_layers(self) -> None:
        model = self.module.AuthorStyleRNAFM(DummyLayeredRNAFM(), dropout=0.0)
        parameters, _, _ = self.module.configure_trainable_parameters(model, 12)
        groups = self.module.layerwise_backbone_groups(
            model, parameters, base_lr=1e-4, decay=0.5
        )
        learning_rates = {group["role"]: group["lr"] for group in groups}
        self.assertAlmostEqual(learning_rates["backbone_depth_0"], 1.25e-5)
        self.assertAlmostEqual(learning_rates["backbone_depth_1"], 2.5e-5)
        self.assertAlmostEqual(learning_rates["backbone_depth_2"], 5e-5)
        self.assertAlmostEqual(learning_rates["backbone_depth_3"], 1e-4)

    def test_contextual_deepires_forward(self) -> None:
        model = self.module.ContextualDeepIRES(DummyRNAFM(), dropout=0.0).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        with torch.no_grad():
            logits, lm_logits = model.forward_with_lm(tokens)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())

    def test_contextual_deepires_singleton_training_batch(self) -> None:
        model = self.module.ContextualDeepIRES(DummyRNAFM(), dropout=0.0).train()
        tokens = torch.tensor([[0, 4, 5, 6, 7, 2, 1, 1]])
        logits, lm_logits = model.forward_with_lm(tokens)
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(lm_logits.shape, (1, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())

    def test_deep_structure_adapter_forward_and_zero_start(self) -> None:
        model = self.module.DeepStructureAdapter(
            DummyRNAFM(), dropout=0.0, zero_start=True
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 7, 5)
        profile[0, 4:, 4] = 0
        profile[1, :, 4] = 1
        with torch.no_grad():
            logits, lm_logits = model.forward_with_lm(tokens, profile)
            sequence, _, _ = model.sequence_hidden_with_lm(tokens)
            baseline_logits = model.output(sequence)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(logits, baseline_logits))

    def test_hybrid_structure_adapter_forward_and_zero_start(self) -> None:
        model = self.module.HybridDeepIRESStructureAdapter(
            DummyRNAFM(),
            dropout=0.0,
            nucleotide_ids=[4, 5, 6, 7],
            zero_start=True,
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 7, 5)
        profile[0, 4:, 4] = 0
        profile[1, :, 4] = 1
        with torch.no_grad():
            logits, lm_logits = model.forward_with_lm(tokens, profile)
            sequence, _, _ = model.sequence_hidden_with_lm(tokens)
            baseline_logits = model.output(sequence)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(logits, baseline_logits))

    def test_token_structure_aux_adapter_forward_and_zero_start(self) -> None:
        model = self.module.TokenStructureAuxAdapter(
            DummyRNAFM(), dropout=0.0, zero_start=True
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 7, 5)
        profile[0, 4:, 4] = 0
        profile[1, :, 4] = 1
        with torch.no_grad():
            logits, lm_logits, structure, valid = model.forward_with_aux(tokens, profile)
            sequence, _, _ = model.sequence_hidden_with_lm(tokens)
            baseline_logits = model.output(sequence)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertEqual(structure.shape, (2, 7, 3))
        self.assertEqual(valid.shape, (2, 7))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(logits, baseline_logits))

    def test_multilayer_structure_aux_adapter_forward(self) -> None:
        model = self.module.MultiLayerTokenStructureAuxAdapter(
            DummyRNAFM(), dropout=0.0, zero_start=False
        ).eval()
        tokens = torch.tensor([[0, 4, 5, 6, 7, 2, 1, 1]])
        profile = torch.randn(1, 7, 5)
        profile[:, :, 4] = 1
        with torch.no_grad():
            logits, lm_logits, structure, valid = model.forward_with_aux(tokens, profile)
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(lm_logits.shape, (1, 8, 8))
        self.assertEqual(structure.shape, (1, 7, 3))
        self.assertTrue(valid.all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_pretrained_penultimate_adapter_uses_hidden_not_scalar_output(self) -> None:
        model = self.module.PretrainedPenultimateStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        with torch.no_grad():
            hidden = model.adapter_hidden(tokens, profile)
            logits, lm_logits = model.forward_with_lm(tokens, profile)
        self.assertEqual(hidden.shape, (2, 32))
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())

    def test_pretrained_gated_adapter_conditions_one_joint_predictor(self) -> None:
        model = self.module.PretrainedGatedDeepStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        final_classifier_calls = []
        handle = model.adapter.sequence.classifier[-1].register_forward_hook(
            lambda *_: final_classifier_calls.append(True)
        )
        with torch.no_grad():
            logits, lm_logits = model.forward_with_lm(tokens, profile)
        handle.remove()
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(final_classifier_calls, [])

    def test_pretrained_separated_adapter_keeps_structure_hidden_explicit(self) -> None:
        model = self.module.PretrainedSeparatedDeepStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        final_classifier_calls = []
        handle = model.adapter.sequence.classifier[-1].register_forward_hook(
            lambda *_: final_classifier_calls.append(True)
        )
        with torch.no_grad():
            local_sequence, structure = model.adapter_components(tokens, profile)
            logits, lm_logits = model.forward_with_lm(tokens, profile)
        handle.remove()
        self.assertEqual(local_sequence.shape, (2, 16))
        self.assertEqual(structure.shape, (2, 16))
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(final_classifier_calls, [])

    def test_refinement_adapter_zero_start_preserves_joint_model(self) -> None:
        lookup = [0, 0, 0, 0, 1, 2, 3, 4]
        base = self.module.PretrainedDeepStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=lookup,
        ).eval()
        refined = self.module.WarmstartedStructureRefinementRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=lookup,
        ).eval()
        source = base.state_dict()
        target = refined.state_dict()
        target.update({name: value for name, value in source.items() if name in target})
        refined.load_state_dict(target, strict=True)
        tokens = torch.tensor(
            [[0, 4, 5, 6, 7, 2, 1, 1], [0, 7, 6, 5, 4, 7, 6, 2]]
        )
        profile = torch.randn(2, 174, 5)
        final_classifier_calls = []
        handle = refined.adapter.sequence.classifier[-1].register_forward_hook(
            lambda *_: final_classifier_calls.append(True)
        )
        with torch.no_grad():
            base_logits, _ = base.forward_with_lm(tokens, profile)
            refined_logits, _ = refined.forward_with_lm(tokens, profile)
        handle.remove()
        self.assertTrue(torch.allclose(base_logits, refined_logits, atol=1e-7))
        self.assertEqual(final_classifier_calls, [])

    def test_pretrained_multiscale_adapter_uses_16d_and_32d_hidden_states(self) -> None:
        model = self.module.PretrainedMultiscaleStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        with torch.no_grad():
            hidden = model.adapter_hidden(tokens, profile)
            logits, lm_logits = model.forward_with_lm(tokens, profile)
        self.assertEqual(hidden.shape, (2, 48))
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())

    def test_sequence_deep_ablation_ignores_structure_profile(self) -> None:
        model = self.module.PretrainedSequenceDeepRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor([[0, 4, 5, 6, 7, 2, 1, 1]])
        first = torch.zeros(1, 174, 5)
        second = torch.randn(1, 174, 5)
        with torch.no_grad():
            first_hidden = model.adapter_hidden(tokens, first)
            second_hidden = model.adapter_hidden(tokens, second)
        self.assertEqual(first_hidden.shape, (1, 16))
        self.assertTrue(torch.equal(first_hidden, second_hidden))

    def test_pretrained_multipool_adapter_forward(self) -> None:
        model = self.module.PretrainedMultiPoolStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        with torch.no_grad():
            sequence, residue, lm_logits = model.sequence_hidden_with_lm(tokens)
            logits, _ = model.forward_with_lm(tokens, profile)
        self.assertEqual(sequence.shape, (2, 80))
        self.assertEqual(residue.shape, (2, 8, 640))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_tunable_structure_adapter_only_unfreezes_gate_and_delta(self) -> None:
        model = self.module.TunablePretrainedStructureRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).train()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        logits, _ = model.forward_with_lm(tokens, profile)
        logits.sum().backward()
        backbone_parameters, adapter_parameters, head_parameters = (
            self.module.configure_trainable_parameters(model, 0)
        )
        self.assertTrue(model.adapter.gate.weight.requires_grad)
        self.assertTrue(model.adapter.structure_delta.weight.requires_grad)
        self.assertFalse(model.adapter.sequence.classifier[0].weight.requires_grad)
        self.assertIsNotNone(model.adapter.gate.weight.grad)
        self.assertIsNotNone(model.adapter.structure_delta.weight.grad)
        self.assertIsNone(model.adapter.sequence.classifier[0].weight.grad)
        self.assertEqual(backbone_parameters, [])
        self.assertIn(id(model.adapter.gate.weight), {id(value) for value in adapter_parameters})
        self.assertIn(id(model.output.weight), {id(value) for value in head_parameters})

    def test_sequence_penultimate_ablation_ignores_structure_profile(self) -> None:
        model = self.module.PretrainedSequencePenultimateRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=DummyPretrainedAdapter(),
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor([[0, 4, 5, 6, 7, 2, 1, 1]])
        first = torch.zeros(1, 174, 5)
        second = torch.randn(1, 174, 5)
        with torch.no_grad():
            first_hidden = model.adapter_hidden(tokens, first)
            second_hidden = model.adapter_hidden(tokens, second)
        self.assertEqual(first_hidden.shape, (1, 32))
        self.assertTrue(torch.equal(first_hidden, second_hidden))

    def test_penultimate_warmstart_preserves_teacher_margin(self) -> None:
        adapter = DummyPretrainedAdapter()
        model = self.module.PretrainedPenultimateWarmstartRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=adapter,
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        with torch.no_grad():
            hidden = model.adapter_hidden(tokens, profile)
            teacher_margin = adapter.sequence.classifier[-1](hidden).squeeze(1)
            logits, _ = model.forward_with_lm(tokens, profile)
            joint_margin = logits[:, 1] - logits[:, 0]
        self.assertTrue(torch.allclose(joint_margin, teacher_margin, atol=1e-6))

    def test_raw_penultimate_warmstart_preserves_teacher_margin(self) -> None:
        adapter = DummyPretrainedAdapter()
        model = self.module.PretrainedRawPenultimateWarmstartRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=adapter,
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        with torch.no_grad():
            hidden = model.adapter_hidden(tokens, profile)
            teacher_margin = adapter.sequence.classifier[-1](hidden).squeeze(1)
            logits, lm_logits = model.forward_with_lm(tokens, profile)
            joint_margin = logits[:, 1] - logits[:, 0]
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.allclose(joint_margin, teacher_margin, atol=1e-6))

    def test_cross_gated_adapter_zero_start_preserves_teacher_margin(self) -> None:
        adapter = DummyPretrainedAdapter()
        model = self.module.CrossGatedPenultimateRNAFM(
            DummyRNAFM(),
            dropout=0.0,
            adapter=adapter,
            deepires_token_lookup=[0, 0, 0, 0, 1, 2, 3, 4],
        ).eval()
        tokens = torch.tensor(
            [
                [0, 4, 5, 6, 7, 2, 1, 1],
                [0, 7, 6, 5, 4, 7, 6, 2],
            ]
        )
        profile = torch.randn(2, 174, 5)
        with torch.no_grad():
            hidden = model.adapter_hidden(tokens, profile)
            teacher_margin = adapter.sequence.classifier[-1](hidden).squeeze(1)
            logits, lm_logits = model.forward_with_lm(tokens, profile)
            joint_margin = logits[:, 1] - logits[:, 0]
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(lm_logits.shape, (2, 8, 8))
        self.assertTrue(torch.allclose(joint_margin, teacher_margin, atol=1e-6))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.output.parameters()))

    def test_pairwise_auc_loss_rewards_correct_order(self) -> None:
        labels = torch.tensor([1, 1, 0, 0])
        ordered = torch.tensor([[0.0, 2.0], [0.0, 1.0], [0.0, -1.0], [0.0, -2.0]])
        reversed_logits = ordered.flip(0)
        ordered_loss = self.module.pairwise_auc_loss(ordered, labels)
        reversed_loss = self.module.pairwise_auc_loss(reversed_logits, labels)
        self.assertLess(float(ordered_loss), float(reversed_loss))


if __name__ == "__main__":
    unittest.main()
