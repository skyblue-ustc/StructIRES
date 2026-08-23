# StructIRES: first classifier milestone plan

## Objective

Establish whether a classifier that combines a public RNA-FM sequence prior
with **position-aware, RNA secondary-structure ensemble information** improves
IRES recognition over a matched RNA-FM sequence-only model.  The design
optimizer is downstream of this classifier; it will not be presented as an
independent activity predictor.

## Protocol amendment: checkpoint-native primary comparison (2026-08-23)

The strict 90%-identity protocol below remains the generalization development
track.  The primary near-term comparison instead starts from the *released
IRES-supervised RNA-FM fold checkpoint*, not a newly initialized classifier
head. This directly answers whether structural information improves an
established IRES recognizer.

- **S-release:** frozen released RNA-FM + released $640\rightarrow40\rightarrow2$
  IRES head, evaluated on its corresponding official fold test once.
- **StructIRES-release:** S-release plus a trainable, zero-initialized
  structural residual. The residual consumes a precomputed 21-dimensional,
  label-free ViennaRNA MFE/ensemble feature vector and is modulated by a
  learned gate.
- **Selection discipline:** a deterministic stratified validation subset is
  cut only from the fold's upstream training records. Epoch and threshold are
  selected there; the official fold test is not used for selection.
- **Invariant:** before training, StructIRES-release must reproduce S-release
  probabilities numerically. Failure aborts the run.

This is the fastest credible route to an apples-to-apples baseline improvement.
The position-profile CNN variants below are retained as a stricter,
identity-aware follow-up rather than being conflated with the released-split
result.

## Fixed data protocol

- **Development protocol:** the existing length-174, 90% sequence-identity
  cluster split (`70/15/15`, seed 42).  Training, architecture decisions,
  threshold selection, and early stopping use training/validation only.
- **Final test:** the test partition stays untouched until a configuration is
  locked.  Report AUROC, AUPRC, F1, accuracy, sensitivity, specificity, MCC,
  calibration (ECE), and seed mean +/- standard deviation.
- **Paper-aligned reference:** the ten published 90/10 repeated holds are
  reproduced separately as a protocol-aligned reference.  They are not a
  replacement for the cluster-aware test because sequences recur across holds.

## Models and ablations

All variants use the same public RNA-FM t12 backbone and the same split.

| ID | Variant | Purpose |
|---|---|---|
| S0 | k-mer logistic baseline | transparent non-neural lower bound |
| S1 | frozen RNA-FM + sequence head | tests the pretrained sequence prior |
| S2 | S1 with last 1--2 Transformer blocks unfrozen | controlled task adaptation |
| T0 | BPP/MFE position-profile encoder only | tests whether secondary structure itself carries signal |
| F1 | concatenated RNA-FM and profile encoders | tests complementarity |
| F2 | gated RNA-FM/profile fusion (StructIRES) | proposed classifier; gate is learned from both embeddings |
| F3 | F2 minus MFE-state channel / minus BPP channel | structural ablation |

The structural profile for every nucleotide contains: ensemble base-pairing
probability, MFE paired/unpaired state, and local normalized position.  This
is deliberately different from edit distance: it represents predicted RNA
secondary-structure constraints at individual positions.  The profile encoder
is a small 1-D CNN with global pooling, so it is trainable end-to-end with the
fusion head while ViennaRNA itself remains a fixed, label-free feature source.

## Training and selection

1. Run a 3-epoch RNA-FM smoke test to prove data ordering, GPU execution, and
   artifact writing.  It is never reported as a scientific result.
2. Train S1 for 12 epochs, validation-AUPRC early selection, three seeds.
3. If S1 underfits, evaluate S2 with last one and then two blocks unfrozen;
   choose the smallest setting based only on validation AUPRC/MCC.
4. Build the position-profile cache once from the public sequences.  It uses
   no labels and is SHA-verified in the run manifest.
5. Train T0, F1 and F2 with identical seeds/batches.  Select one configuration
   using validation AUPRC first and validation MCC as a tie-breaker.
6. Evaluate that locked configuration once on the test set and bootstrap the
   per-sequence test predictions.  If fusion does not improve, report it as an
   ablation and retain the best supported model rather than claiming a gain.

## Acceptance criteria

- **Primary:** an improvement over S1 in held-out AUPRC, with the direction
  consistent in at least two of three seeds.
- **Secondary:** no material degradation in AUROC/F1/MCC, and paired BPP/MFE
  profiles provide an interpretable contribution in F3.
- **Design gate:** only a classifier that passes the primary criterion becomes
  the activity scorer for StructIRES-EA.  Candidate selection then optimizes
  the frozen classifier score jointly with parent-relative BPP distance,
  energy change, and protected-anchor retention.  This is a computational
  ranking result, not a claim of experimental activity.

## Time-box and decisions

- Today: finish smoke, finish paper-aligned reference, build profile cache,
  and obtain S1/S2 validation trajectories.
- Next GPU window: T0/F1/F2 in parallel where capacity allows.
- Asset failure rule: if a required checkpoint/cache is absent or fails an
  integrity check, stop the dependent run, record the exact file/command, and
  ask the maintainer to download it; do not retry downloads indefinitely.

## Paper deliverables

1. Main classifier table: external baselines plus S0/S1/S2/F2, with each
   evaluation regime labeled.
2. Ablation table: S1, T0, F1, F2, F3 on the locked split.
3. Method figure: RNA-FM sequence path, BPP/MFE profile path, learned gate,
   and downstream constrained candidate ranking.
4. Structure figure: an IAPV-like exemplar only as an explanatory secondary
   structure visualization, plus aggregate BPP-preservation statistics for
   scored candidates.  It is not evidence of wet-lab function.
