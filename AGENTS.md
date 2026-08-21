# Repository operating rules

## Mainline

- The primary task is 174-nt de novo IRES candidate design.
- Seeded IRES mutation optimization is the validation task.
- The method is a generator-agnostic structure/energy-constrained design layer built on standard multi-objective search.
- Old SFT/DPO experiments are historical evidence, not the central method.
- Rfam analysis is supporting evidence, not a second paper task.

## Attribution

- RFamLlama is third-party work by Sun, Li, and Deng (2024).
- Never write `our RFamLlama` or claim its pretraining as this project's contribution.
- External code, checkpoints, and data are installed outside Git and referenced by manifest.

## Evaluation

- Keep Task A and Task B in separate tables.
- The primary comparison is paired `Full vs Function-only` within the same public backbone and budget.
- IRES-DM released sequences are an external reference unless retrained under the shared protocol.
- Report function, structural applicability, validity, novelty/diversity, and efficiency together.
- No optimization scorer may be the sole final evaluator.
- Without wet-lab validation, use `computational candidate enrichment`, not verified biological improvement.

## File safety

- Do not commit large or licensed assets.
- Do not overwrite prior run directories.
- Every formal run requires a manifest and protocol-identifying run ID.
- Keep repository paths portable; use environment variables or repository-relative paths.

