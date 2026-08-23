# Repository operating rules

## Mainline

- The primary task is assay-aware, seeded optimization of full-length, experimentally supported viral IRESes.
- The method is a generator-agnostic structure-, energy-, and cargo-context-constrained design layer built on standard multi-objective search.
- The 174-nt de novo task is a legacy DNA-reporter benchmark against IRES-DM, not the primary biological claim.
- Before any large training run, complete the assay-shift gate comparing DNA-reporter labels with RNA-based IRES-TrAPPr measurements.
- Old SFT/DPO experiments are historical evidence, not the central method.
- Rfam analysis is supporting evidence, not a second paper task.

## Attribution

- RFamLlama is third-party work by Sun, Li, and Deng (2024).
- Never write `our RFamLlama` or claim its pretraining as this project's contribution.
- External code, checkpoints, and data are installed outside Git and referenced by manifest.

## Evaluation

- Keep assay audit, seeded mutation, and legacy 174-nt de novo results in separate tables.
- The primary comparison is paired `robust_full vs score_only` from the same seed, initial population, edit limit, and oracle-query budget.
- IRES-DM released sequences are an external reference unless retrained under the shared protocol.
- Report function, structural applicability, validity, novelty/diversity, and efficiency together.
- No optimization scorer may be the sole final evaluator.
- Every functional label and evaluator must state its assay context: DNA/lentiviral, circRNA-plasmid, direct RNA, or other.
- RFamLlama and GenerRNA are optional proposal backbones, not same-task IRES design baselines.
- Without wet-lab validation, use `computational candidate enrichment`, not verified biological improvement.

## Biosafety boundary

- Keep the project computational and limited to short, isolated, non-coding IRES regulatory fragments.
- Do not design or reconstruct complete viral genomes, replication-competent constructs, coding virulence factors, or infectious systems.
- Do not optimize pathogenicity, replication, host range, immune evasion, transmission, or expression of harmful cargo.
- Use inert reporter cargo contexts only; HCV and other pathogen-derived IRES controls are evaluation references, not targets for enhanced biological activity.
- Do not provide wet-lab construction, delivery, culture, infection, or validation protocols.
- Generated candidate sequences remain in ignored local run artifacts pending explicit biosafety and publication review.

## File safety

- Do not commit large or licensed assets.
- Do not overwrite prior run directories.
- Every formal run requires a manifest and protocol-identifying run ID.
- Keep repository paths portable; use environment variables or repository-relative paths.
