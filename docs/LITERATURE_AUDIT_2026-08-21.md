# Literature and baseline audit — 2026-08-21

## Decision

Do not begin a large 174-nt generator training run yet. The scientifically defensible mainline is:

1. audit the measurement shift between the 2016 DNA/lentiviral reporter data used by IRES-AI and the 2026 RNA-based IRES-TrAPPr data;
2. make full-length, experimentally supported viral IRES mutation design the primary task;
3. optimize function conservatively while preserving the IRES structural ensemble and reducing IRES–cargo crosstalk;
4. retain 174-nt de novo generation as a separately labelled legacy-reporter benchmark against IRES-DM.

This preserves the original IRES-design story while avoiding the unsupported equation “old reporter score = bona fide IRES activity.”

## Primary-source method matrix

| Work | Data and task | Method | Experimental evidence | Role here |
|---|---|---|---|---|
| [IRES-AI / IRES-DM / IRES-EA (2026)](https://www.nature.com/articles/s42256-026-01213-z) | 46,774 labelled sequences, dominated by 174-nt fragments from the 2016 lentiviral MPRA; de novo generation and mutation | IRES-LM ensemble; U-Net diffusion generation; predictor-guided evolutionary mutation | 12,000 IRES-EA and 12,000 IRES-DM candidates tested in a lentiviral/FACS MPRA | Only exact same-task external work. IRES-EA is the direct mutation baseline; IRES-DM is the direct de novo baseline/reference. |
| [IRES-TrAPPr (2026 preprint)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13174604/) | RNA-based MPRA; nearly 2,000 candidates, IAPV/HCV mutational scans, 37 Type IV and 182 Type VI candidates | Co-transfected A-cap IRES and G-cap CDI mRNAs; nascent-translation affinity capture | Orthogonal luciferase validation; 24 active Type IV and 43 active Type VI candidates | Primary assay-provenance anchor and locked external audit. Not a generator baseline. |
| [Delli Ponti et al. (2026)](https://link.springer.com/article/10.1186/s13062-025-00706-y) | Seven viral/cellular Rfam IRES families; single-nucleotide variants | RNAfold structure identity, paired fraction and MFE plus catRAPID ITAF/RBP interaction scoring | Computational only | Direct structure-only mutation baseline. It means “MFE + fold preservation” alone is not a novel contribution. |
| [Albatross (2026 preprint)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13228543/) | 96 full-length IRESes across six cell types; about 50,000 diversity-balanced IRES sequences; about 75,000 dependency maps | RiNALMo continued pretraining and dependency mapping | DMS-MaPseq structure profiles and functional atlas | Independent structure evaluator and seed/data source, not a generator baseline. |
| [IRES–cargo interplay (2026)](https://www.nature.com/articles/s41422-026-01233-9.pdf) | 45 viral IRESes and SV-A IRES/cargo variants in circular RNA | circSHAPE-MaP-guided structure analysis; IRES Crosstalk Ratio and Structure Consistency | Disrupting IRES–cargo pairing restored cargo expression | Direct evidence and metric precedent for cargo-context constraints. |
| [LAMAR (2025)](https://link.springer.com/article/10.1186/s13059-025-03752-x) | General RNA foundation model with an IRES prediction task | RNA encoder plus downstream prediction heads | CircRNA luciferase validation across cell lines | Secondary frozen function evaluator only; its training-label provenance must remain visible. |
| [PARADE (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722239/) | 60,000 5′/3′ UTR fragments across six cell types; a second 24,000-sequence designed MPRA | Predictor plus cold diffusion, genetic algorithm, random screening and motif design | Cell and mouse validation | Algorithmic-control precedent. Especially important because random screening was competitive with diffusion for short 5′ UTRs. Not an IRES baseline. |
| [SANDSTORM/GARDN (2025)](https://www.nature.com/articles/s41467-025-59389-8) | 5′ UTRs, RBSs, guides and toehold switches | Sequence/structure CNN predictor plus WGAN-GP generator and latent optimization | Experimental RBS and toehold validation | Predictor-coupled design precedent; optional method reference, not an IRES baseline. |
| [GARNET (2024)](https://www.nature.com/articles/s41467-024-54812-y) | Functional RNA mutation design | RNA-LM likelihood differences rank mutations | Thermostability/activity tests | Precedent for the LM-likelihood mutation baseline. |
| [RFamLlama (2024)](https://openreview.net/forum?id=dXnQedxEJD) | Conditional generation across Rfam families | Decoder-only Llama with Rfam ID prefixes | Infernal recovery and limited structural prediction, not IRES activity | Optional proposal backbone/supplement only. |
| [GenerRNA (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310814) | General de novo RNA generation and protein-binding fine-tuning | Autoregressive Transformer pretrained on about 16 million RNAs | In-silico MFE, novelty and affinity | Optional proposal backbone/supplement only. |
| [GoForth (2026 preprint)](https://arxiv.org/abs/2605.07608) | Full/partial inverse folding with base and coding constraints | Encoder-decoder conditional language model trained on ViennaRNA-witnessed pairs | Thermodynamic benchmark only | Optional structure-proposal reference, not an IRES-function baseline. |

## Findings that change the old plan

### 1. The old 174-nt target is assay-specific

IRES-AI's training set is dominated by the Weingarten-Gabbay 174-nt lentiviral reporter library. IRES-TrAPPr reports that the old short candidates it retested had negligible activity in a direct RNA assay and notes that 174 nt is shorter than many bona fide viral IRESes. Therefore:

- the old label can still define a reproducible `legacy_dna_reporter` benchmark;
- it cannot be the only training target or final biological evaluator;
- no paper claim may silently rename a high old-oracle score as real IRES translation.

### 2. Structure and MFE are necessary controls, not sufficient novelty

Delli Ponti et al. already use fold identity, paired fraction, lower MFE and predicted protein interactions to propose IRES mutations. The project must include this rule as a baseline and go beyond it through assay robustness, ensemble-level structure, context-specific crosstalk and uncertainty.

### 3. A strong random-search baseline is mandatory

PARADE found random screening competitive with diffusion on short 5′ UTR design in several comparisons. Any apparent gain over raw sampling can otherwise be explained by a larger candidate/query budget. Random screening, score-only GA and the proposed method must receive identical oracle calls.

### 4. Generic RNA generators are not headline baselines

RFamLlama and GenerRNA model natural RNA distributions; neither establishes IRES activity design under the target assay. They may supply proposals or a supplementary backbone ablation, but the headline comparison is against IRES-EA/IRES-DM and strict search controls.

## Frozen task hierarchy

### Task 0 — assay-shift audit (mandatory gate)

Question: do models trained on the old DNA/lentiviral assay rank RNA-based IRES-TrAPPr activity?

- Reproduce the IRES-LM/UTR-LM evaluation on its native split.
- Join sequences that occur in both assay sources without using RNA-based test labels for tuning.
- Evaluate ranking, calibration and top-k overlap on IRES-TrAPPr.
- Stratify by source, length, IRES type, sequence similarity and assay.
- Report cross-assay disagreement examples and applicability distance.

Pass condition: a usable out-of-family signal with calibrated uncertainty. Failure does not end the project; it forces the RNA-based scorer to be treated as an audit/filter rather than a dense optimization oracle.

### Task 1 — primary: full-length seeded IRES optimization

Seeds come from experimentally supported viral IRESes, initially IAPV, HCV, CrPV and SV-A, then a frozen type-balanced panel from IRES-TrAPPr/Albatross.

The proposed variant, `robust_full`, uses:

1. a conservative function objective: lower-confidence aggregation of independently trained, assay-labelled scorers;
2. seed/consensus ensemble preservation using base-pair probabilities rather than MFE alone;
3. IRES–cargo crosstalk minimization and structure-consistency preservation across a frozen cargo panel;
4. applicability, novelty and diversity constraints;
5. standard NSGA-II so the contribution is the formulation and evidence, not a renamed optimizer.

### Task 2 — secondary: legacy 174-nt de novo benchmark

Reproduce IRES-DM and evaluate its released sequences under the old assay context. Compare against random screening, score-only GA and any optional public proposal backbone with exactly matched candidate and oracle-query budgets. Report RNA-based evaluator transfer separately; do not merge the two assays into one score.

## Baseline set to implement

### Primary seeded task

| ID | Purpose | Strict fairness requirement |
|---|---|---|
| `random_mutation` | non-neural lower bound | Same seeds, edit distribution, candidates and queries. |
| `lm_likelihood_mutation` | GARNET-style naturalness/fitness prior | Same parent LM and edit budget where paired. |
| `structure_only_mutation` | Delli-Ponti-style rule | Same mutation pool; no function score. |
| `score_only_ga` | isolates gain from attacking the function oracle | Same population, initialization, generations and queries. |
| `ires_ea` | exact IRES mutation method | Same seed and mutation/query budget when technically possible; otherwise clearly external. |
| `nsga2` + `robust_full` | proposed formulation | Paired against `score_only_ga` with all other settings fixed. |

### Legacy de novo task

| ID | Purpose | Status |
|---|---|---|
| `ires_dm` | released exact-task reference | External, not budget-matched. |
| `ires_dm_retrained` | strict same-data reproduction | Required before a strict superiority claim. |
| `random_screen` | PARADE-style strong lower bound | Required. |
| `score_only_ga` | predictor-guided search control | Required. |
| `official_rfamllama`, `generna` | generic proposal ablations | Optional/supplementary. |

## Data sufficiency

- **Enough now:** legacy 174-nt reproduction, assay-shift audit, simple/frozen-embedding scorers, full-length seed selection, exhaustive or sampled low-edit mutation pools, structure/cargo metrics and computational ablations.
- **Not enough:** training a new foundation model; claiming de novo bona fide IRES discovery from 67 RNA-active examples; claiming biological improvement without new wet-lab validation.
- **Recommended modeling:** k-mer/GC/length baselines, frozen RNA-LM embeddings with linear or shallow heads, and a small calibrated ensemble. Use family/similarity holdouts. Do not train a large end-to-end generator on the RNA-based active set.

## Stop/go gates

1. **Data gate:** recover the nine IRES-TrAPPr supplementary tables, verify sequence/label fields, and create immutable hashes. **Retrieved and hashed on 2026-08-21; normalization/split audit remains.**
2. **Leakage gate:** cluster all sources and freeze family/similarity splits before fitting any scorer.
3. **Assay gate:** quantify old-to-new transfer before using an old scorer for candidate selection.
4. **Baseline gate:** random, structure-only, score-only GA and IRES-EA smoke tests must all run before the proposed method.
5. **Budget gate:** reject any table where compared methods used different candidate or oracle-query budgets.
6. **Claim gate:** without wet lab, use “computational enrichment/robustness,” never “improved IRES activity.”

## Planned paper figures

1. assay provenance and technical-route schematic;
2. cross-assay generalization/calibration and failure analysis;
3. locked benchmark performance of all strict baselines;
4. Pareto front for function, ensemble preservation and cargo crosstalk;
5. paired ablations (`score_only`, `mfe`, `ensemble`, `context`, `robust_full`);
6. diversity/applicability/efficiency and representative structures.

## Local paper cache

The reviewed PDFs are stored under ignored `assets/papers/core/`. They are local research assets and must not be committed or redistributed. The committed source/hash inventory is `assets/manifests/literature_sources.json`.

The IRES-TrAPPr PDF and all nine XLSX supplements were retrieved from Europe PMC's official `supplementaryFiles` API after NCBI's individual-file mirror returned intermittent 404/anti-bot responses. The tables include 240 IAPV/control rows, 108 HCV variant rows, 227 Type IV/VI/control rows (67 literal active labels), additional viral/cellular/circRNA audits, stress measurements and luciferase summaries. See `assets/manifests/ires_trappr_supplement.json`. The primary sequencing accession is `PRJNA1453056`.
