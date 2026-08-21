# Third-party provenance

Third-party source code, weights, datasets, and papers are not copied into this repository. Baseline adapters consume user-supplied local installations and record their source revision in a run manifest.

| Asset | Upstream | Intended role | Redistribution rule |
|---|---|---|---|
| RFamLlama | <https://github.com/JinyuanSun/RFamLlama> | public RNA LM baseline | Cite Sun, Li, and Deng (2024); verify code and weight licenses separately. |
| GenerRNA | <https://huggingface.co/pfnet/GenerRNA> | independent public RNA LM baseline | Preserve the model-card license and exact checkpoint revision. |
| IRES-LM / IRES-EA / IRES-DM | <https://github.com/a96123155/IRES_Prediction_Design> | IRES scorer, mutation baseline, diffusion reference | Do not copy the nested repository into this project; record upstream commit and data terms. |
| DeepIRES | Original authors' distribution | independent IRES scorer | Evaluation only; follow upstream license. |
| ViennaRNA | <https://www.tbi.univie.ac.at/RNA/ViennaRNA/> | folding and energy features | System dependency; cite the package version used. |
| IRESbase | <https://academic.oup.com/gpb/article/18/2/129/7229793> | natural IRES reference | Follow database terms and record download date. |

## RFamLlama boundary

The historical working directory from which this repository was assembled began as a clone of the original RFamLlama repository. The public repository created from this scaffold is a new project with independent history. It does not claim the RFamLlama architecture, name, tokenizer, pretrained weights, or Rfam family-conditioning method as original work.

No file from an external repository should be committed unless its license permits redistribution, required notices are preserved, and the file is listed here.

