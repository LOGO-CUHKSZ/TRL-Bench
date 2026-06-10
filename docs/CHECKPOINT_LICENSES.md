# Checkpoint License Audit

This document records the upstream license, attribution requirements, and re-host
permission for each of the six upstream-pretrained models used in TRL-Bench.

The "Decision" field reflects the conservative default for a *public, potentially
commercial* mirror at `logo-lab/trl-bench-ckpts/<model>/`. If TRL-Bench is
released strictly as a non-commercial academic artefact, the decisions for
TaBERT / TabSketchFM may be revisited. Checkpoints without clear redistribution
terms are documented upstream-only and are not mirrored.

SHA256 values are recorded only for checkpoints that existed in the source
working tree's `checkpoints/<model>/` directory at audit time. Models without a
local checkpoint are marked `TBD; compute after download`.

---

## Summary table

| Model        | License            | Re-host?                      | Decision                                    |
|--------------|--------------------|-------------------------------|---------------------------------------------|
| TUTA         | MIT                | YES, attribution required     | Mirror                                      |
| TaBERT       | CC BY-NC 4.0       | NON-COMMERCIAL ONLY           | Mirror under NC tag (TRL-Bench is academic) |
| TURL         | Apache-2.0         | YES, attribution required     | Mirror                                      |
| TabSketchFM  | CC BY-NC-ND 4.0    | NO (no derivatives)           | Document upstream URL only                  |
| Starmie      | No LICENSE file    | No (upstream-only)            | Document upstream URL only (user retrains)  |
| TABBIE       | MIT (SFIG611 fork) | YES, attribution required     | Document upstream URL only                  |

---

### TUTA

- **Upstream repo:** https://github.com/microsoft/TUTA_table_understanding
- **License:** MIT License (`Copyright (c) Microsoft Corporation`)
- **Re-host permitted:** YES — attribution required
- **Attribution text:**
  - Include the MIT copyright notice ("Copyright (c) Microsoft Corporation.")
    and a copy of the MIT License with any redistributed checkpoint.
  - Cite: Wang, Z. et al. "TUTA: Tree-based Transformers for Generally
    Structured Table Pre-training." KDD 2021.
- **Checkpoint URL (upstream):**
  - TUTA (implicit, used in this repo): https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g/view?usp=sharing
  - TUTA-explicit: https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW/view?usp=sharing
  - TUTA-base: https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l/view?usp=sharing
- **SHA256:**
  - `checkpoints/tuta/tuta.bin`: `51d05030ff23f257c4eedc6db350e1c7b733f61226c822bf63459d4f3c6b4db8`
- **Decision:** Mirror to `logo-lab/trl-bench-ckpts/tuta/` with a copy of the
  upstream MIT LICENSE and a NOTICE referencing the citation above.
- **Status:** Live as of 2026-05-18 at
  https://huggingface.co/logo-lab/trl-bench-ckpts (path: `tuta/tuta.bin`).
  Per-model `LICENSE` and `NOTICE` files were uploaded alongside the binary.

---

### TaBERT

- **Upstream repo:** https://github.com/facebookresearch/TaBERT  (archived Oct 2023, read-only)
- **License:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0).  See https://github.com/facebookresearch/TaBERT/blob/main/LICENSE.md
- **Re-host permitted:** NON-COMMERCIAL ONLY — sharing of adapted material is
  permitted under the same license, but commercial use is prohibited. Mirroring
  to a generic public bucket (no guarantee of non-commercial downstream use) is
  not consistent with the license.
- **Attribution text:**
  - Retain Facebook Inc. / Meta Platforms attribution and the CC BY-NC 4.0
    notice; provide a link to the licensed material; indicate any modifications.
  - Cite: Yin, P., Neubig, G., Yih, W., Riedel, S. "TaBERT: Pretraining for
    Joint Understanding of Textual and Tabular Data." ACL 2020. arXiv:2005.08314.
- **Checkpoint URL (upstream):** Google Drive (per upstream README)
  - `tabert_base_k1`: https://drive.google.com/uc?id=1-pdtksj9RzC4yEqdrJQaZu4-dIEXZbM9
  - `tabert_base_k3`: https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj
  - `tabert_large_k1`: https://drive.google.com/uc?id=1eLJFUWnrJRo6QpROYWKXlbSOjRDDZ3yZ
  - `tabert_large_k3`: https://drive.google.com/uc?id=17NTNIqxqYexAzaH_TgEfK42-KmjIRC-g
  - Shared folder: https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg
- **SHA256:**
  - `checkpoints/tabert/tabert_base_k1/model.bin`: `c4f5ea5e8512d6f4c898966fac5c291248f45eb2301f182ad39481fb44e128ea`
  - `checkpoints/tabert/tabert_base_k3/model.bin`: `6a1736360627b033f8c52dcc19b864f0446722edf3c406101179f4bb8f6984f6`
  - `checkpoints/tabert/tabert_large_k1/model.bin`: `d369f6331107a7a38ddb1323aebea4f7eb1354c97b977487cc41db78e13913eb`
  - `checkpoints/tabert/tabert_large_k3/model.bin`: `6df07e347e16ba1198104804d9be851b1e497ff4c1ba94ecb9a8ac4a3ff3119d`
- **Decision (2026-05-20 — revised from earlier "upstream-only" stance):**
  Mirror to `logo-lab/trl-bench-ckpts/tabert/tabert_base_k3/model.bin` under
  the upstream CC BY-NC 4.0 license. TRL-Bench is released as a
  non-commercial academic artefact (see top-level LICENSE); the mirror
  inherits the same non-commercial restriction. The LICENSE and NOTICE
  files live alongside the binary on HF and reproduce the upstream CC
  BY-NC 4.0 terms verbatim plus the canonical citation. Commercial
  downstream users MUST NOT use this checkpoint — train your own model or
  obtain an alternate license from Meta Platforms.
- **Status:** Live as of 2026-05-20 at
  https://huggingface.co/logo-lab/trl-bench-ckpts (paths:
  `tabert/tabert_base_k3/{model.bin,tb_config.json,version.txt}` plus
  `tabert/{LICENSE,NOTICE}`).

---

### TURL

- **Upstream repo:** https://github.com/sunlab-osu/TURL
- **License:** Apache License 2.0 (verified locally against the upstream
  `models/turl/code/LICENSE` file).
- **Re-host permitted:** YES — attribution required.
- **Attribution text:**
  - Retain the Apache-2.0 LICENSE file and any NOTICE file in redistribution.
  - State modifications in modified files ("prominent notices").
  - Cite: Deng, X., Sun, H., Lees, A., Wu, Y., Yu, C. "TURL: Table Understanding
    through Representation Learning." PVLDB 14(3), 2020. http://www.vldb.org/pvldb/vol14/p307-deng.pdf
- **Checkpoint URL (upstream):** SharePoint folder linked from the upstream
  README:
  https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/deng_595_buckeyemail_osu_edu/EjZWRtslWX9CubQ92jlmNTgB74hxxXszy9BUaXG5OL5F-g
- **SHA256:**
  - `checkpoints/turl/pretrained/pytorch_model.bin`: `edfe013b68083e91d66e15bef40de7a52f2851b5d803fffcd9c6d96a475dc37e`
- **Decision:** Mirror to `logo-lab/trl-bench-ckpts/turl/` with a copy of the
  upstream Apache-2.0 LICENSE and a NOTICE that references the upstream paper.
- **Status:** Live as of 2026-05-18 at
  https://huggingface.co/logo-lab/trl-bench-ckpts (path:
  `turl/pretrained/pytorch_model.bin`). Per-model `LICENSE` and `NOTICE` files
  were uploaded alongside the binary.

---

### TabSketchFM

- **Upstream repo:** https://github.com/IBM/tabsketchfm
- **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
  International (CC BY-NC-ND 4.0).  GitHub returns NOASSERTION/Other for the
  SPDX classifier; the upstream README explicitly states "This code is released
  with CC BY-NC-ND 4.0 License" plus an additional restrictive-use clause from
  IBM.
- **Re-host permitted:** NO.  The ND clause forbids sharing of adapted material;
  a fine-tuned or repacked checkpoint can plausibly be considered an adaptation.
  Commercial use is also prohibited.  The additional IBM clause ("only for the
  purpose of comparing this code to other code for scientific experimental
  purposes, where that distribution is not for a fee") further restricts use.
- **Attribution text:**
  - Retain creator attribution to IBM and full CC BY-NC-ND 4.0 notice; do not
    modify or repack the checkpoint.
  - Cite: Khatiwada, A. et al. "TabSketchFM: Sketch-based Tabular Representation
    Learning for Data Discovery over Data Lakes." IEEE ICDE 2025.  arXiv:2407.01619.
- **Checkpoint URL (upstream):** Pretrained `.ckpt` is referenced from the
  upstream README; downloads are linked from
  https://github.com/IBM/tabsketchfm and the
  TabSketchFM Zenodo record (LakeBench): https://doi.org/10.5281/zenodo.8014642
- **SHA256:**
  - `checkpoints/tabsketchfm/epoch=10-step=27786.ckpt`: `26f2107d7640bf9485026ff643ff83e569e412572aa68d540d88494f7f9f211d`
  - `checkpoints/tabsketchfm/epoch=15-step=12112.ckpt`: `90d38c0d8f22d171b759758a782314460f1dec8fc5cfc44e1cfa32e943500525`
- **Decision:** Document upstream URL only.  Do not mirror to
  `logo-lab/trl-bench-ckpts/`.  Users must obtain the checkpoint directly from
  IBM under the upstream license.

---

### Starmie

- **Upstream repo:** https://github.com/megagonlabs/starmie
- **License:** **NO LICENSE FILE PRESENT** in the upstream repository (verified
  via the GitHub API: no `LICENSE` file at repository root, and the GitHub
  `license` metadata field returns `null`).  The repository README contains a
  disclosure section governing the **third-party datasets** bundled with the
  code, but does **not** state a license for the code or for any released model
  weights.  Default copyright applies: "All rights reserved" to the authors.
- **Re-host permitted:** No. With no upstream license file, the weights are not
  redistributed here; users train their own via `run_pretrain.py`.
- **Attribution text:** Even when only documenting the upstream link, cite:
  Fan, G., Wang, J., Li, Y., Zhang, D., Miller, R. J. "Semantics-aware Dataset
  Discovery from Data Lakes with Contextualized Column-based Representation
  Learning." PVLDB 16(7), 2023.
- **Checkpoint URL (upstream):** No pretrained checkpoint is released by the
  upstream repo.  Starmie expects users to train their own model via
  `run_pretrain.py`; downstream tasks then load the locally produced
  `model_drop_col_head_column_X.pt`.  Note this in the release docs.
- **SHA256:** `TBD; no local checkpoint`  (Starmie is trained locally per
  dataset; there is no single canonical pretrained binary to hash.)
- **Decision:** Document upstream URL only.  Do **not** mirror.  Before any
  redistribution of Starmie weights produced inside TRL-Bench, contact Megagon
  Labs (the authors) to confirm a license under which those weights may be
  shared.

---

### TABBIE

- **Upstream repo:** https://github.com/SFIG611/tabbie
- **Upstream source:** The wrappers in `src/trl_bench/models/tabbie/` point at
  `SFIG611/tabbie` (MIT), a release of the TABBIE code (Iida, Thai, Manjunatha,
  Iyyer; NAACL 2021; arXiv:2105.02584). Weights are obtained from that upstream
  source and are not redistributed here.
- **License:** MIT License (as declared in `SFIG611/tabbie/LICENSE`).
- **Re-host permitted:** YES under MIT, attribution required — *contingent* on
  resolving the canonicity question above.
- **Attribution text:**
  - Retain the SFIG611 MIT copyright notice and the MIT License text with any
    redistributed checkpoint.
  - Cite the original paper: Iida, H., Thai, D., Manjunatha, V., Iyyer, M.
    "TABBIE: Pretrained Representations of Tabular Data." NAACL 2021.
    arXiv:2105.02584. https://aclanthology.org/2021.naacl-main.270/
- **Checkpoint URL (upstream):** Google Drive folder referenced from
  `SFIG611/tabbie` README:
  https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0
  Setup instructions specify the files `freq.tar.gz` and `mix.tar.gz`.
- **SHA256:** `TBD; no local checkpoint` (no TABBIE checkpoint was present in
  the source working tree's `checkpoints/` directory; download the upstream
  `mix.tar.gz` and compute on first use.)
- **Decision:** Document upstream URL only **until** canonicity of
  `SFIG611/tabbie` is confirmed.  If confirmed canonical (e.g., by paper-author
  acknowledgement), mirror to `logo-lab/trl-bench-ckpts/tabbie/` with the
  upstream MIT LICENSE and a NOTICE citing the NAACL 2021 paper.

---

## Checkpoints documented upstream-only

1. **Starmie** — upstream repo has no LICENSE file, so weights are not mirrored;
   TRL-Bench uses a user-trained Starmie flow (`run_pretrain.py`, per-dataset).
2. **TABBIE** — weights are obtained from the upstream `SFIG611/tabbie` source
   and are not redistributed here.
3. **TabSketchFM mirror policy** — has CC BY-NC-ND 4.0 (no-derivatives);
   stays upstream-only. TaBERT (CC BY-NC 4.0, was previously also
   upstream-only) is now mirrored on logo-lab/trl-bench-ckpts under the
   non-commercial inheritance described in its section above.

## Per-wrapper "where to place it" (for `--checkpoint-root <root>`)

The dispatcher resolves `<checkpoint-root>/<template>` per
`ExtractorConfig.checkpoint_template` (see `src/trl_bench/registry.py`).
Defaults to `./checkpoints` (override via `--checkpoint-root` or
`$TRL_BENCH_CKPT_ROOT`). After running `scripts/download_checkpoints.sh`
the expected on-disk layout is:

| Wrapper      | Expected path under `<checkpoint-root>/`                                  | Source                                                                                             |
|--------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| TaBERT       | `tabert/tabert_base_k3/model.bin`                                         | HF: `logo-lab/trl-bench-ckpts` (NC mirror) OR upstream Google Drive                                |
| TURL         | `turl/pretrained/{pytorch_model.bin,config.json}`                         | HF: `logo-lab/trl-bench-ckpts` (auto-fetched)                                                      |
| TUTA         | `tuta/tuta.bin`                                                           | HF: `logo-lab/trl-bench-ckpts` (auto-fetched)                                                      |
| TabSketchFM  | `tabsketchfm/epoch=10-step=27786.ckpt`                                    | MANUAL: https://doi.org/10.5281/zenodo.8014642                                                     |
| TABBIE       | `tabbie/weights.pt`                                                       | MANUAL: SFIG611/tabbie Google Drive                                                                |
| Starmie      | `starmie/<dataset>/model_drop_col,sample_row_head_column_0.pt`            | RETRAIN: `python -m trl_bench.models.starmie.run_pretrain --data_path <dataset>` per-dataset       |
