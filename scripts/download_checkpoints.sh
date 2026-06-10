#!/usr/bin/env bash
#
# Download all model checkpoints required to run the benchmark.
#
# Three paths:
#   1) HF-native models: pre-pull into ~/.cache/huggingface/ to avoid
#      thundering-herd across parallel slurm jobs.
#   2) logo-lab mirror: pull from logo-lab/trl-bench-ckpts where license allows.
#   3) Upstream-only: curl/wget from upstream URLs where license forbids re-host.
#
# Usage: bash scripts/download_checkpoints.sh [model_name ...]
# With no args: download everything required to run the full benchmark.

set -euo pipefail

CKPT_ROOT="${TRL_BENCH_CKPT_ROOT:-./checkpoints}"
mkdir -p "$CKPT_ROOT"

# == 1) HF-native pre-pull ====================================================
HF_NATIVE=(
    "bert-base-uncased"
    "thenlper/gte-base"
    "google/tapas-base"
    "microsoft/tapex-base"
)

for m in "${HF_NATIVE[@]}"; do
    echo "[hf-native] pre-pulling $m ..."
    python -c "from transformers import AutoModel, AutoTokenizer; \
               AutoTokenizer.from_pretrained('$m'); AutoModel.from_pretrained('$m')"
done

# TabICL / TabPFN: trigger PyPI auto-fetch
echo "[hf-native] pre-pulling TabICL ..."
python -c "import tabicl; tabicl.TabICLClassifier()  # triggers ckpt download"
echo "[hf-native] pre-pulling TabPFN ..."
python -c "import tabpfn; tabpfn.TabPFNClassifier()  # triggers ckpt download"

# Sentence-T5 and MPNet via sentence-transformers
echo "[hf-native] pre-pulling sentence-t5-base + mpnet ..."
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('sentence-transformers/sentence-t5-base'); \
           SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# == 2) logo-lab mirror (license-permitting models) ==========================
# Per docs/CHECKPOINT_LICENSES.md:
#   TUTA: MIT (mirror-permitted)
#   TURL: Apache-2.0 (mirror-permitted)
#   TaBERT: CC BY-NC 4.0 (mirrored under upstream's non-commercial license —
#           do NOT use this checkpoint in a commercial deployment; train your
#           own model or fetch from the upstream Google Drive instead, see
#           docs/CHECKPOINT_LICENSES.md for the URL.)
LOGO_LAB_MIRROR=(
    "tuta"
    "turl"
    "tabert"
)
for m in "${LOGO_LAB_MIRROR[@]}"; do
    echo "[logo-lab] pulling $m ..."
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('logo-lab/trl-bench-ckpts', allow_patterns='${m}/*', \
                                  local_dir='$CKPT_ROOT')"
done

# == 3) Upstream-only (not redistributed here; obtain from the upstream source) ====
# Per docs/CHECKPOINT_LICENSES.md:
#   TabSketchFM: CC BY-NC-ND 4.0 — non-commercial + no-derivatives; upstream only.
#   Starmie: no checkpoint distributed upstream; users train via
#            `python -m trl_bench.models.starmie.run_pretrain --data_path
#            <dataset> --checkpoint_dir $CKPT_ROOT/starmie/<dataset>`.
#   TABBIE: obtain from the upstream SFIG611/tabbie source (MIT); not mirrored.
#
# URLs below are pulled verbatim from docs/CHECKPOINT_LICENSES.md.
#
# NOTE on Google Drive direct downloads: the `uc?id=<id>` form is suitable for
# small files but Drive interposes a virus-scan interstitial for larger files
# (>~100MB) that `curl -L` cannot follow. For those checkpoints, prefer
# `gdown` (PyPI) or fetch via a browser per the upstream README. The script
# attempts `curl -L` as a best effort and surfaces a clear instruction when it
# falls back.

declare -A UPSTREAM_URL
declare -A UPSTREAM_LOCAL

# TabSketchFM: IBM, https://github.com/IBM/tabsketchfm — upstream README links
# to the LakeBench Zenodo record for downloads; no single stable direct URL was
# recorded in the audit, so this remains a documented manual step.
UPSTREAM_URL[tabsketchfm]="https://doi.org/10.5281/zenodo.8014642  # see docs/CHECKPOINT_LICENSES.md"
UPSTREAM_LOCAL[tabsketchfm]="tabsketchfm/epoch=10-step=27786.ckpt"

# Starmie: no upstream checkpoint — retrain. Print instructions instead of downloading.
# TABBIE: obtained from upstream source — print instructions.

for m in "${!UPSTREAM_URL[@]}"; do
    dst="$CKPT_ROOT/${UPSTREAM_LOCAL[$m]}"
    mkdir -p "$(dirname "$dst")"
    url="${UPSTREAM_URL[$m]}"
    if [[ "$url" == *"<id>"* ]] || [[ "$url" == *"see docs"* ]]; then
        echo "[upstream] WARNING: $m URL is a placeholder. See docs/CHECKPOINT_LICENSES.md"
        echo "[upstream]   Expected location: $dst"
        echo "[upstream]   Skipping; user must fetch manually."
        continue
    fi
    echo "[upstream] downloading $m from $url ..."
    if ! curl -L -o "$dst" "$url"; then
        echo "[upstream] WARNING: curl failed for $m. For Google Drive links, try:"
        echo "[upstream]   pip install gdown && gdown --fuzzy '$url' -O '$dst'"
        echo "[upstream]   or open the URL in a browser and save to: $dst"
    fi
done

# Special-case messages for Starmie and TABBIE
echo ""
echo "[manual] Starmie: NO checkpoint distributed upstream."
echo "[manual]   Retrain via: python src/trl_bench/models/starmie/run_pretrain.py"
echo "[manual]   See docs/CHECKPOINT_LICENSES.md for license details."
echo ""
echo "[manual] TABBIE: obtain from the upstream source — see docs/CHECKPOINT_LICENSES.md."
echo "[manual]   Place the upstream TABBIE weights at:"
echo "[manual]   $CKPT_ROOT/tabbie/weights.pt"
echo "[manual]   The SFIG611/tabbie Google Drive folder is:"
echo "[manual]   https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0"

# == 4) Verify sha256 ========================================================
echo ""
echo "[verify] checking sha256 sums of downloaded files ..."
if [ -f "$(dirname "${BASH_SOURCE[0]}")/checksums.sha256" ]; then
    # Skip lines that are comments or marked TBD
    grep -vE '^[[:space:]]*(#|$)|TBD' "$(dirname "${BASH_SOURCE[0]}")/checksums.sha256" | \
        while read -r line; do
            hash=$(echo "$line" | awk '{print $1}')
            relpath=$(echo "$line" | awk '{print $2}')
            full="$CKPT_ROOT/$relpath"
            if [ -f "$full" ]; then
                echo "$hash  $full" | sha256sum -c -
            else
                echo "[verify] SKIP: $relpath (not present)"
            fi
        done
fi

echo "[done] checkpoint download script complete. Manual steps may be required for Starmie / TABBIE."
