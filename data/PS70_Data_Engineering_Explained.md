# PS70 CycloneWatch — Data Engineering Pipeline Explained

**Owner:** Abhinav Pal (IT) — Data Engineering
**Project:** PS70 CycloneWatch, SIH 2026
**Purpose of this document:** explain what the data pipeline does, why it's built this way, what bugs were found and fixed along the way, and exactly what's handed off to ML (Aditya) at the end.

---

## 1. What this pipeline produces

Two historical cyclones — **Biparjoy (2023, Arabian Sea)** and **Amphan (2020, Bay of Bengal)** — turned into a training-ready dataset:

- Real satellite imagery (Infrared + Water Vapour) at **3-hourly cadence**, standardized into fixed-size tensors
- Each frame joined to its **real ground-truth cyclone centre position** (latitude/longitude) at the matching timestamp
- Full **provenance** for every frame: source, satellite, product, timestamp, bounding box, coordinate system, license
- A single manifest file ML can load directly into a PyTorch dataloader

This satisfies the project's non-negotiable principles: **real data over fabricated sophistication**, and **every prediction traceable back to source imagery, timestamp, and metadata**.

---

## 2. Why a real pipeline, not a single download

The two cyclones lasted 5–13 days each. At 3-hourly cadence that's **~40–90 satellite frames per storm** — not one snapshot. The system needs sequences because:

- The **temporal model** (ML's Day 3 deliverable) predicts T+12/T+24 from a T-12→T0 sequence of frames — impossible with a single image.
- **Historical replay** (the flagship credibility feature — see execution manual §8A) needs a full T-48h→T0 slider a judge can scrub through.
- **MAE calculation** needs many predicted-vs-actual position pairs across time, not one comparison.

Early iterations of this pipeline accidentally collapsed each storm into a single frame. The fixes below explain how that got caught and corrected.

---

## 3. Pipeline stages

```
NOAA GridSat-B1 (AWS S3)
        |
        v
  aws_downloader.py        -->  data/raw/*.nc + download_manifest.csv
        |
        v
  standardize_data.py      -->  data/normalized/<event>/frames/*.npz + *.json
        |                       + normalized_manifest.csv
        v
  split_ibtracs.py         -->  data/ground_truth/<event>_best_track.csv
        |                       (from raw IBTrACS NI export)
        v
  validate_and_join.py     -->  data/training_manifest.csv   (hand off to ML)
  update_metadata.py       -->  data/metadata.csv             (provenance record)
```

Each stage is independent enough to re-run on its own once its inputs exist, and each writes a manifest so the next stage — and any human — can audit exactly what happened.

### 3.1 `aws_downloader.py` — Ingestion

Connects **anonymously** to NOAA's public `noaa-cdr-gridsat-b1-pds` S3 bucket (no MOSDAC account, no manual approval wait) and pulls one NetCDF file per 3-hourly timestamp across each storm's full lifecycle.

Key design points:
- **Deterministic, parseable filenames**: `<event_id>_<YYYYMMDDTHHMMSSZ>.nc` — e.g. `biparjoy_2023_20230606T000000Z.nc`. Every downstream stage can recover the exact UTC timestamp from the filename with zero ambiguity.
- **Idempotent**: skips re-downloading a file that already exists on disk and is non-empty, so re-running the script after an interruption doesn't waste bandwidth or time.
- **Retries transient network failures** (DNS blips, connection timeouts) with exponential backoff instead of crashing the whole run on one bad request.
- **Incremental manifest writes**: `download_manifest.csv` is written one row at a time and flushed to disk immediately, so a crash mid-run never loses the record of what already succeeded.
- **Efficient fallback search**: if the exact expected S3 key is missing, it falls back to a cached, one-time-per-year index lookup rather than re-listing the entire year's file prefix on every miss.

**Output:** `data/raw/*.nc` + `data/raw/download_manifest.csv` (per-timestamp status: `success` / `skipped_existing` / `missing_from_bucket` / `failed`, and which method found it).

### 3.2 `standardize_data.py` — Preprocessing

Turns raw global NetCDF files into the canonical model input shape the whole team agreed on: **`[C, H, W]`** (channels, height, width).

Steps per file:
1. Open with `xarray`, confirm the IR (`irwin_cdr`) and Water Vapour (`irwvp`) variables actually exist.
2. Crop to the storm's spatial bounding box (different box per basin — Arabian Sea for Biparjoy, Bay of Bengal for Amphan).
3. Clean NaNs, compute per-frame min/max normalization stats, stack IR + WV into one `[2, H, W]` tensor.
4. Save as compressed `.npz`, plus a **sidecar `.json`** per frame recording source, bbox, CRS, resolution, and normalization stats — full per-frame provenance, not just a project-wide assumption.
5. Log every outcome (`PASS` / `REVIEW` / `FAILED`, with a reason) to `normalized_manifest.csv`.

A corrupt or incomplete file doesn't kill the batch — it's logged and skipped, matching the project rule to "reject corrupt files clearly" rather than silently propagating bad data.

**Output:** `data/normalized/<event_id>/frames/*.npz` (+ `.json` sidecars) + `data/normalized/normalized_manifest.csv`.

### 3.3 `split_ibtracs.py` — Ground truth extraction

The official NOAA/NCEI IBTrACS export for the North Indian Ocean basin (`ibtracs.NI.list.v04r00.csv`) contains **every storm in the basin since 1842** in one file, with a units row directly under the header. This script:

- Skips the units row correctly when loading.
- Filters to exactly the two locked events by `NAME` + `SEASON` (`BIPARJOY`/2023, `AMPHAN`/2020).
- Normalizes every timestamp to UTC.
- Prefers **RSMC New Delhi's own reported position** (`NEWDELHI_LAT`/`NEWDELHI_LON`) over the blended multi-agency value, falling back to the blended value only where New Delhi's own report is blank — this matters for an apples-to-apples "model vs. IMD" comparison later.
- Writes one clean CSV per event with the minimal schema the join step needs: `event_id, timestamp, lat, lon` (plus wind/pressure kept for reference).

**Output:** `data/ground_truth/biparjoy_2023_best_track.csv` (83 points) and `data/ground_truth/amphan_2020_best_track.csv` (51 points).

### 3.4 `validate_and_join.py` — The training manifest

This is the file that actually goes to ML. For every standardized frame:

1. Re-validates the array on disk (NaN %, all-zero check) rather than trusting the standardization step's numbers blindly.
2. Joins to the matching ground-truth best-track point using the frame's own timestamp — **nearest match within a 90-minute tolerance window**, not exact equality, since satellite frames and best-track points aren't guaranteed to land on the identical second.
3. Records the match distance in minutes so match quality is auditable, not just assumed.
4. Leaves `pattern_label` as `"unlabeled"` — this script only owns the **position** join. Structural pattern labels (eye / banding / shear-affected / disorganized) are Research's taxonomy deliverable and need a separate join before the classification head is trainable.

**Output:** `data/training_manifest.csv` — one row per usable frame, with `event_id`, `timestamp`, `tensor_shape`, `nan_percentage`, `center_lat`, `center_lon`, `gt_match_distance_min`, `pattern_label`, `validation_status`.

### 3.5 `update_metadata.py` — Provenance record

Builds `data/metadata.csv` for judges/documentation, sourced from `normalized_manifest.csv` and each frame's sidecar JSON — not re-derived from filenames. Records source (`noaa_gridsat_b1`), satellite, product, channel, timestamp, CRS, bounding box, resolution, and an explicit license line (NOAA CDR GridSat-B1 — U.S. Government work, public domain).

---

## 4. Bugs found and fixed along the way

Documenting these honestly, per the project's own principle of not hiding problems:

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Single frame per event instead of a sequence | `2020.npz`, `2023.npz` — one file each | Downloader/standardizer only ever produced one timestamp | Rewrote both to loop over the full 3-hourly event window |
| Ground-truth join returning 100% blank lat/lon | Every row in `training_manifest.csv` had empty `center_lat`/`center_lon` | Timestamp was being regex-parsed out of filenames that didn't contain one (`2020.npz`), or the ground-truth file wasn't named in a way the join could match | Switched join to read timestamps from `normalized_manifest.csv`; fixed filename convention upstream |
| `update_metadata.py` timestamp bug | Would have written `timestamp = "2023"` for every Biparjoy frame | `filename.split("_")[1]` landed on the year fragment because `event_id` itself contains an underscore | Rewrote to source timestamps from `normalized_manifest.csv` / sidecar JSON instead of re-parsing filenames |
| Downloader crash on transient DNS error | `socket.gaierror` mid-run, and all prior successful downloads unlogged | No retry logic; manifest only written after the full loop finished | Added exponential-backoff retries on network calls; manifest now writes incrementally and flushes after every row |
| Raw IBTrACS file couldn't be joined | Ground-truth folder contained the full unfiltered basin file, not per-event CSVs | `ibtracs.NI.list.v04r00.csv` has no underscore prefix to match on, and includes a units row the join script never skipped | Wrote `split_ibtracs.py` to filter, clean, and split it into the two per-event files the pipeline expects |
| Duplicate `RAW_DIR`/`NORMALIZED_DIR` definitions | Silent path bugs depending on working directory | Second variable assignment overwrote the first | Removed the duplicate |

---

## 5. Current handoff status to ML (Aditya)

**Ready now:**
- Full 3-hourly satellite sequences for both events, standardized to `[C, H, W]`
- Real centre-position ground truth joined to every frame (0-minute match distance on spot-checked rows)
- Full provenance chain for every frame

**This means the centre-position regression head and the temporal (T+12/T+24 track) model can be trained on `training_manifest.csv` right now.**

**Not ready yet:**
- `pattern_label` is `"unlabeled"` on every row. The structural pattern classification head (eye / banding / curved-band / shear-affected / disorganized) needs Research's taxonomy labels joined in as a separate step before it's trainable.

---

## 6. How to reproduce this pipeline from scratch

```bash
# 1. Download raw satellite sequences (one .nc per 3-hourly timestamp, per event)
python3 aws_downloader.py

# 2. Standardize into [C, H, W] tensors + sidecar provenance JSON
python3 standardize_data.py

# 3. Split the raw IBTrACS basin file into per-event best-track CSVs
#    (place ibtracs.NI.list.v04r00.csv in data/ground_truth/ first)
python3 split_ibtracs.py

# 4. Join frames to ground truth + build the provenance record (either order)
python3 validate_and_join.py
python3 update_metadata.py
```

If starting from a clean slate, delete `data/raw/*`, `data/normalized/*`, `data/metadata.csv`, and `data/training_manifest.csv` first — but keep `data/ground_truth/` and `data/labels/` (Research's deliverables) untouched.
