#!/usr/bin/env python3
"""Group the near-duplicate candidates from results/split_audit.json by
source-prefix pair and prepare the human review workflow.

Outputs:
  results/near_duplicate_group_summary.csv   -- one row per (source_a, source_b)
  results/near_duplicate_pairs_evidence.csv  -- one row per candidate pair,
                                                 with pHash distance + SSIM
  results/near_duplicate_contact_sheets/<a>__<b>.jpg  -- side-by-side sheets
  data/source_alias_review.csv               -- editable decisions file
                                                 (existing decisions are KEPT)

Frame-offset detection: for every candidate pair with numeric frame
positions, offset = frame_a - frame_b; the modal offset and its support are
reported per group ("framesawit39-N <-> frame1-(N-3)"-style patterns are
strong alias evidence when consistent).

Nothing here decides anything automatically: dHash<=3 pairs are CANDIDATES.
You review the contact sheets, set decision = confirmed_alias /
different_source (or leave uncertain) in data/source_alias_review.csv, then
run scripts/14_build_source_groups.py.
"""
import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# split.py loaded directly (not via `import fedxpalm...`): the package
# __init__ imports torch, which this pure data tool doesn't need.
import importlib.util as _ilu

_split_path = Path(__file__).resolve().parent.parent / "src" / "fedxpalm" / "data" / "split.py"
_spec = _ilu.spec_from_file_location("_fedx_split", _split_path)
_split = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_split)
_parse_source_frame = _split._parse_source_frame

REVIEW_CSV = Path("data/source_alias_review.csv")
REVIEW_FIELDS = ["source_a", "source_b", "decision", "note"]
DECISIONS = ("confirmed_alias", "different_source", "uncertain")


# ---------- extra perceptual evidence (pHash + simplified global SSIM) ----------

def _load_gray(path: Path, size: int):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("L").resize((size, size), Image.LANCZOS)


def phash64(path: Path):
    """64-bit DCT perceptual hash (32x32 -> low-frequency 8x8, median threshold,
    DC excluded). More robust than dHash to blur/contrast changes."""
    import numpy as np
    a = np.asarray(_load_gray(path, 32), dtype=np.float64)
    n = 32
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    dct_mat = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    low = (dct_mat @ a @ dct_mat.T)[:8, :8].flatten()
    coeffs = low[1:]  # drop DC
    med = np.median(coeffs)
    bits = 0
    for v in coeffs:
        bits = (bits << 1) | (1 if v > med else 0)
    return bits


def ssim_global(path_a: Path, path_b: Path) -> float:
    """Single-window (global) SSIM on 64x64 grayscale -- a coarse similarity
    score in [-1, 1], reported as supporting evidence only."""
    import numpy as np
    a = np.asarray(_load_gray(path_a, 64), dtype=np.float64)
    b = np.asarray(_load_gray(path_b, 64), dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                 / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


# ---------- review csv ----------

def load_review(path: Path = REVIEW_CSV) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["source_a"], r["source_b"]): r for r in csv.DictReader(f)}


def merge_review(pairs: list[tuple[str, str]], path: Path = REVIEW_CSV) -> dict:
    """Add any new source pairs as 'uncertain'; NEVER touch existing rows."""
    existing = load_review(path)
    added = 0
    for a, b in pairs:
        if (a, b) not in existing:
            existing[(a, b)] = {"source_a": a, "source_b": b, "decision": "uncertain", "note": ""}
            added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for key in sorted(existing):
            w.writerow({k: existing[key].get(k, "") for k in REVIEW_FIELDS})
    print(f"[review] {path}: {added} new pair(s) added as 'uncertain', "
          f"{len(existing) - added} existing row(s) kept")
    return existing


# ---------- contact sheets ----------

def contact_sheet(group_pairs: list[dict], splits_dir: Path, out_path: Path,
                  max_pairs: int, thumb_h: int = 220) -> None:
    from PIL import Image, ImageDraw

    chosen = group_pairs[:max_pairs]
    thumbs = []
    for p in chosen:
        row_imgs = []
        for side in ("a", "b"):
            split, name = p[side].split("/", 1)
            fp = splits_dir / split / "images" / name
            try:
                with Image.open(fp) as im:
                    im = im.convert("RGB")
                    w = int(im.width * thumb_h / im.height)
                    row_imgs.append(im.resize((w, thumb_h)))
            except Exception:
                ph = Image.new("RGB", (thumb_h, thumb_h), (40, 40, 40))
                ImageDraw.Draw(ph).text((8, 8), "missing:\n" + name, fill=(255, 80, 80))
                row_imgs.append(ph)
        thumbs.append((row_imgs, p))

    cap_h = 42
    row_w = max(a.width + b.width + 24 for (a, b), _ in thumbs)
    row_w = max(row_w, 1000)  # captions are ~100 chars; don't let narrow images clip them
    sheet = Image.new("RGB", (row_w + 16, (thumb_h + cap_h + 12) * len(thumbs) + 8), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    y = 8
    for (im_a, im_b), p in thumbs:
        sheet.paste(im_a, (8, y))
        sheet.paste(im_b, (8 + im_a.width + 24, y))
        cap = (f"{p['a']}  <->  {p['b']}   dHash={p['hamming']}"
               f"   pHash={p.get('phash_dist', '?')}   SSIM={p.get('ssim', '?')}"
               f"   offset={p.get('offset', '?')}")
        draw.text((8, y + thumb_h + 4), cap, fill=(10, 10, 10))
        y += thumb_h + cap_h + 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="results/split_audit.json")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--out-summary", default="results/near_duplicate_group_summary.csv")
    parser.add_argument("--out-pairs", default="results/near_duplicate_pairs_evidence.csv")
    parser.add_argument("--sheets-dir", default="results/near_duplicate_contact_sheets")
    parser.add_argument("--max-sheet-pairs", type=int, default=16, help="pairs per contact sheet (10-20 sensible)")
    parser.add_argument("--no-sheets", action="store_true")
    parser.add_argument("--no-evidence", action="store_true",
                        help="skip pHash/SSIM computation (needs numpy+Pillow)")
    args = parser.parse_args()

    with open(args.audit) as f:
        audit = json.load(f)
    leakage = audit.get("leakage") or {}
    # "raw_dhash_candidates" is the current key; the pre-review-workflow
    # audit format called it "near_duplicates_across_splits"
    candidates = leakage.get("raw_dhash_candidates") or leakage.get("near_duplicates_across_splits") or []
    if not candidates:
        print(f"No near-duplicate candidates in {args.audit} -- nothing to review.")
        return
    with open(args.config) as f:
        splits_dir = Path(yaml.safe_load(f)["output_dir"])

    evidence_ok = not args.no_evidence
    if evidence_ok:
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as e:
            evidence_ok = False
            print(f"[!] {e} -- pHash/SSIM evidence skipped")

    # per-pair enrichment
    pairs = []
    phash_cache: dict[str, int | None] = {}
    for c in candidates:
        (split_a, name_a), (split_b, name_b) = c["a"].split("/", 1), c["b"].split("/", 1)
        src_a, frame_a = _parse_source_frame(name_a)
        src_b, frame_b = _parse_source_frame(name_b)
        # canonical order so (X,Y) and (Y,X) land in the same group
        if (src_b, c["b"]) < (src_a, c["a"]):
            src_a, src_b = src_b, src_a
            frame_a, frame_b = frame_b, frame_a
            c = {"a": c["b"], "b": c["a"], "hamming": c["hamming"]}
        p = dict(c)
        p.update(source_a=src_a, source_b=src_b,
                 offset=(frame_a - frame_b) if frame_a is not None and frame_b is not None else None)
        if evidence_ok:
            fp_a = splits_dir / c["a"].split("/", 1)[0] / "images" / c["a"].split("/", 1)[1]
            fp_b = splits_dir / c["b"].split("/", 1)[0] / "images" / c["b"].split("/", 1)[1]
            try:
                for fp in (str(fp_a), str(fp_b)):
                    if fp not in phash_cache:
                        phash_cache[fp] = phash64(Path(fp))
                ha, hb = phash_cache[str(fp_a)], phash_cache[str(fp_b)]
                p["phash_dist"] = bin(ha ^ hb).count("1") if ha is not None and hb is not None else ""
                p["ssim"] = round(ssim_global(fp_a, fp_b), 3)
            except FileNotFoundError as e:
                p["phash_dist"], p["ssim"] = "", ""
                print(f"[!] image missing, evidence skipped: {e}")
        pairs.append(p)

    # group by source pair
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pairs:
        groups[(p["source_a"], p["source_b"])].append(p)

    review = merge_review(sorted(groups.keys()))

    # group summary rows
    rows = []
    for (a, b), plist in sorted(groups.items()):
        hams = [p["hamming"] for p in plist]
        offsets = [p["offset"] for p in plist if p["offset"] is not None]
        modal_offset, support = ("", 0)
        pattern = ""
        if offsets:
            modal_offset, support = Counter(offsets).most_common(1)[0]
            sign = f"-{modal_offset}" if modal_offset >= 0 else f"+{-modal_offset}"
            pattern = f"{a}-N <-> {b}-(N{sign})"
        rv = review.get((a, b), {})
        rows.append({
            "bunch_id_a": a,
            "split_a": ";".join(sorted({p["a"].split("/", 1)[0] for p in plist})),
            "bunch_id_b": b,
            "split_b": ";".join(sorted({p["b"].split("/", 1)[0] for p in plist})),
            "num_candidate_pairs": len(plist),
            "min_hamming": min(hams),
            "median_hamming": st.median(hams),
            "max_hamming": max(hams),
            "matching_frame_pattern": pattern,
            "suspected_frame_offset": modal_offset if offsets else "",
            "offset_support": f"{support}/{len(offsets)}" if offsets else "",
            "review_status": rv.get("decision", "uncertain"),
            "review_note": rv.get("note", ""),
        })

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(args.out_pairs, "w", newline="") as f:
        fields = ["source_a", "source_b", "a", "b", "hamming", "offset", "phash_dist", "ssim"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)

    # contact sheets: best evidence first (hamming asc, then |offset-modal|, then phash)
    sheets = []
    if not args.no_sheets:
        for (a, b), plist in sorted(groups.items()):
            modal = Counter(p["offset"] for p in plist if p["offset"] is not None)
            modal = modal.most_common(1)[0][0] if modal else None
            plist = sorted(plist, key=lambda p: (
                p["hamming"],
                abs(p["offset"] - modal) if modal is not None and p["offset"] is not None else 99,
                p.get("phash_dist") if isinstance(p.get("phash_dist"), int) else 99,
            ))
            out_path = Path(args.sheets_dir) / f"{a}__{b}.jpg"
            try:
                contact_sheet(plist, splits_dir, out_path, args.max_sheet_pairs)
                sheets.append(str(out_path))
            except ImportError:
                print("[!] Pillow missing -- contact sheets skipped")
                break

    # ---- console summary ----
    n_conf = sum(1 for r in rows if r["review_status"] == "confirmed_alias")
    n_diff = sum(1 for r in rows if r["review_status"] == "different_source")
    n_unc = len(rows) - n_conf - n_diff
    print(f"\n{len(pairs)} candidate pairs -> {len(rows)} source-prefix pair group(s)")
    print(f"{'source_a':<20}{'source_b':<20}{'pairs':>6}{'minH':>6}{'medH':>6}"
          f"{'offset':>8}{'support':>9}  status")
    for r in rows:
        print(f"{r['bunch_id_a']:<20}{r['bunch_id_b']:<20}{r['num_candidate_pairs']:>6}"
              f"{r['min_hamming']:>6}{r['median_hamming']:>6}{str(r['suspected_frame_offset']):>8}"
              f"{r['offset_support']:>9}  {r['review_status']}")
    print(f"\nreview status: {n_conf} confirmed_alias, {n_diff} different_source, {n_unc} uncertain")
    print(f"Saved {out_summary}, {args.out_pairs}")
    if sheets:
        print(f"Contact sheets to review ({len(sheets)}):")
        for s in sheets:
            print(f"  {s}")
    print(f"\nNext: review the sheets, edit decisions in {REVIEW_CSV}, then run "
          f"scripts/14_build_source_groups.py")


if __name__ == "__main__":
    main()
