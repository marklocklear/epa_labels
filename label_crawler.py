#!/usr/bin/env python3
"""
EPA PPLS Label Crawler (CSV -> EPA JSON -> PDF -> Extract text)

Input:
  - product_with_reg_and_name.csv with at least columns:
      epa_registration_number, product_name

Output JSON format (ExtensionBot/MERLIN):
  [
    {
      "title": "...",
      "link": "https://...pdf",
      "state": "NA",
      "content": [{"content_text": "..."}]
    },
    ...
  ]

Notes:
  - Skips rows when pdffile is missing/blank.
  - Skips PDFs that fail a basic "quality" check (configurable).
  - Skips PDFs that error/403/etc. (does not include them in output).
"""

import csv
import io
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
from pdfminer.high_level import extract_text

# -----------------------------
# Config
# -----------------------------

CSV_FILENAME = "product_with_reg_and_name.csv"

EPA_PPLS_JSON_BASE = "https://ordspub.epa.gov/ords/pesticides/cswu/ppls/"
EPA_PDF_BASE = "https://www3.epa.gov/pesticides/chem_search/ppls/"

USER_AGENT = "ExtensionBot"

# Limit how many CSV rows to process (None = full run)
page_limit: Optional[int] = 10  # e.g. 50 for testing, or None for full crawl

# Networking
REQUEST_TIMEOUT = 10  # seconds (per request)
PDF_DOWNLOAD_TIMEOUT = 20  # seconds

# Quality checks (tune as needed)
MIN_PDF_BYTES = 20_000          # skip tiny PDFs
MAX_PDF_BYTES = 40_000_000      # safety cap (40MB) - skip huge PDFs
QUALITY_CHECK_PAGES = 2         # how many pages to sample for text extraction
MIN_EXTRACTED_CHARS = 400       # if extracted text chars < this -> likely poor/scanned/unreadable
MIN_ALPHA_RATIO = 0.35          # proportion of alphabetic chars in extracted text

# Output
OUTPUT_JSON = "epa_ppls_labels.json"


# -----------------------------
# Helpers
# -----------------------------

def clean_product_name(name: str) -> str:
    # Normalize whitespace for nicer titles
    return re.sub(r"\s+", " ", (name or "").strip())


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def safe_get_json(session: requests.Session, url: str) -> Optional[dict]:
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            print(f"  [skip] JSON fetch failed {r.status_code}: {url}")
            return None
        return r.json()
    except Exception as e:
        print(f"  [skip] JSON error: {url} -> {e}")
        return None


def extract_pdffile(payload: dict) -> Optional[str]:
    """
    EPA ORDS response:
      {"items":[{"pdffiles":[{"pdffile":"....pdf"}]}]}
    But be defensive and handle a few variants.
    """
    if not isinstance(payload, dict):
        return None

    # Some hypothetical direct placements
    direct = payload.get("pdffile")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    items = payload.get("items")
    if isinstance(items, list) and items:
        for it in items:
            if not isinstance(it, dict):
                continue

            # Variant A: pdffile directly on item
            v = it.get("pdffile")
            if isinstance(v, str) and v.strip():
                return v.strip()

            # Variant B (your example): pdffiles is a list of dicts with pdffile
            pf_list = it.get("pdffiles")
            if isinstance(pf_list, list) and pf_list:
                for pf in pf_list:
                    if isinstance(pf, dict):
                        fname = pf.get("pdffile")
                        if isinstance(fname, str) and fname.strip():
                            return fname.strip()

    return None


def head_content_length(session: requests.Session, url: str) -> Optional[int]:
    try:
        r = session.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return None
        cl = r.headers.get("Content-Length")
        if cl is None:
            return None
        return int(cl)
    except Exception:
        return None


def download_pdf(session: requests.Session, url: str) -> Optional[bytes]:
    try:
        r = session.get(url, timeout=PDF_DOWNLOAD_TIMEOUT)
        if r.status_code != 200:
            print(f"  [skip] PDF fetch failed {r.status_code}: {url}")
            return None
        return r.content
    except Exception as e:
        print(f"  [skip] PDF download error: {url} -> {e}")
        return None


def quality_check_pdf(pdf_bytes: bytes) -> Tuple[bool, str]:
    """
    Fast, pragmatic quality check:
      - size bounds
      - sample extracted text from first N pages
      - require minimum extracted characters and a minimum alpha ratio

    Returns (ok, reason).
    """
    nbytes = len(pdf_bytes)
    if nbytes < MIN_PDF_BYTES:
        return False, f"too_small_bytes={nbytes}"
    if nbytes > MAX_PDF_BYTES:
        return False, f"too_large_bytes={nbytes}"

    try:
        # pdfminer can read from a file-like object
        with io.BytesIO(pdf_bytes) as bio:
            text = extract_text(bio, maxpages=QUALITY_CHECK_PAGES) or ""
    except Exception as e:
        return False, f"pdf_text_extract_failed={e}"

    # Normalize and measure
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False, "no_text_extracted"

    chars = len(text)
    alpha = sum(1 for c in text if c.isalpha())
    alpha_ratio = (alpha / chars) if chars else 0.0

    if chars < MIN_EXTRACTED_CHARS:
        return False, f"low_text_chars={chars}"
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False, f"low_alpha_ratio={alpha_ratio:.2f}"

    return True, "ok"


def extract_full_text(pdf_bytes: bytes) -> str:
    """
    Full PDF text extraction. If this fails, returns empty string.
    """
    try:
        with io.BytesIO(pdf_bytes) as bio:
            txt = extract_text(bio) or ""
        # Light cleanup; keep newlines reasonably
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()
    except Exception:
        return ""


# -----------------------------
# Main crawl
# -----------------------------

def main() -> None:
    if not os.path.exists(CSV_FILENAME):
        print(f"[error] CSV not found: {CSV_FILENAME}")
        sys.exit(1)

    session = build_session()

    results: List[Dict] = []
    processed = 0
    added = 0
    skipped = 0

    with open(CSV_FILENAME, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if page_limit is not None and processed >= page_limit:
                break

            processed += 1
            reg = (row.get("epa_registration_number") or "").strip()
            product_name = clean_product_name(row.get("product_name") or "")

            print(f"\n[{processed}] reg={reg} name={product_name[:80]}")
            if not reg:
                print("  [skip] missing epa_registration_number")
                skipped += 1
                continue

            # 1) Fetch ORDS JSON
            ords_url = f"{EPA_PPLS_JSON_BASE}{reg}"
            payload = safe_get_json(session, ords_url)
            if not payload:
                skipped += 1
                continue

            pdffile = extract_pdffile(payload)
            if not pdffile:
                print("  [skip] pdffile empty")
                skipped += 1
                continue

            # 2) Build PDF URL
            pdf_url = f"{EPA_PDF_BASE}{pdffile}"
            print(f"  pdf: {pdf_url}")

            # 3) (Optional) HEAD size check first
            cl = head_content_length(session, pdf_url)
            if cl is not None:
                if cl < MIN_PDF_BYTES:
                    print(f"  [skip] HEAD says too small: {cl} bytes")
                    skipped += 1
                    continue
                if cl > MAX_PDF_BYTES:
                    print(f"  [skip] HEAD says too large: {cl} bytes")
                    skipped += 1
                    continue

            # 4) Download PDF
            pdf_bytes = download_pdf(session, pdf_url)
            if not pdf_bytes:
                skipped += 1
                continue

            # 5) Quality gate
            ok, reason = quality_check_pdf(pdf_bytes)
            if not ok:
                print(f"  [skip] quality_check failed: {reason}")
                skipped += 1
                continue

            # 6) Extract full text
            text = extract_full_text(pdf_bytes)
            if not text:
                print("  [skip] extracted empty text after passing quality (rare)")
                skipped += 1
                continue

            # 7) Build output record (ExtensionBot format)
            record = {
                "title": product_name or reg,
                "link": pdf_url,
                "epa_registration_number": reg,
                "state": "NA",
                "content": [
                    {"content_text": text}
                ],
            }
            results.append(record)
            added += 1
            print(f"  [added] total_added={added}")

            # Polite pacing helps reduce upstream throttling in large runs
            time.sleep(0.1)

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)

    print("\n====================")
    print("Crawl complete")
    print(f"Pages processed: {processed}")
    print(f"Pages added:     {added}")
    print(f"Pages skipped:   {skipped}")
    print(f"Output:          {OUTPUT_JSON}")
    print("====================")


if __name__ == "__main__":
    main()
