# fcc_ecfs.py
# Helper utilities to search and download FCC ECFS PDF filings
# All comments in English as requested.

from __future__ import annotations

import io
import os
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import pdfplumber            # pip install pdfplumber
import requests              # pip install requests

API_KEY = os.getenv("FCC_API_KEY", "CHANGE_ME")            # FCC Public API key
SEARCH_URL = "https://publicapi.fcc.gov/ecfs/filings"
TIMEOUT = 30
ROOT_DIR = Path("downloads")                               # base output folder


def _build_url(src: str, fname: str) -> str:
    """Return a direct PDF URL with ?download=1&filename=…"""
    url = src.replace("/document/", "/documents/", 1)
    return f"{url}?download=1&filename={urllib.parse.quote(fname)}"


def _sanitize(name: str) -> str:
    keep = "-_.() "
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)


def _fetch_filings(company: str, batch: int = 25) -> List[dict]:
    filings, offset = [], 0
    while True:
        resp = requests.get(
            SEARCH_URL,
            params={
                "api_key": API_KEY,
                "q": company,
                "limit": batch,
                "sort": "date_received,DESC",
                "offset": offset,
            },
            headers={"accept": "application/json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        chunk = resp.json().get("filing", [])
        if not chunk:
            break
        filings += chunk
        if len(chunk) < batch:
            break
        offset += batch
    return filings


def search(company: str) -> List[Dict]:
    """
    Return a numbered list with minimal metadata for every PDF attachment.

    Each element:
        {
          "idx": 1-based index,
          "date": "YYYY-MM-DD",
          "filename": "Some.pdf",
          "url": "https://…?download=1&filename=Some.pdf"
        }
    """
    filings = _fetch_filings(company)
    items: List[Tuple[str, str, str]] = []
    for f in filings:
        date = (f.get("date_received") or f.get("date_disseminated", ""))[:10]
        for d in f.get("documents", []):
            fname = d.get("filename", "")
            if fname.lower().endswith(".pdf"):
                items.append((_build_url(d["src"], fname), fname, date))

    # enumerate & build list of dicts
    out: List[Dict] = []
    for idx, (url, fname, date) in enumerate(items, start=1):
        out.append({"idx": idx, "date": date, "filename": fname, "url": url})
    return out


def _download_pdf(url: str) -> io.BytesIO:
    r = requests.get(url, headers={"accept": "application/pdf"}, timeout=TIMEOUT)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("application/pdf"):
        raise RuntimeError("URL did not return a PDF")
    return io.BytesIO(r.content)


def _pdf_to_text(buf: io.BytesIO) -> str:
    with pdfplumber.open(buf) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def get_texts(company: str, choices: List[int]) -> Dict[str, str]:
    """
    Download and convert the selected PDFs to text.

    Parameters
    ----------
    company : str
    choices : list[int] – list of 1-based indexes as returned by search()

    Returns
    -------
    dict: mapping "<idx>. <original filename>" → "<plain text>"
    """
    catalog = search(company)
    sel = [item for item in catalog if item["idx"] in choices]
    if not sel:
        raise ValueError("No matching indexes.")

    # create output dir once
    dst = ROOT_DIR / _sanitize(company)
    dst.mkdir(parents=True, exist_ok=True)

    dup = Counter()
    out: Dict[str, str] = {}

    for item in sel:
        basename = Path(_sanitize(item["filename"])).stem
        tag = f"{item['date']}_{item['idx']:03d}_{basename}"
        dup[tag] += 1
        unique = tag if dup[tag] == 1 else f"{dup[tag]}_{tag}"
        txt_path = dst / f"{unique}.txt"

        text = _pdf_to_text(_download_pdf(item["url"]))
        txt_path.write_text(text, encoding="utf-8")
        out[f"{item['idx']}. {item['filename']}"] = text

    # save/update index file
    idx_file = dst / "pdf_index.tsv"
    mode = "a" if idx_file.exists() else "w"
    with idx_file.open(mode, encoding="utf-8") as fh:
        if mode == "w":
            fh.write("timestamp\tcompany\tfilename\turl\n")
        ts = datetime.utcnow().isoformat(timespec="seconds")
        for item in sel:
            fh.write(f"{ts}\t{company}\t{item['filename']}\t{item['url']}\n")

    return out
