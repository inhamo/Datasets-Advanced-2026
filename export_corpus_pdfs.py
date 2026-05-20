"""
Export corpus emails (.eml) and news (.md) to realistic PDFs.

Usage:
  py export_corpus_pdfs.py
  py export_corpus_pdfs.py --corpus-dir corpus_context --update-manifest
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from commons.pdf_renderers import render_news_article_pdf, render_outlook_email_pdf

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE_DIR / "corpus_context"


def export_all(corpus_dir: Path, update_manifest: bool) -> dict[str, int]:
    stats = {"emails": 0, "news": 0, "errors": 0}
    new_rows: list[dict[str, str]] = []

    for eml in sorted(corpus_dir.rglob("emails/*.eml")):
        pdf = eml.with_suffix(".pdf")
        try:
            render_outlook_email_pdf(eml, pdf)
            stats["emails"] += 1
            parts = eml.relative_to(corpus_dir).parts
            if len(parts) >= 3:
                year, month = parts[0], parts[1]
                new_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "artifact_type": "email_pdf",
                        "path": str(pdf.relative_to(corpus_dir)),
                        "date": "",
                        "title": eml.stem,
                        "source": "pdf_export",
                    }
                )
        except Exception as exc:
            print(f"FAIL email {eml}: {exc}")
            stats["errors"] += 1

    for md in sorted(corpus_dir.rglob("news/*.md")):
        pdf = md.with_suffix(".pdf")
        try:
            render_news_article_pdf(md, pdf)
            stats["news"] += 1
            parts = md.relative_to(corpus_dir).parts
            if len(parts) >= 3:
                year, month = parts[0], parts[1]
                new_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "artifact_type": "news_pdf",
                        "path": str(pdf.relative_to(corpus_dir)),
                        "date": "",
                        "title": md.stem,
                        "source": "pdf_export",
                    }
                )
        except Exception as exc:
            print(f"FAIL news {md}: {exc}")
            stats["errors"] += 1

    if update_manifest and new_rows:
        manifest = corpus_dir / "index" / "corpus_manifest.csv"
        existing: list[dict[str, str]] = []
        if manifest.exists():
            with manifest.open(encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        existing_paths = {r["path"] for r in existing}
        for row in new_rows:
            if row["path"] not in existing_paths:
                existing.append(row)
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["year", "month", "artifact_type", "path", "date", "title", "source"],
            )
            w.writeheader()
            w.writerows(existing)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export corpus to Outlook/news PDFs.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--update-manifest", action="store_true", default=True)
    args = parser.parse_args()

    if not args.corpus_dir.exists():
        print(f"Corpus not found: {args.corpus_dir}. Run generate_corpus_context.py first.")
        return

    stats = export_all(args.corpus_dir, args.update_manifest)
    print(f"PDF export complete: {stats['emails']} emails, {stats['news']} articles, {stats['errors']} errors")


if __name__ == "__main__":
    main()
