"""Render Outlook-style emails and news articles as PDF."""

from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTLOOK_BLUE = colors.HexColor("#0078D4")
OUTLOOK_HEADER_BG = colors.HexColor("#F3F2F1")
OUTLOOK_BORDER = colors.HexColor("#EDEBE9")
NEWS_MASTHEAD = {
    "fin24": ("Fin24", colors.HexColor("#003366")),
    "business_day": ("Business Day", colors.HexColor("#1A1A1A")),
    "news24": ("News24", colors.HexColor("#C41230")),
    "daily_maverick": ("Daily Maverick", colors.HexColor("#2E2E2E")),
    "mg": ("Mail & Guardian", colors.HexColor("#8B0000")),
    "moneyweb": ("Moneyweb", colors.HexColor("#005A8C")),
}


def _md_to_reportlab(text: str) -> str:
    """Convert light markdown to ReportLab paragraph markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    text = text.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
    return text


def _parse_eml(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    body = msg.get_body(preferencelist=("plain",))
    text = body.get_content() if body else ""
    if isinstance(text, bytes):
        text = text.decode(body.get_content_charset() or "utf-8", errors="replace")
    return {
        "from": str(msg.get("From", "")),
        "to": str(msg.get("To", "")),
        "cc": str(msg.get("Cc", "") or ""),
        "subject": str(msg.get("Subject", "")),
        "date": str(msg.get("Date", "")),
        "body": text.strip(),
    }


def _parse_news_md(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
            body = parts[2].strip()
    return {
        "outlet": meta.get("outlet", "News"),
        "outlet_id": meta.get("outlet_id", ""),
        "author": meta.get("author", "Staff Reporter"),
        "published": meta.get("published", ""),
        "title": meta.get("title", path.stem),
        "department": meta.get("department", ""),
        "project_type": meta.get("project_type", ""),
        "style": meta.get("style", ""),
        "image_path": meta.get("image_path", ""),
        "image_alt": meta.get("image_alt", ""),
        "body": body,
    }


def _resolve_news_image(md_path: Path, image_path: str) -> Path | None:
    if not image_path:
        return None
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for parent in [md_path.parent, *md_path.parents]:
        resolved = parent / image_path
        if resolved.exists():
            return resolved
    return None


def render_outlook_email_pdf(eml_path: Path, pdf_path: Path | None = None) -> Path:
    """Render an .eml file as an Outlook-like PDF."""
    pdf_path = pdf_path or eml_path.with_suffix(".pdf")
    data = _parse_eml(eml_path)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=data["subject"][:80],
        author=data["from"],
    )

    styles = getSampleStyleSheet()
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#605E5C"),
    )
    value_style = ParagraphStyle(
        "Value",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#323130"),
        leading=13,
    )
    subject_style = ParagraphStyle(
        "Subject",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#201F1E"),
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#201F1E"),
        leading=15,
        spaceBefore=12,
    )
    banner_style = ParagraphStyle(
        "Banner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.white,
    )

    header_tbl = Table(
        [[Paragraph("Outlook", banner_style)]],
        colWidths=[doc.width],
    )
    header_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), OUTLOOK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story = [header_tbl, Spacer(1, 6)]

    def field_row(label: str, value: str) -> None:
        if not value:
            return
        tbl = Table(
            [
                [
                    Paragraph(label, label_style),
                    Paragraph(value.replace("\n", "<br/>"), value_style),
                ]
            ],
            colWidths=[2.2 * cm, doc.width - 2.2 * cm],
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), OUTLOOK_HEADER_BG),
                    ("BOX", (0, 0), (-1, -1), 0.5, OUTLOOK_BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(tbl)

    field_row("From", data["from"])
    field_row("To", data["to"])
    field_row("Cc", data["cc"])
    field_row("Sent", data["date"])

    story.append(Spacer(1, 8))
    story.append(Paragraph(_md_to_reportlab(data["subject"]), subject_style))
    story.append(HRFlowable(width="100%", thickness=1, color=OUTLOOK_BORDER))
    story.append(Paragraph(_md_to_reportlab(data["body"]), body_style))

    doc.build(story)
    return pdf_path


def render_news_article_pdf(md_path: Path, pdf_path: Path | None = None) -> Path:
    """Render a clean bulletin-style news article PDF."""
    pdf_path = pdf_path or md_path.with_suffix(".pdf")
    art = _parse_news_md(md_path)
    outlet_id = art.get("outlet_id", "")
    masthead, accent = NEWS_MASTHEAD.get(outlet_id, (art["outlet"], colors.HexColor("#333333")))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.6 * cm,
        title=art["title"],
    )

    masthead_style = ParagraphStyle("Masthead", fontName="Helvetica-Bold", fontSize=24, leading=26, textColor=colors.HexColor("#111827"), spaceAfter=2)
    tagline_style = ParagraphStyle("Tagline", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#6B7280"), spaceAfter=8)
    eyebrow_style = ParagraphStyle("Eyebrow", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white, alignment=TA_LEFT)
    headline_style = ParagraphStyle("Headline", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=colors.HexColor("#111827"), spaceAfter=8)
    byline_style = ParagraphStyle("Byline", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#6B7280"), spaceAfter=10)
    dek_style = ParagraphStyle("Dek", fontName="Helvetica", fontSize=12, leading=17, textColor=colors.HexColor("#374151"), spaceAfter=14)
    body_style = ParagraphStyle("ArticleBody", fontName="Helvetica", fontSize=10.5, leading=15, alignment=TA_LEFT, textColor=colors.HexColor("#1F2937"), spaceAfter=10)
    card_style = ParagraphStyle("Card", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#374151"))
    footer_style = ParagraphStyle("Footer", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#888888"), alignment=TA_LEFT)

    pub_display = art["published"]
    try:
        from datetime import datetime

        pub_display = datetime.fromisoformat(art["published"]).strftime("%d %B %Y")
    except ValueError:
        pass

    story: list[Any] = []
    story.append(Paragraph("Buletin", masthead_style))
    story.append(Paragraph(f"{masthead} / Business, Finance and Digital Banking", tagline_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#111827")))
    story.append(Spacer(1, 12))

    department = art.get("department", "")
    project_type = art.get("project_type", "")
    meta_label = " / ".join([x for x in [department, project_type.replace("_", " ").title() if project_type else ""] if x])
    if meta_label:
        chip = Table([[Paragraph(meta_label, eyebrow_style)]], colWidths=[doc.width])
        chip.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), accent), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(chip)
        story.append(Spacer(1, 10))

    story.append(Paragraph(_md_to_reportlab(art["title"]), headline_style))
    story.append(Paragraph(f"By {_md_to_reportlab(art['author'])} &nbsp;|&nbsp; {pub_display}", byline_style))

    paragraphs = [p.strip() for p in art["body"].split("\n\n") if p.strip()]
    if paragraphs:
        story.append(Paragraph(_md_to_reportlab(paragraphs[0]), dek_style))
    if len(paragraphs) > 1:
        summary_card = Table(
            [[Paragraph("<b>Sector signal</b><br/>External news item for analytics practice.", card_style), Paragraph(f"<b>Project link</b><br/>{_md_to_reportlab(meta_label or 'General banking analytics')}", card_style)]],
            colWidths=[doc.width * 0.48, doc.width * 0.48],
        )
        summary_card.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F4F6")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")), ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story.append(summary_card)
        story.append(Spacer(1, 14))
    for para in paragraphs[1:]:
        story.append(Paragraph(_md_to_reportlab(para), body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="30%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(f"(c) {masthead}. Synthetic bulletin article for data pipeline and project-brief exercises.", footer_style))

    doc.build(story)
    return pdf_path


def render_news_article_pdf(md_path: Path, pdf_path: Path | None = None) -> Path:
    """Render a news markdown file as a printable article PDF."""
    pdf_path = pdf_path or md_path.with_suffix(".pdf")
    art = _parse_news_md(md_path)
    outlet_id = art.get("outlet_id", "")
    masthead, accent = NEWS_MASTHEAD.get(outlet_id, (art["outlet"], colors.HexColor("#333333")))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title=art["title"],
    )

    styles = getSampleStyleSheet()
    masthead_style = ParagraphStyle(
        "Masthead",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=accent,
        spaceAfter=2,
    )
    tagline_style = ParagraphStyle(
        "Tagline",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=colors.HexColor("#666666"),
        spaceAfter=14,
    )
    headline_style = ParagraphStyle(
        "Headline",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111111"),
        spaceAfter=10,
    )
    byline_style = ParagraphStyle(
        "Byline",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=16,
    )
    body_style = ParagraphStyle(
        "ArticleBody",
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#222222"),
        spaceAfter=10,
    )
    footer_style = ParagraphStyle(
        "Footer",
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        alignment=TA_LEFT,
    )
    caption_style = ParagraphStyle(
        "ImageCaption",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#666666"),
        spaceAfter=12,
    )

    pub_display = art["published"]
    try:
        from datetime import datetime

        pub_display = datetime.fromisoformat(art["published"]).strftime("%d %B %Y")
    except ValueError:
        pass

    story: list[Any] = []
    story.append(Paragraph(masthead, masthead_style))
    story.append(Paragraph("Business &amp; Finance — South Africa", tagline_style))
    story.append(HRFlowable(width="100%", thickness=2, color=accent))
    story.append(Spacer(1, 10))
    story.append(Paragraph(_md_to_reportlab(art["title"]), headline_style))
    story.append(
        Paragraph(
            f"By {_md_to_reportlab(art['author'])} &nbsp;|&nbsp; {pub_display}",
            byline_style,
        )
    )
    image_file = _resolve_news_image(md_path, art.get("image_path", ""))
    if image_file:
        story.append(Image(str(image_file), width=doc.width, height=doc.width * 0.42))
        if art.get("image_alt"):
            story.append(Paragraph(_md_to_reportlab(art["image_alt"]), caption_style))
        else:
            story.append(Spacer(1, 10))

    for para in art["body"].split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(_md_to_reportlab(para), body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="30%", thickness=0.5, color=colors.lightgrey))
    story.append(
        Paragraph(
            f"© {masthead}. Synthetic training article for data pipeline exercises.",
            footer_style,
        )
    )

    doc.build(story)
    return pdf_path
