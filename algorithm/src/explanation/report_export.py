from __future__ import annotations
from pathlib import Path
from typing import Iterable
import markdown as md
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def _header_footer(canvas, doc) -> None:
    canvas.saveState()

    width, height = A4

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)

    canvas.drawString(2 * cm, 1.2 * cm, "Smart Developer")
    canvas.drawRightString(width - 2 * cm, 1.2 * cm, f"Page {doc.page}")

    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    styles = {
        "h1": ParagraphStyle(
            "CustomH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "CustomH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "CustomH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "CustomSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.grey,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "CustomBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=4,
        ),
    }

    return styles


def _clean_inline_html(text: str) -> str:
    return (
        text.replace("<strong>", "<b>")
        .replace("</strong>", "</b>")
        .replace("<em>", "<i>")
        .replace("</em>", "</i>")
    )


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    text = _clean_inline_html(text)
    return Paragraph(text, style)


def _table_from_html(table_tag, styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[str]] = []

    for tr in table_tag.find_all("tr"):
        row: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            row.append(cell.get_text(" ", strip=True))
        if row:
            rows.append(row)

    if not rows:
        return Table([[""]])

    # Convert strings to Paragraphs so long text wraps.
    wrapped_rows = [
        [_paragraph(str(cell), styles["small"]) for cell in row]
        for row in rows
    ]

    col_count = max(len(row) for row in rows)

    # Reasonable widths for the current report table.
    if col_count == 6:
        col_widths = [6.0 * cm, 1.4 * cm, 1.5 * cm, 2.6 * cm, 2.0 * cm, 2.0 * cm]
    else:
        usable_width = A4[0] - 4 * cm
        col_widths = [usable_width / col_count] * col_count

    table = Table(wrapped_rows, colWidths=col_widths, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table


def _html_to_flowables(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    styles = _styles()
    flowables = []

    body = soup.body or soup

    for elem in body.children:
        if getattr(elem, "name", None) is None:
            text = str(elem).strip()
            if text:
                flowables.append(_paragraph(text, styles["body"]))
            continue

        name = elem.name

        if name == "h1":
            flowables.append(_paragraph(elem.decode_contents(), styles["h1"]))
        elif name == "h2":
            flowables.append(_paragraph(elem.decode_contents(), styles["h2"]))
        elif name == "h3":
            flowables.append(_paragraph(elem.decode_contents(), styles["h3"]))
        elif name == "p":
            text = elem.decode_contents().strip()
            if text:
                if text.startswith("<em>") or text.startswith("<i>"):
                    flowables.append(_paragraph(text, styles["small"]))
                else:
                    flowables.append(_paragraph(text, styles["body"]))
        elif name == "ul":
            for li in elem.find_all("li", recursive=False):
                flowables.append(
                    Paragraph(
                        _clean_inline_html(li.decode_contents()),
                        styles["bullet"],
                        bulletText="-",
                    )
                )
        elif name == "table":
            flowables.append(_table_from_html(elem, styles))
            flowables.append(Spacer(1, 8))
        elif name == "hr":
            flowables.append(Spacer(1, 8))
        else:
            text = elem.get_text(" ", strip=True)
            if text:
                flowables.append(_paragraph(text, styles["body"]))

    return flowables


def export_markdown_report_to_pdf(
    markdown_text: str,
    output_pdf_path: str | Path,
) -> Path:
    """
    Export a markdown Smart Developer report to PDF.

    This is intentionally simple and dependency-light:
    markdown -> HTML -> ReportLab PDF.
    """
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = md.markdown(markdown_text, extensions=["tables", "sane_lists"])
    flowables = _html_to_flowables(html)

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
    )

    template = PageTemplate(
        id="main",
        frames=[frame],
        onPage=_header_footer,
    )

    doc.addPageTemplates([template])
    doc.build(flowables)

    return output_path