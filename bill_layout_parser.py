"""Bill PDF layout parsing module.

This module exposes a single public API:

    parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict

All parsing logic is intentionally centralized here for easy integration into a
larger internal system. The current implementation focuses on a concrete PDF
preprocessing layer (PyMuPDF-based loading, page rendering, and text block
extraction), while keeping layout-model and Qwen interactions as mockable
placeholders.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import fitz  # pymupdf
except Exception:  # noqa: BLE001
    fitz = None  # type: ignore[assignment]


_ALLOWED_PAGE_TYPES: set[str] = {
    "bill_summary_page",
    "bill_charge_page",
    "bill_detail_page",
}

_ALLOWED_STRUCTURE_TYPES: set[str] = {
    "table",
    "text",
    "kv",
    "image",
}

# Reserved for future strict validation when taxonomy is finalized.
_ALLOWED_REGION_TYPES: set[str] = {
    "bill_charge_page.charge_items",
    "bill_summary_page.summary_info",
    "bill_detail_page.detail_lines",
}


@dataclass(slots=True)
class ParserConfig:
    """Configuration for bill layout parsing."""

    doclayout_url: str = ""
    qwen_url: str = ""
    request_timeout_s: float = 15.0
    include_section_images: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ParserConfig":
        if data is None:
            return cls()
        return cls(
            doclayout_url=str(data.get("doclayout_url", "")),
            qwen_url=str(data.get("qwen_url", "")),
            request_timeout_s=float(data.get("request_timeout_s", 15.0)),
            include_section_images=bool(data.get("include_section_images", False)),
        )


@dataclass(slots=True)
class Block:
    """Unified internal block structure."""

    block_id: str
    text: str
    bbox: list[float]  # absolute coordinates [x0, y0, x1, y1]
    page_index: int
    source: str  # e.g. "pdf_text" / "ocr"


@dataclass(slots=True)
class SectionCandidate:
    """Intermediate section candidate before final output serialization."""

    section_id: str
    region_type: str
    structure_type: str
    bbox: list[float]  # absolute coordinates in page space
    source_block_ids: list[str]
    confidence: float


@dataclass(slots=True)
class PageImage:
    """Rendered page representation for downstream model calls."""

    page_index: int
    width: int
    height: int
    image_format: str
    image_bytes: bytes


@dataclass(slots=True)
class PageParseResult:
    """Internal page-level parse payload."""

    page_index: int
    page_type: str
    width: float
    height: float
    blocks: list[Block]
    section_candidates: list[SectionCandidate]
    page_image: PageImage


class BillLayoutParser:
    """Internal parser implementation for bill PDF layout extraction."""

    def __init__(self, config: ParserConfig) -> None:
        self._config = config

    def parse(self, pdf_path: str) -> dict[str, list[dict[str, Any]]]:
        """Parse a bill PDF file path into structured output."""
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Input must be a .pdf file: {pdf_path}")

        pdf_doc = self._load_pdf(path)
        page_entries: list[dict[str, Any]] = []

        try:
            for page_index in range(pdf_doc.page_count):
                page = pdf_doc.load_page(page_index)
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                page_image = self._render_page(page, page_index)
                blocks = self._extract_pdf_blocks(page, page_index)

                doclayout_result = self._call_doclayout(page_image)
                qwen_result = self._call_qwen(page_image, blocks, doclayout_result)

                section_candidates = self._merge_blocks(blocks, doclayout_result, qwen_result)

                page_result = PageParseResult(
                    page_index=page_index,
                    page_type=str(qwen_result.get("page_type", "bill_charge_page")),
                    width=page_width,
                    height=page_height,
                    blocks=blocks,
                    section_candidates=section_candidates,
                    page_image=page_image,
                )

                sections = self._resolve_sections(page_result)
                final_page_type = (
                    page_result.page_type
                    if page_result.page_type in _ALLOWED_PAGE_TYPES
                    else "bill_charge_page"
                )

                page_entries.append(
                    self._build_output(
                        page_index=page_result.page_index,
                        page_type=final_page_type,
                        sections=sections,
                    )
                )
        finally:
            pdf_doc.close()

        return {"output_data": page_entries}

    def _load_pdf(self, pdf_path: Path) -> Any:
        """Open a PDF using PyMuPDF.

        Raises:
            ValueError: If PDF cannot be opened/parsed.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required but not installed")
        try:
            return fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to open PDF: {pdf_path}") from exc

    def _render_page(self, page: Any, page_index: int) -> PageImage:
        """Render a page to RGB PNG bytes for downstream model calls."""
        pix = page.get_pixmap(colorspace=fitz.csRGB, alpha=False)
        return PageImage(
            page_index=page_index,
            width=pix.width,
            height=pix.height,
            image_format="png",
            image_bytes=pix.tobytes("png"),
        )

    def _extract_pdf_blocks(self, page: Any, page_index: int) -> list[Block]:
        """Extract native text blocks from a page.

        Empty/whitespace-only text blocks are skipped.
        """
        raw_blocks = page.get_text("blocks")
        blocks: list[Block] = []

        for idx, raw in enumerate(raw_blocks):
            if len(raw) < 5:
                continue
            x0, y0, x1, y1, text = raw[:5]
            text_value = str(text).strip()
            if not text_value:
                continue

            blocks.append(
                Block(
                    block_id=f"p{page_index}_b{idx}",
                    text=text_value,
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    page_index=page_index,
                    source="pdf_text",
                )
            )

        return blocks

    def _call_doclayout(self, page_image: PageImage) -> dict[str, Any]:
        """Placeholder doclayout call.

        TODO: Replace with real HTTP client call to PP-DocLayout_plus-L.
        """
        _ = page_image
        return {"regions": []}

    def _call_qwen(
        self,
        page_image: PageImage,
        blocks: list[Block],
        doclayout_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Placeholder Qwen call.

        TODO: Replace with real multimodal request to Qwen endpoint.
        """
        _ = page_image
        _ = blocks
        _ = doclayout_result
        return {
            "page_type": "bill_charge_page",
            "default_structure_type": "text",
        }

    def _merge_blocks(
        self,
        blocks: list[Block],
        doclayout_result: dict[str, Any],
        qwen_result: dict[str, Any],
    ) -> list[SectionCandidate]:
        """Merge PDF blocks and model outputs into section candidates."""
        regions = doclayout_result.get("regions", [])
        candidates: list[SectionCandidate] = []

        for idx, region in enumerate(regions, start=1):
            candidates.append(
                SectionCandidate(
                    section_id=f"s{idx}",
                    region_type=str(region.get("region_type", "bill_charge_page.charge_items")),
                    structure_type=str(region.get("structure_type", qwen_result.get("default_structure_type", "text"))),
                    bbox=[float(v) for v in region.get("bbox", [0.0, 0.0, 0.0, 0.0])],
                    source_block_ids=[b.block_id for b in blocks],
                    confidence=float(region.get("confidence", 0.0)),
                )
            )

        if not candidates and blocks:
            # Fallback candidate to keep downstream output stable.
            x0 = min(b.bbox[0] for b in blocks)
            y0 = min(b.bbox[1] for b in blocks)
            x1 = max(b.bbox[2] for b in blocks)
            y1 = max(b.bbox[3] for b in blocks)
            candidates.append(
                SectionCandidate(
                    section_id="s1",
                    region_type="bill_charge_page.charge_items",
                    structure_type=str(qwen_result.get("default_structure_type", "text")),
                    bbox=[x0, y0, x1, y1],
                    source_block_ids=[b.block_id for b in blocks],
                    confidence=0.5,
                )
            )

        return candidates

    def _resolve_sections(self, page_result: PageParseResult) -> list[dict[str, Any]]:
        """Resolve final section schema with normalized bbox coordinates."""
        sections: list[dict[str, Any]] = []

        for idx, candidate in enumerate(page_result.section_candidates, start=1):
            structure_type = (
                candidate.structure_type
                if candidate.structure_type in _ALLOWED_STRUCTURE_TYPES
                else "text"
            )

            normalized_bbox = self._normalize_bbox(
                candidate.bbox,
                page_width=page_result.width,
                page_height=page_result.height,
            )
            image_b64 = self._crop_section_image_base64(
                page_image=page_result.page_image,
                bbox_normalized=normalized_bbox,
            )

            sections.append(
                {
                    "section_id": f"p{page_result.page_index}_s{idx}",
                    "region_type": candidate.region_type,
                    "structure_type": structure_type,
                    "bbox": normalized_bbox,
                    "source_block_ids": candidate.source_block_ids,
                    "confidence": candidate.confidence,
                    "image_base64": image_b64,
                }
            )

        return sections

    def _crop_section_image_base64(self, page_image: PageImage, bbox_normalized: list[float]) -> str:
        """Return base64 crop data placeholder.

        TODO: Implement true bbox crop from `page_image.image_bytes`.
        """
        _ = bbox_normalized
        if not self._config.include_section_images:
            return ""
        return base64.b64encode(page_image.image_bytes).decode("ascii")

    def _normalize_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        """Normalize absolute bbox to [0, 1], clipping out-of-range values."""
        if len(bbox) != 4:
            raise ValueError("bbox must have exactly 4 elements")
        if page_width <= 0 or page_height <= 0:
            raise ValueError("page dimensions must be positive")

        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [
            max(0.0, min(1.0, x0 / page_width)),
            max(0.0, min(1.0, y0 / page_height)),
            max(0.0, min(1.0, x1 / page_width)),
            max(0.0, min(1.0, y1 / page_height)),
        ]

    def _build_output(
        self,
        page_index: int,
        page_type: Literal["bill_summary_page", "bill_charge_page", "bill_detail_page"],
        sections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build one page output object in final external schema."""
        if page_type not in _ALLOWED_PAGE_TYPES:
            raise ValueError(f"Unsupported page_type: {page_type}")
        return {
            "page_index": page_index,
            "page_type": page_type,
            "page_sections": sections,
        }


def parse_bill_pdf(pdf_path: str, config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Parse bill PDF layout and return structured output.

    This is the only public API exported by this module.
    """
    parser = BillLayoutParser(ParserConfig.from_dict(config))
    return parser.parse(pdf_path)
