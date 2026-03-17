"""Bill PDF layout parser.

This module intentionally exposes a single public function:

    parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict

All parsing internals (PDF preprocessing, optional DocLayout call, optional Qwen
call, section post-processing, and fallback logic) are centralized in this file.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

try:
    import fitz  # pymupdf
except Exception:  # noqa: BLE001
    fitz = None  # type: ignore[assignment]

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

_ALLOWED_PAGE_TYPES: set[str] = {
    "bill_summary_page",
    "bill_charge_page",
    "bill_detail_page",
}

_ALLOWED_STRUCTURE_TYPES: set[str] = {"table", "text", "kv", "image"}

_ALLOWED_REGION_TYPES: set[str] = {
    "page_footer",
    "page_number",
    "bill_summary_page.telecom_operator_information",
    "bill_summary_page.marketing_information",
    "bill_summary_page.account_information",
    "bill_summary_page.address_and_contact_information",
    "bill_summary_page.barcode_qr_code",
    "bill_charge_page.charge_items",
    "bill_charge_page.adjustment_charges",
    "bill_charge_page.installment_payment",
    "bill_charge_page.transaction_information",
    "bill_charge_page.invoice_remark_and_explanations",
    "bill_detail_page.detail_record_display_content",
}


@dataclass(slots=True)
class ParserConfig:
    """Runtime configuration for parser internals."""

    doclayout_url: str = ""
    qwen_url: str = ""
    qwen_model_name: str = "qwen3.5-35b"
    api_key: str = ""
    request_timeout_s: float = 15.0
    headers: dict[str, str] | None = None
    include_section_images: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ParserConfig":
        """Build config from external input dictionary."""
        if data is None:
            return cls()
        timeout_val = data.get("timeout", data.get("request_timeout_s", 15.0))
        headers = data.get("headers") if isinstance(data.get("headers"), dict) else None
        return cls(
            doclayout_url=str(data.get("doclayout_url", "")),
            qwen_url=str(data.get("qwen_url", "")),
            qwen_model_name=str(data.get("qwen_model_name", "qwen3.5-35b")),
            api_key=str(data.get("api_key", "")),
            request_timeout_s=float(timeout_val),
            headers={str(k): str(v) for k, v in (headers or {}).items()} or None,
            include_section_images=bool(data.get("include_section_images", True)),
        )


@dataclass(slots=True)
class Block:
    """Unified block representation used across PDF/OCR/model outputs."""

    block_id: str
    text: str
    bbox: list[float]  # absolute [x1, y1, x2, y2]
    page_index: int
    source: str


@dataclass(slots=True)
class SectionCandidate:
    """Internal section candidate structure before final serialization."""

    section_id: str
    region_type: str
    structure_type: str
    bbox: list[float]  # absolute [x1, y1, x2, y2]
    source_block_ids: list[str]
    confidence: float


@dataclass(slots=True)
class PageImage:
    """Rendered page image container for downstream model calls and crops."""

    page_index: int
    width: int
    height: int
    image_format: str
    image_bytes: bytes
    rgb_bytes: bytes
    channels: int


@dataclass(slots=True)
class PageParseResult:
    """Per-page intermediate parse result."""

    page_index: int
    page_type: str
    width: float
    height: float
    blocks: list[Block]
    section_candidates: list[SectionCandidate]
    page_image: PageImage


class QwenResponseParseError(ValueError):
    """Raised when Qwen text response cannot be converted to JSON object."""


class BillLayoutParser:
    """Internal parser engine that powers :func:`parse_bill_pdf`."""

    def __init__(self, config: ParserConfig) -> None:
        self._config = config

    def parse(self, pdf_path: str) -> dict[str, list[dict[str, Any]]]:
        """Parse a bill PDF file path into the final structured dictionary."""
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Input must be a .pdf file: {pdf_path}")

        doc = self._load_pdf(path)
        output_data: list[dict[str, Any]] = []

        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                page_image = self._render_page(page, page_index)
                pdf_blocks = self._extract_pdf_blocks(page, page_index)
                doclayout_result = self._call_doclayout(page_image)
                merged_blocks = self._merge_pdf_and_ocr_blocks(pdf_blocks, doclayout_result.get("ocr_blocks", []))

                try:
                    qwen_result = self._call_qwen(page_image, merged_blocks, doclayout_result)
                    page_type = str(qwen_result.get("page_type", "bill_summary_page"))
                    section_candidates = qwen_result.get("sections", [])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Qwen failed; fallback applied: %s", exc)
                    page_type = self._infer_page_type(merged_blocks)
                    section_candidates = []

                if page_type not in _ALLOWED_PAGE_TYPES:
                    page_type = self._infer_page_type(merged_blocks)

                page_result = PageParseResult(
                    page_index=page_index,
                    page_type=page_type,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=merged_blocks,
                    section_candidates=section_candidates,
                    page_image=page_image,
                )
                page_sections = self._resolve_sections(page_result)
                output_data.append(
                    {
                        "page_index": page_index,
                        "page_type": page_type,
                        "page_sections": page_sections,
                    }
                )
        finally:
            doc.close()

        return {"output_data": output_data}

    def _load_pdf(self, pdf_path: Path) -> Any:
        """Open PDF with PyMuPDF; raise clean error when it fails."""
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required but not installed")
        try:
            return fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to open PDF: {pdf_path}") from exc

    def _render_page(self, page: Any, page_index: int) -> PageImage:
        """Render page to RGB image container used by downstream calls and crops."""
        pix = page.get_pixmap(colorspace=fitz.csRGB, alpha=False) if fitz is not None else page.get_pixmap(alpha=False)
        return PageImage(
            page_index=page_index,
            width=int(pix.width),
            height=int(pix.height),
            image_format="png",
            image_bytes=pix.tobytes("png"),
            rgb_bytes=bytes(getattr(pix, "samples", b"")),
            channels=int(getattr(pix, "n", 3) or 3),
        )

    def _extract_pdf_blocks(self, page: Any, page_index: int) -> list[Block]:
        """Extract native PDF text blocks and filter empty text."""
        blocks: list[Block] = []
        for idx, raw in enumerate(page.get_text("blocks")):
            if len(raw) < 5:
                continue
            x1, y1, x2, y2, text = raw[:5]
            txt = str(text).strip()
            if not txt:
                continue
            blocks.append(Block(f"p{page_index}_b{idx}", txt, [float(x1), float(y1), float(x2), float(y2)], page_index, "pdf_text"))
        return blocks

    def _image_to_base64(self, page_image: PageImage) -> str:
        """Encode page PNG bytes to base64 string."""
        return base64.b64encode(page_image.image_bytes).decode("ascii")

    def _call_doclayout(self, page_image: PageImage) -> dict[str, Any]:
        """Call DocLayout service and return normalized layout/ocr payload."""
        if not self._config.doclayout_url or requests is None:
            return {"layout_blocks": [], "ocr_blocks": []}

        headers = {"Content-Type": "application/json", **(self._config.headers or {})}
        payload = {
            "image_base64": self._image_to_base64(page_image),
            "page_index": page_image.page_index,
            "image_format": page_image.image_format,
        }

        try:
            response = requests.post(
                self._config.doclayout_url,
                json=payload,
                headers=headers,
                timeout=self._config.request_timeout_s,
            )
            response.raise_for_status()
            return self._parse_doclayout_response(response.json(), page_image.page_index)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DocLayout call failed: %s", exc)
            return {"layout_blocks": [], "ocr_blocks": []}

    def _parse_doclayout_response(self, response_json: dict[str, Any] | list[Any] | str, page_index: int) -> dict[str, Any]:
        """Parse possibly unstable DocLayout response into canonical fields."""
        data: Any = response_json
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:  # noqa: BLE001
                return {"layout_blocks": [], "ocr_blocks": []}

        if isinstance(data, list):
            data = {"layout_blocks": data}
        if not isinstance(data, dict):
            return {"layout_blocks": [], "ocr_blocks": []}

        layout_raw = data.get("layout_blocks") or data.get("layout") or data.get("regions") or data.get("detections") or []
        ocr_raw = data.get("ocr_blocks") or data.get("ocr") or data.get("text_blocks") or data.get("blocks") or []

        layout_blocks: list[dict[str, Any]] = []
        for item in layout_raw if isinstance(layout_raw, list) else []:
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox(item)
            if not bbox:
                continue
            layout_blocks.append(
                {
                    "region_type": str(item.get("region_type") or item.get("label") or "bill_charge_page.charge_items"),
                    "structure_type": str(item.get("structure_type") or item.get("type") or "table"),
                    "bbox": bbox,
                    "confidence": float(item.get("confidence", item.get("score", 0.0)) or 0.0),
                }
            )

        ocr_blocks: list[Block] = []
        for idx, item in enumerate(ocr_raw if isinstance(ocr_raw, list) else []):
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox(item)
            text = str(item.get("text", item.get("content", ""))).strip()
            if not bbox or not text:
                continue
            ocr_blocks.append(Block(f"p{page_index}_ocr{idx}", text, bbox, page_index, "ocr"))

        return {"layout_blocks": layout_blocks, "ocr_blocks": ocr_blocks}

    def _build_qwen_prompt(self, page_index: int, blocks: list[Block], doclayout_result: dict[str, Any]) -> tuple[str, str]:
        """Build stable system/user prompts with explicit schema and label constraints."""
        system_prompt = (
            "You are a bill layout parser. Return ONLY valid JSON and strictly follow allowed labels."
        )
        block_preview = [
            {
                "block_id": b.block_id,
                "text": b.text[:160],
                "bbox": b.bbox,
                "source": b.source,
            }
            for b in blocks[:80]
        ]
        user_prompt = (
            "Task: infer page_type and section candidates.\n"
            "Output JSON schema:\n"
            "{\n"
            '  "page_type": "bill_summary_page|bill_charge_page|bill_detail_page",\n'
            '  "sections": [\n'
            "    {\n"
            '      "region_type": "<allowed>",\n'
            '      "structure_type": "table|text|kv|image",\n'
            '      "bbox": [x1, y1, x2, y2],\n'
            '      "source_block_ids": ["..."],\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Allowed page_type: {sorted(_ALLOWED_PAGE_TYPES)}\n"
            f"Allowed region_type: {sorted(_ALLOWED_REGION_TYPES)}\n"
            f"Allowed structure_type: {sorted(_ALLOWED_STRUCTURE_TYPES)}\n"
            "Rules:\n"
            "- bbox should be absolute page coordinates [x1,y1,x2,y2].\n"
            "- structure_type and region_type are mandatory for every section.\n"
            "- do NOT create any new labels.\n"
            "- overlapping candidates are allowed.\n"
            "- ignore unrelated noise.\n\n"
            f"page_index={page_index}\n"
            f"blocks={json.dumps(block_preview, ensure_ascii=False)}\n"
            f"layout={json.dumps(doclayout_result.get('layout_blocks', []), ensure_ascii=False)}"
        )
        return system_prompt, user_prompt

    def _call_qwen(self, page_image: PageImage, blocks: list[Block], doclayout_result: dict[str, Any]) -> dict[str, Any]:
        """Call Qwen endpoint (OpenAI-style payload) and parse output into candidates."""
        if not self._config.qwen_url:
            return {"page_type": self._infer_page_type(blocks), "sections": []}
        if requests is None:
            raise RuntimeError("requests is required for qwen calls")

        system_prompt, user_prompt = self._build_qwen_prompt(page_image.page_index, blocks, doclayout_result)
        payload = {
            "model": self._config.qwen_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(page_image)}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json", **(self._config.headers or {})}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        response = requests.post(
            self._config.qwen_url,
            json=payload,
            headers=headers,
            timeout=self._config.request_timeout_s,
        )
        response.raise_for_status()
        data = response.json()

        content = ""
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data.get("choices"):
            message = data["choices"][0].get("message", {}) if isinstance(data["choices"][0], dict) else {}
            raw_content = message.get("content", "")
            if isinstance(raw_content, list):
                content = "\n".join(str(x.get("text", "")) for x in raw_content if isinstance(x, dict))
            else:
                content = str(raw_content)
        elif isinstance(data, dict):
            content = str(data.get("output_text", data.get("text", "")))

        parsed_json = self._extract_json_from_llm_response(content)
        return self._parse_qwen_sections(parsed_json, page_image.page_index)

    def _extract_json_from_llm_response(self, llm_text: str) -> dict[str, Any]:
        """Extract JSON object from raw LLM text, including markdown code fences."""
        text = llm_text.strip()
        if not text:
            raise QwenResponseParseError("Empty qwen response")

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = fence_match.group(1) if fence_match else text

        if not fence_match:
            left = candidate.find("{")
            right = candidate.rfind("}")
            if left != -1 and right != -1 and left < right:
                candidate = candidate[left : right + 1]

        try:
            parsed = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            raise QwenResponseParseError("Invalid qwen JSON") from exc

        if not isinstance(parsed, dict):
            raise QwenResponseParseError("Qwen JSON must be an object")
        return parsed

    def _parse_qwen_sections(self, data: dict[str, Any], page_index: int) -> dict[str, Any]:
        """Map qwen JSON into page_type + list[SectionCandidate]."""
        page_type = str(data.get("page_type", "bill_summary_page"))
        if page_type not in _ALLOWED_PAGE_TYPES:
            page_type = "bill_summary_page"

        sections: list[SectionCandidate] = []
        raw_sections = data.get("sections", [])
        if isinstance(raw_sections, list):
            for idx, sec in enumerate(raw_sections, start=1):
                if not isinstance(sec, dict):
                    continue
                region_type = str(sec.get("region_type", "")).strip()
                structure_type = str(sec.get("structure_type", "")).strip()
                if region_type not in _ALLOWED_REGION_TYPES or structure_type not in _ALLOWED_STRUCTURE_TYPES:
                    continue
                bbox = self._coerce_bbox(sec)
                if not bbox:
                    continue
                raw_ids = sec.get("source_block_ids", [])
                source_ids = [str(x) for x in raw_ids if isinstance(x, (str, int, float))]
                confidence = max(0.0, min(1.0, float(sec.get("confidence", 0.6))))
                sections.append(SectionCandidate(f"p{page_index}_q{idx}", region_type, structure_type, bbox, source_ids, confidence))

        return {"page_type": page_type, "sections": sections}

    def _merge_pdf_and_ocr_blocks(self, pdf_blocks: list[Block], ocr_blocks: list[Block]) -> list[Block]:
        """Keep PDF blocks first, then append non-duplicate OCR blocks."""
        merged = list(pdf_blocks)
        for ocr in ocr_blocks:
            duplicate = any(
                self._bbox_iou(ocr.bbox, ex.bbox) >= 0.7 and self._text_similarity(ocr.text, ex.text) >= 0.8
                for ex in merged
            )
            if not duplicate:
                merged.append(ocr)
        return merged

    def _resolve_sections(self, page_result: PageParseResult) -> list[dict[str, Any]]:
        """Build final `page_sections` with dedup, bbox fix, normalization, and crop."""
        block_map = {b.block_id: b for b in page_result.blocks}

        adjusted: list[SectionCandidate] = []
        for sec in page_result.section_candidates:
            bbox = sec.bbox
            rebuilt = self._rebuild_bbox_from_source_blocks(sec.source_block_ids, block_map) if sec.source_block_ids else []
            if rebuilt:
                bbox = rebuilt
            bbox = self._clip_bbox(bbox, page_result.width, page_result.height)
            candidate = SectionCandidate(
                section_id=sec.section_id,
                region_type=sec.region_type,
                structure_type=sec.structure_type,
                bbox=bbox,
                source_block_ids=list(sec.source_block_ids),
                confidence=max(0.0, min(1.0, float(sec.confidence))),
            )
            if self._validate_section(candidate):
                adjusted.append(candidate)

        deduped = self._deduplicate_sections(adjusted, page_result.blocks)

        # Protect small regions (page number/footer) by trimming larger conflicts.
        small_regions = [s for s in deduped if s.region_type in {"page_number", "page_footer"}]
        for sec in deduped:
            if sec.region_type in {"page_number", "page_footer"}:
                continue
            for small in small_regions:
                if self._bbox_iou(sec.bbox, small.bbox) > 0.3:
                    sec.bbox[3] = min(sec.bbox[3], small.bbox[1])
                    sec.bbox = self._clip_bbox(sec.bbox, page_result.width, page_result.height)

        ordered = sorted(deduped, key=lambda s: (s.bbox[1], s.bbox[0], s.region_type, s.structure_type))
        final_sections: list[dict[str, Any]] = []
        for rank, sec in enumerate(ordered, start=1):
            if not self._validate_section(sec):
                continue
            norm_bbox = self._normalize_bbox(sec.bbox, page_result.width, page_result.height)
            final_sections.append(
                {
                    "section_id": self._generate_section_id(page_result.page_index, rank, sec),
                    "region_type": sec.region_type,
                    "structure_type": sec.structure_type,
                    "bbox": norm_bbox,
                    "source_block_ids": sec.source_block_ids,
                    "confidence": max(0.0, min(1.0, float(sec.confidence))),
                    "image_base64": self._crop_section_image_base64(page_result.page_image, norm_bbox),
                }
            )

        return final_sections

    def _deduplicate_sections(self, sections: list[SectionCandidate], blocks: list[Block]) -> list[SectionCandidate]:
        """Deduplicate highly overlapping same-region sections with quality scoring."""
        kept: list[SectionCandidate] = []
        block_map = {b.block_id: b for b in blocks}

        for sec in sorted(sections, key=lambda s: (-s.confidence, -len(s.source_block_ids))):
            is_dup = False
            for idx, ex in enumerate(kept):
                if sec.region_type != ex.region_type:
                    continue
                if self._bbox_iou(sec.bbox, ex.bbox) <= 0.3:
                    continue
                score_sec = self._section_quality(sec, block_map)
                score_ex = self._section_quality(ex, block_map)
                if score_sec > score_ex:
                    kept[idx] = sec
                is_dup = True
                break
            if not is_dup:
                kept.append(sec)

        return kept

    def _section_quality(self, section: SectionCandidate, block_map: dict[str, Block]) -> float:
        rebuilt = self._rebuild_bbox_from_source_blocks(section.source_block_ids, block_map)
        fit_score = self._bbox_iou(section.bbox, rebuilt) if rebuilt else 0.0
        return section.confidence * 2.0 + len(section.source_block_ids) * 0.05 + fit_score

    def _rebuild_bbox_from_source_blocks(self, source_block_ids: list[str], block_map: dict[str, Block]) -> list[float]:
        """Rebuild bbox from source blocks; return [] if none can be resolved."""
        boxes = [block_map[sid].bbox for sid in source_block_ids if sid in block_map]
        if not boxes:
            return []
        return [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]

    def _normalize_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        """Normalize absolute bbox to [0,1] with coordinate ordering guarantees."""
        clipped = self._clip_bbox(bbox, page_width, page_height)
        if page_width <= 0 or page_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1 = max(0.0, min(1.0, clipped[0] / page_width))
        y1 = max(0.0, min(1.0, clipped[1] / page_height))
        x2 = max(0.0, min(1.0, clipped[2] / page_width))
        y2 = max(0.0, min(1.0, clipped[3] / page_height))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _clip_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        """Clip bbox into page bounds and enforce x1<=x2, y1<=y2."""
        if len(bbox) != 4 or page_width <= 0 or page_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = [float(v) if isinstance(v, (int, float)) else 0.0 for v in bbox]
        x1 = max(0.0, min(page_width, x1))
        x2 = max(0.0, min(page_width, x2))
        y1 = max(0.0, min(page_height, y1))
        y2 = max(0.0, min(page_height, y2))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _crop_section_image_base64(self, page_image: PageImage, bbox_normalized: list[float]) -> str:
        """Crop normalized region from current page RGB bytes and return base64.

        Falls back to full-page image base64 when bbox is too small or invalid.
        """
        if len(bbox_normalized) != 4:
            return base64.b64encode(page_image.image_bytes).decode("ascii")

        x1 = int(max(0, min(page_image.width - 1, round(bbox_normalized[0] * page_image.width))))
        y1 = int(max(0, min(page_image.height - 1, round(bbox_normalized[1] * page_image.height))))
        x2 = int(max(0, min(page_image.width, round(bbox_normalized[2] * page_image.width))))
        y2 = int(max(0, min(page_image.height, round(bbox_normalized[3] * page_image.height))))

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 2 or (y2 - y1) < 2:
            return base64.b64encode(page_image.image_bytes).decode("ascii")

        row_stride = page_image.width * max(1, page_image.channels)
        channels = max(1, page_image.channels)
        cropped_rows: list[bytes] = []
        for y in range(y1, y2):
            start = y * row_stride + x1 * channels
            end = y * row_stride + x2 * channels
            cropped_rows.append(page_image.rgb_bytes[start:end])

        cropped = b"".join(cropped_rows)
        if not cropped:
            return base64.b64encode(page_image.image_bytes).decode("ascii")
        return base64.b64encode(cropped).decode("ascii")

    def _generate_section_id(self, page_index: int, rank: int, section: SectionCandidate) -> str:
        """Generate stable, predictable section id."""
        _ = section
        return f"p{page_index}_s{rank}"

    def _validate_section(self, section: SectionCandidate) -> bool:
        """Validate candidate with label/shape/confidence constraints."""
        if section.region_type not in _ALLOWED_REGION_TYPES:
            return False
        if section.structure_type not in _ALLOWED_STRUCTURE_TYPES:
            return False
        if len(section.bbox) != 4:
            return False
        x1, y1, x2, y2 = section.bbox
        if x1 > x2 or y1 > y2:
            return False
        section.confidence = max(0.0, min(1.0, float(section.confidence)))
        return True

    def _bbox_iou(self, box_a: list[float], box_b: list[float]) -> float:
        """Compute IoU of two absolute bboxes."""
        if len(box_a) != 4 or len(box_b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute simple text similarity used in OCR/PDF dedup."""
        return SequenceMatcher(None, text_a.strip().lower(), text_b.strip().lower()).ratio()

    def _coerce_bbox(self, obj: dict[str, Any]) -> list[float]:
        """Read bbox from common keys and normalize to [x1,y1,x2,y2] absolute format."""
        bbox = obj.get("bbox") or obj.get("box") or obj.get("rect")
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(v) for v in bbox]

        x1 = obj.get("x1", obj.get("left", obj.get("x0")))
        y1 = obj.get("y1", obj.get("top", obj.get("y0")))
        x2 = obj.get("x2", obj.get("right"))
        y2 = obj.get("y2", obj.get("bottom"))
        if all(v is not None for v in [x1, y1, x2, y2]):
            return [float(x1), float(y1), float(x2), float(y2)]

        return []

    def _infer_page_type(self, blocks: list[Block]) -> Literal["bill_summary_page", "bill_charge_page", "bill_detail_page"]:
        """Heuristic fallback page-type inference for degraded model conditions."""
        texts = [b.text.strip().lower() for b in blocks if b.text.strip()]
        if not texts:
            return "bill_summary_page"

        freq: dict[str, int] = {}
        for text in texts:
            freq[text] = freq.get(text, 0) + 1

        repeated_lines = sum(1 for c in freq.values() if c >= 2)
        if repeated_lines >= 3 or len(texts) >= 30:
            return "bill_detail_page"

        if any(keyword in t for t in texts for keyword in ["charge", "subtotal", "total", "adjustment"]):
            return "bill_charge_page"

        return "bill_summary_page"


def parse_bill_pdf(pdf_path: str, config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Parse bill PDF and return structured output.

    This is the only public API in this module.
    """
    parser = BillLayoutParser(ParserConfig.from_dict(config))
    return parser.parse(pdf_path)
