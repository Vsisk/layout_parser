"""Bill PDF layout parser.

Public API:
    parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Protocol

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


class OCRProcessorProtocol(Protocol):
    def process(self, img_path: str) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class ParserConfig:
    """Runtime configuration."""

    ocr_processor: OCRProcessorProtocol | None = None
    qwen_url: str = ""
    qwen_model_name: str = "qwen3.5-35b"
    api_key: str = ""
    request_timeout_s: float = 15.0
    headers: dict[str, str] | None = None
    include_section_images: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ParserConfig":
        if data is None:
            return cls()
        timeout_val = data.get("timeout", data.get("request_timeout_s", 15.0))
        headers = data.get("headers") if isinstance(data.get("headers"), dict) else None
        ocr_processor = data.get("ocr_processor")
        return cls(
            ocr_processor=ocr_processor,
            qwen_url=str(data.get("qwen_url", "")),
            qwen_model_name=str(data.get("qwen_model_name", "qwen3.5-35b")),
            api_key=str(data.get("api_key", "")),
            request_timeout_s=float(timeout_val),
            headers={str(k): str(v) for k, v in (headers or {}).items()} or None,
            include_section_images=bool(data.get("include_section_images", True)),
        )


@dataclass(slots=True)
class Block:
    """Unified internal block."""

    block_id: str
    text: str
    bbox: list[float]
    page_index: int
    source: str
    block_type: str = "text"
    confidence: float = 1.0


@dataclass(slots=True)
class SectionCandidate:
    section_id: str
    region_type: str
    structure_type: str
    bbox: list[float]
    source_block_ids: list[str]
    confidence: float


@dataclass(slots=True)
class PageImage:
    page_index: int
    width: int
    height: int
    image_format: str
    image_bytes: bytes
    rgb_bytes: bytes
    channels: int


@dataclass(slots=True)
class PageParseResult:
    page_index: int
    page_type: str
    width: float
    height: float
    blocks: list[Block]
    section_candidates: list[SectionCandidate]
    page_image: PageImage


class QwenResponseParseError(ValueError):
    """Raised when qwen output is not valid JSON object."""


class BillLayoutParser:
    """Internal parser engine."""

    def __init__(self, config: ParserConfig) -> None:
        self._config = config

    def parse(self, pdf_path: str) -> dict[str, list[dict[str, Any]]]:
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
                ocr_blocks = self._call_ocr_processor(page_image)
                merged_blocks = self._merge_pdf_and_ocr_blocks(pdf_blocks, ocr_blocks)

                try:
                    qwen_result = self._call_qwen(page_image, pdf_blocks, ocr_blocks)
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

                output_data.append(
                    {
                        "page_index": page_index,
                        "page_type": page_type,
                        "page_sections": self._resolve_sections(page_result),
                    }
                )
        finally:
            doc.close()

        return {"output_data": output_data}

    def _load_pdf(self, pdf_path: Path) -> Any:
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required but not installed")
        try:
            return fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to open PDF: {pdf_path}") from exc

    def _render_page(self, page: Any, page_index: int) -> PageImage:
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
        blocks: list[Block] = []
        for idx, raw in enumerate(page.get_text("blocks")):
            if len(raw) < 5:
                continue
            x1, y1, x2, y2, text = raw[:5]
            txt = str(text).strip()
            if not txt:
                continue
            blocks.append(
                Block(
                    block_id=f"p{page_index}_b{idx}",
                    text=txt,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    page_index=page_index,
                    source="pdf_text",
                    block_type="text",
                    confidence=1.0,
                )
            )
        return blocks

    def _call_ocr_processor(self, page_image: PageImage) -> list[Block]:
        """Call injected OCR/layout processor and normalize to internal Block list."""
        processor = self._config.ocr_processor
        if processor is None:
            return []

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                tmp.write(page_image.image_bytes)
                tmp.flush()
                raw_items = processor.process(tmp.name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ocr_processor.process failed: %s", exc)
            return []

        if not isinstance(raw_items, list):
            return []

        blocks: list[Block] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox_from_coordinate(item.get("coordinate")) or self._coerce_bbox(item)
            if not bbox:
                continue
            label = str(item.get("label", "text")).strip().lower() or "text"
            score = max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0)))
            blocks.append(
                Block(
                    block_id=f"p{page_image.page_index}_ocr{idx}",
                    text="",
                    bbox=bbox,
                    page_index=page_image.page_index,
                    source="ocr_layout",
                    block_type=label,
                    confidence=score,
                )
            )
        return blocks

    def _build_qwen_prompt(self, page_index: int, pdf_blocks: list[Block], ocr_layout_blocks: list[Block]) -> tuple[str, str]:
        system_prompt = "You are a bill layout parser. Return ONLY valid JSON with allowed labels."
        pdf_preview = [
            {"block_id": b.block_id, "text": b.text[:160], "bbox": b.bbox}
            for b in pdf_blocks[:80]
        ]
        ocr_preview = [
            {
                "block_id": b.block_id,
                "bbox": b.bbox,
                "block_type": b.block_type,
                "confidence": b.confidence,
            }
            for b in ocr_layout_blocks[:120]
        ]
        user_prompt = (
            "Task: infer page_type and section candidates.\n"
            "Output schema:\n"
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
            "- region_type must be inferred by you (ocr_layout blocks do not provide region_type).\n"
            "- bbox must be absolute page coordinates [x1,y1,x2,y2].\n"
            "- do not create new labels.\n"
            "- overlapping section candidates are allowed.\n"
            "- ignore irrelevant noise.\n\n"
            f"page_index={page_index}\n"
            f"pdf_text_blocks={json.dumps(pdf_preview, ensure_ascii=False)}\n"
            f"ocr_layout_blocks={json.dumps(ocr_preview, ensure_ascii=False)}"
        )
        return system_prompt, user_prompt

    def _call_qwen(self, page_image: PageImage, pdf_blocks: list[Block], ocr_layout_blocks: list[Block]) -> dict[str, Any]:
        if not self._config.qwen_url:
            return {"page_type": self._infer_page_type(pdf_blocks + ocr_layout_blocks), "sections": []}
        if requests is None:
            raise RuntimeError("requests is required for qwen calls")

        system_prompt, user_prompt = self._build_qwen_prompt(page_image.page_index, pdf_blocks, ocr_layout_blocks)
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
                            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(page_image.image_bytes).decode('ascii')}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json", **(self._config.headers or {})}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        resp = requests.post(
            self._config.qwen_url,
            json=payload,
            headers=headers,
            timeout=self._config.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        content = ""
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data.get("choices"):
            message = data["choices"][0].get("message", {}) if isinstance(data["choices"][0], dict) else {}
            raw = message.get("content", "")
            if isinstance(raw, list):
                content = "\n".join(str(x.get("text", "")) for x in raw if isinstance(x, dict))
            else:
                content = str(raw)
        elif isinstance(data, dict):
            content = str(data.get("output_text", data.get("text", "")))

        return self._parse_qwen_sections(self._extract_json_from_llm_response(content), page_image.page_index)

    def _extract_json_from_llm_response(self, llm_text: str) -> dict[str, Any]:
        text = llm_text.strip()
        if not text:
            raise QwenResponseParseError("Empty qwen response")

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = match.group(1) if match else text
        if not match:
            left, right = candidate.find("{"), candidate.rfind("}")
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
        page_type = str(data.get("page_type", "bill_summary_page"))
        if page_type not in _ALLOWED_PAGE_TYPES:
            page_type = "bill_summary_page"

        sections: list[SectionCandidate] = []
        for idx, sec in enumerate(data.get("sections", []) if isinstance(data.get("sections"), list) else [], start=1):
            if not isinstance(sec, dict):
                continue
            region_type = str(sec.get("region_type", "")).strip()
            structure_type = str(sec.get("structure_type", "")).strip()
            if region_type not in _ALLOWED_REGION_TYPES or structure_type not in _ALLOWED_STRUCTURE_TYPES:
                continue
            bbox = self._coerce_bbox(sec)
            if not bbox:
                continue
            source_ids = [str(x) for x in sec.get("source_block_ids", []) if isinstance(x, (str, int, float))]
            confidence = max(0.0, min(1.0, float(sec.get("confidence", 0.6))))
            sections.append(SectionCandidate(f"p{page_index}_q{idx}", region_type, structure_type, bbox, source_ids, confidence))
        return {"page_type": page_type, "sections": sections}

    def _merge_pdf_and_ocr_blocks(self, pdf_blocks: list[Block], ocr_blocks: list[Block]) -> list[Block]:
        merged = list(pdf_blocks)
        for ocr in ocr_blocks:
            is_dup = any(self._bbox_iou(ocr.bbox, b.bbox) >= 0.7 and self._text_similarity(ocr.text, b.text) >= 0.8 for b in merged)
            if not is_dup:
                merged.append(ocr)
        return merged

    def _resolve_sections(self, page_result: PageParseResult) -> list[dict[str, Any]]:
        block_map = {b.block_id: b for b in page_result.blocks}
        candidates: list[SectionCandidate] = []

        for sec in page_result.section_candidates:
            bbox = sec.bbox
            rebuilt = self._rebuild_bbox_from_source_blocks(sec.source_block_ids, block_map) if sec.source_block_ids else []
            if rebuilt:
                bbox = rebuilt
            clipped = self._clip_bbox(bbox, page_result.width, page_result.height)
            candidate = SectionCandidate(sec.section_id, sec.region_type, sec.structure_type, clipped, list(sec.source_block_ids), max(0.0, min(1.0, sec.confidence)))
            if self._validate_section(candidate):
                candidates.append(candidate)

        deduped = self._deduplicate_sections(candidates, page_result.blocks)
        ordered = sorted(deduped, key=lambda s: (s.bbox[1], s.bbox[0], s.region_type, s.structure_type))

        out: list[dict[str, Any]] = []
        for rank, sec in enumerate(ordered, start=1):
            if not self._validate_section(sec):
                continue
            norm_bbox = self._normalize_bbox(sec.bbox, page_result.width, page_result.height)
            out.append(
                {
                    "section_id": self._generate_section_id(page_result.page_index, rank),
                    "region_type": sec.region_type,
                    "structure_type": sec.structure_type,
                    "bbox": norm_bbox,
                    "source_block_ids": sec.source_block_ids,
                    "confidence": max(0.0, min(1.0, float(sec.confidence))),
                    "image_base64": self._crop_section_image_base64(page_result.page_image, norm_bbox),
                }
            )
        return out

    def _deduplicate_sections(self, sections: list[SectionCandidate], blocks: list[Block]) -> list[SectionCandidate]:
        kept: list[SectionCandidate] = []
        block_map = {b.block_id: b for b in blocks}
        for sec in sorted(sections, key=lambda s: (-s.confidence, -len(s.source_block_ids))):
            duplicated = False
            for i, ex in enumerate(kept):
                if sec.region_type != ex.region_type:
                    continue
                if self._bbox_iou(sec.bbox, ex.bbox) <= 0.3:
                    continue
                if self._section_quality(sec, block_map) > self._section_quality(ex, block_map):
                    kept[i] = sec
                duplicated = True
                break
            if not duplicated:
                kept.append(sec)
        return kept

    def _section_quality(self, section: SectionCandidate, block_map: dict[str, Block]) -> float:
        rebuilt = self._rebuild_bbox_from_source_blocks(section.source_block_ids, block_map)
        fit = self._bbox_iou(section.bbox, rebuilt) if rebuilt else 0.0
        return section.confidence * 2.0 + len(section.source_block_ids) * 0.05 + fit

    def _rebuild_bbox_from_source_blocks(self, source_block_ids: list[str], block_map: dict[str, Block]) -> list[float]:
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
        clipped = self._clip_bbox(bbox, page_width, page_height)
        if page_width <= 0 or page_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = clipped
        n = [x1 / page_width, y1 / page_height, x2 / page_width, y2 / page_height]
        return [max(0.0, min(1.0, n[0])), max(0.0, min(1.0, n[1])), max(0.0, min(1.0, n[2])), max(0.0, min(1.0, n[3]))]

    def _clip_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        if len(bbox) != 4 or page_width <= 0 or page_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = [float(v) if isinstance(v, (int, float)) else 0.0 for v in bbox]
        x1 = max(0.0, min(page_width, x1))
        x2 = max(0.0, min(page_width, x2))
        y1 = max(0.0, min(page_height, y1))
        y2 = max(0.0, min(page_height, y2))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _crop_section_image_base64(self, page_image: PageImage, bbox_normalized: list[float]) -> str:
        if len(bbox_normalized) != 4:
            return base64.b64encode(page_image.image_bytes).decode("ascii")
        x1 = int(max(0, min(page_image.width - 1, round(bbox_normalized[0] * page_image.width))))
        y1 = int(max(0, min(page_image.height - 1, round(bbox_normalized[1] * page_image.height))))
        x2 = int(max(0, min(page_image.width, round(bbox_normalized[2] * page_image.width))))
        y2 = int(max(0, min(page_image.height, round(bbox_normalized[3] * page_image.height))))
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 2 or (y2 - y1) < 2:
            return base64.b64encode(page_image.image_bytes).decode("ascii")

        channels = max(1, page_image.channels)
        row_stride = page_image.width * channels
        chunks: list[bytes] = []
        for y in range(y1, y2):
            start = y * row_stride + x1 * channels
            end = y * row_stride + x2 * channels
            chunks.append(page_image.rgb_bytes[start:end])
        cropped = b"".join(chunks)
        if not cropped:
            return base64.b64encode(page_image.image_bytes).decode("ascii")
        return base64.b64encode(cropped).decode("ascii")

    def _generate_section_id(self, page_index: int, rank: int) -> str:
        return f"p{page_index}_s{rank}"

    def _validate_section(self, section: SectionCandidate) -> bool:
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

    def _text_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

    def _coerce_bbox(self, obj: dict[str, Any]) -> list[float]:
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

    def _coerce_bbox_from_coordinate(self, coordinate: Any) -> list[float]:
        if isinstance(coordinate, list):
            if len(coordinate) == 4 and all(isinstance(v, (int, float)) for v in coordinate):
                return [float(v) for v in coordinate]
            points: list[tuple[float, float]] = []
            for p in coordinate:
                if isinstance(p, (list, tuple)) and len(p) >= 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                    points.append((float(p[0]), float(p[1])))
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return [min(xs), min(ys), max(xs), max(ys)]
        return []

    def _infer_page_type(self, blocks: list[Block]) -> Literal["bill_summary_page", "bill_charge_page", "bill_detail_page"]:
        texts = [b.text.strip().lower() for b in blocks if b.text.strip()]
        if not texts:
            return "bill_summary_page"

        freq: dict[str, int] = {}
        for t in texts:
            freq[t] = freq.get(t, 0) + 1
        repeated_lines = sum(1 for c in freq.values() if c >= 2)
        if repeated_lines >= 3 or len(texts) >= 30:
            return "bill_detail_page"
        if any(k in t for t in texts for k in ["charge", "subtotal", "total", "adjustment"]):
            return "bill_charge_page"
        return "bill_summary_page"


def parse_bill_pdf(pdf_path: str, config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Parse bill PDF and return final structured dictionary."""
    parser = BillLayoutParser(ParserConfig.from_dict(config))
    return parser.parse(pdf_path)
