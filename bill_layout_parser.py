"""Bill PDF layout parsing module.

This module exposes a single public API:

    parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict

All parsing logic is intentionally centralized here for easy integration into a
larger internal system.
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
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore[assignment]

try:
    import fitz  # pymupdf
except Exception:  # noqa: BLE001
    fitz = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

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
    """Configuration for bill layout parsing."""

    doclayout_url: str = ""
    qwen_url: str = ""
    qwen_model_name: str = "qwen3.5-35b"
    api_key: str = ""
    request_timeout_s: float = 15.0
    headers: dict[str, str] | None = None
    include_section_images: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ParserConfig":
        if data is None:
            return cls()
        timeout_val = data.get("timeout", data.get("request_timeout_s", 15.0))
        headers = data.get("headers")
        safe_headers = headers if isinstance(headers, dict) else None
        return cls(
            doclayout_url=str(data.get("doclayout_url", "")),
            qwen_url=str(data.get("qwen_url", "")),
            qwen_model_name=str(data.get("qwen_model_name", "qwen3.5-35b")),
            api_key=str(data.get("api_key", "")),
            request_timeout_s=float(timeout_val),
            headers=safe_headers,
            include_section_images=bool(data.get("include_section_images", False)),
        )


@dataclass(slots=True)
class Block:
    """Unified internal block structure."""

    block_id: str
    text: str
    bbox: list[float]
    page_index: int
    source: str


@dataclass(slots=True)
class SectionCandidate:
    """Intermediate section candidate before final output serialization."""

    section_id: str
    region_type: str
    structure_type: str
    bbox: list[float]
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


class QwenResponseParseError(ValueError):
    """Raised when Qwen response cannot be parsed into expected JSON."""


class BillLayoutParser:
    """Internal parser implementation for bill PDF layout extraction."""

    def __init__(self, config: ParserConfig) -> None:
        self._config = config

    def parse(self, pdf_path: str) -> dict[str, list[dict[str, Any]]]:
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
                pdf_blocks = self._extract_pdf_blocks(page, page_index)

                doclayout_result = self._call_doclayout(page_image)
                merged_blocks = self._merge_pdf_and_ocr_blocks(
                    pdf_blocks=pdf_blocks,
                    ocr_blocks=doclayout_result.get("ocr_blocks", []),
                )

                try:
                    qwen_result = self._call_qwen(page_image, merged_blocks, doclayout_result)
                    page_type = str(qwen_result.get("page_type", "bill_summary_page"))
                    section_candidates = qwen_result.get("sections", [])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Qwen call/parse failed, fallback enabled: %s", exc)
                    page_type = self._infer_page_type(merged_blocks)
                    section_candidates = []

                page_result = PageParseResult(
                    page_index=page_index,
                    page_type=page_type,
                    width=page_width,
                    height=page_height,
                    blocks=merged_blocks,
                    section_candidates=section_candidates,
                    page_image=page_image,
                )
                sections = self._resolve_sections(page_result)
                final_page_type = (
                    page_result.page_type if page_result.page_type in _ALLOWED_PAGE_TYPES else self._infer_page_type(merged_blocks)
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
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required but not installed")
        try:
            return fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to open PDF: {pdf_path}") from exc

    def _render_page(self, page: Any, page_index: int) -> PageImage:
        if fitz is not None:
            pix = page.get_pixmap(colorspace=fitz.csRGB, alpha=False)
        else:
            pix = page.get_pixmap(alpha=False)
        return PageImage(
            page_index=page_index,
            width=int(pix.width),
            height=int(pix.height),
            image_format="png",
            image_bytes=pix.tobytes("png"),
        )

    def _extract_pdf_blocks(self, page: Any, page_index: int) -> list[Block]:
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

    def _image_to_base64(self, page_image: PageImage) -> str:
        return base64.b64encode(page_image.image_bytes).decode("ascii")

    def _call_doclayout(self, page_image: PageImage) -> dict[str, list[Block] | list[dict[str, Any]]]:
        if not self._config.doclayout_url:
            return {"layout_blocks": [], "ocr_blocks": []}

        headers = {"Content-Type": "application/json"}
        if self._config.headers:
            headers.update({str(k): str(v) for k, v in self._config.headers.items()})

        payload = {
            "image_base64": self._image_to_base64(page_image),
            "page_index": page_image.page_index,
            "image_format": page_image.image_format,
        }

        if requests is None:
            LOGGER.warning("requests is not installed; skipping DocLayout call")
            return {"layout_blocks": [], "ocr_blocks": []}

        try:
            response = requests.post(
                self._config.doclayout_url,
                json=payload,
                headers=headers,
                timeout=self._config.request_timeout_s,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DocLayout call failed: %s", exc)
            return {"layout_blocks": [], "ocr_blocks": []}

        return self._parse_doclayout_response(data, page_image.page_index)

    def _parse_doclayout_response(
        self,
        response_json: dict[str, Any] | list[Any] | str,
        page_index: int,
    ) -> dict[str, list[Block] | list[dict[str, Any]]]:
        data: Any = response_json
        if isinstance(response_json, str):
            try:
                data = json.loads(response_json)
            except Exception:  # noqa: BLE001
                return {"layout_blocks": [], "ocr_blocks": []}

        if isinstance(data, list):
            data = {"layout_blocks": data}
        if not isinstance(data, dict):
            return {"layout_blocks": [], "ocr_blocks": []}

        layout_candidates = (
            data.get("layout_blocks")
            or data.get("layout")
            or data.get("regions")
            or data.get("detections")
            or []
        )
        ocr_candidates = (
            data.get("ocr_blocks")
            or data.get("ocr")
            or data.get("text_blocks")
            or data.get("blocks")
            or []
        )

        normalized_layout: list[dict[str, Any]] = []
        for item in layout_candidates if isinstance(layout_candidates, list) else []:
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox(item)
            if not bbox:
                continue
            normalized_layout.append(
                {
                    "region_type": str(item.get("region_type") or item.get("label") or "bill_charge_page.charge_items"),
                    "structure_type": str(item.get("structure_type") or item.get("type") or "table"),
                    "bbox": bbox,
                    "confidence": float(item.get("confidence", item.get("score", 0.0)) or 0.0),
                }
            )

        normalized_ocr_blocks: list[Block] = []
        for idx, item in enumerate(ocr_candidates if isinstance(ocr_candidates, list) else []):
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox(item)
            text = str(item.get("text", item.get("content", ""))).strip()
            if not bbox or not text:
                continue
            normalized_ocr_blocks.append(
                Block(
                    block_id=f"p{page_index}_ocr{idx}",
                    text=text,
                    bbox=bbox,
                    page_index=page_index,
                    source="ocr",
                )
            )

        return {"layout_blocks": normalized_layout, "ocr_blocks": normalized_ocr_blocks}

    def _build_qwen_prompt(
        self,
        page_index: int,
        blocks: list[Block],
        doclayout_result: dict[str, Any],
    ) -> tuple[str, str]:
        block_preview = [
            {
                "block_id": b.block_id,
                "text": b.text[:200],
                "bbox": b.bbox,
                "source": b.source,
            }
            for b in blocks[:80]
        ]
        layout_preview = doclayout_result.get("layout_blocks", [])

        system_prompt = (
            "You are a bill PDF layout parser. "
            "Return ONLY valid JSON matching the required schema. "
            "Do not create new labels outside allowed sets."
        )
        user_prompt = (
            "Task: classify page_type and produce section candidates for this bill page.\n"
            "Output JSON schema:\n"
            "{\n"
            '  "page_type": "bill_summary_page|bill_charge_page|bill_detail_page",\n'
            '  "sections": [\n'
            "    {\n"
            '      "region_type": "<allowed region type>",\n'
            '      "structure_type": "table|text|kv|image",\n'
            '      "bbox": [x0, y0, x1, y1],\n'
            '      "source_block_ids": ["..."],\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            f"- page_type must be one of: {sorted(_ALLOWED_PAGE_TYPES)}\n"
            f"- region_type must be one of: {sorted(_ALLOWED_REGION_TYPES)}\n"
            f"- structure_type must be one of: {sorted(_ALLOWED_STRUCTURE_TYPES)}\n"
            "- bbox should use page absolute coordinates where possible.\n"
            "- structure_type and region_type are both mandatory for each section.\n"
            "- Do not invent any new tag not listed above.\n"
            "- Overlapping section candidates are allowed; downstream post-processing will resolve.\n"
            "- Ignore irrelevant noise content.\n\n"
            f"Page index: {page_index}\n"
            f"PDF/OCR blocks (preview): {json.dumps(block_preview, ensure_ascii=False)}\n"
            f"Doclayout regions (preview): {json.dumps(layout_preview, ensure_ascii=False)}\n"
        )
        return system_prompt, user_prompt

    def _call_qwen(self, page_image: PageImage, blocks: list[Block], doclayout_result: dict[str, Any]) -> dict[str, Any]:
        if not self._config.qwen_url:
            return {"page_type": self._infer_page_type(blocks), "sections": []}

        if requests is None:
            raise RuntimeError("requests is required for qwen URL call")

        system_prompt, user_prompt = self._build_qwen_prompt(page_image.page_index, blocks, doclayout_result)
        payload: dict[str, Any] = {
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

        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        if self._config.headers:
            headers.update({str(k): str(v) for k, v in self._config.headers.items()})

        response = requests.post(
            self._config.qwen_url,
            json=payload,
            headers=headers,
            timeout=self._config.request_timeout_s,
        )
        response.raise_for_status()
        resp_data = response.json()

        content = ""
        if isinstance(resp_data, dict):
            choices = resp_data.get("choices", [])
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                raw_content = message.get("content", "")
                if isinstance(raw_content, list):
                    text_parts = []
                    for part in raw_content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(str(part["text"]))
                    content = "\n".join(text_parts)
                else:
                    content = str(raw_content)
            else:
                content = str(resp_data.get("output_text", resp_data.get("text", "")))

        parsed_json = self._extract_json_from_llm_response(content)
        return self._parse_qwen_sections(parsed_json, page_image.page_index)

    def _extract_json_from_llm_response(self, llm_text: str) -> dict[str, Any]:
        text = llm_text.strip()
        if not text:
            raise QwenResponseParseError("Empty response from qwen")

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else text

        if not fenced:
            first = candidate.find("{")
            last = candidate.rfind("}")
            if first != -1 and last != -1 and first < last:
                candidate = candidate[first : last + 1]

        try:
            parsed = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            raise QwenResponseParseError("Invalid JSON from qwen") from exc

        if not isinstance(parsed, dict):
            raise QwenResponseParseError("Qwen JSON must be an object")
        return parsed

    def _parse_qwen_sections(self, data: dict[str, Any], page_index: int) -> dict[str, Any]:
        raw_page_type = str(data.get("page_type", "")).strip()
        page_type = raw_page_type if raw_page_type in _ALLOWED_PAGE_TYPES else "bill_summary_page"

        raw_sections = data.get("sections", [])
        sections: list[SectionCandidate] = []
        if isinstance(raw_sections, list):
            for idx, sec in enumerate(raw_sections, start=1):
                if not isinstance(sec, dict):
                    continue
                region_type = str(sec.get("region_type", "")).strip()
                structure_type = str(sec.get("structure_type", "")).strip()
                if region_type not in _ALLOWED_REGION_TYPES:
                    continue
                if structure_type not in _ALLOWED_STRUCTURE_TYPES:
                    continue

                bbox = self._coerce_bbox(sec)
                if not bbox:
                    continue

                raw_ids = sec.get("source_block_ids", [])
                source_ids = [str(v) for v in raw_ids if isinstance(v, (str, int, float))]
                confidence = float(sec.get("confidence", 0.6))

                sections.append(
                    SectionCandidate(
                        section_id=f"s{idx}",
                        region_type=region_type,
                        structure_type=structure_type,
                        bbox=bbox,
                        source_block_ids=source_ids,
                        confidence=confidence,
                    )
                )

        for i, sec in enumerate(sections, start=1):
            sec.section_id = f"p{page_index}_q{i}"

        return {"page_type": page_type, "sections": sections}

    def _merge_pdf_and_ocr_blocks(self, pdf_blocks: list[Block], ocr_blocks: list[Block]) -> list[Block]:
        merged: list[Block] = list(pdf_blocks)
        for ocr in ocr_blocks:
            duplicated = False
            for existing in merged:
                if self._bbox_iou(existing.bbox, ocr.bbox) >= 0.7 and self._text_similarity(existing.text, ocr.text) >= 0.8:
                    duplicated = True
                    break
            if not duplicated:
                merged.append(ocr)
        return merged

    def _bbox_iou(self, box_a: list[float], box_b: list[float]) -> float:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def _text_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

    def _coerce_bbox(self, item: dict[str, Any]) -> list[float]:
        bbox = item.get("bbox") or item.get("box") or item.get("rect")
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(v) for v in bbox]

        x0 = item.get("x0", item.get("left"))
        y0 = item.get("y0", item.get("top"))
        x1 = item.get("x1", item.get("right"))
        y1 = item.get("y1", item.get("bottom"))
        if all(v is not None for v in [x0, y0, x1, y1]):
            return [float(x0), float(y0), float(x1), float(y1)]
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

        charge_keywords = ["charge", "subtotal", "total", "adjustment"]
        if any(any(k in t for k in charge_keywords) for t in texts):
            return "bill_charge_page"

        return "bill_summary_page"

    def _resolve_sections(self, page_result: PageParseResult) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for idx, candidate in enumerate(page_result.section_candidates, start=1):
            structure_type = candidate.structure_type if candidate.structure_type in _ALLOWED_STRUCTURE_TYPES else "text"
            if candidate.region_type not in _ALLOWED_REGION_TYPES:
                continue
            normalized_bbox = self._normalize_bbox(candidate.bbox, page_result.width, page_result.height)
            image_b64 = self._crop_section_image_base64(page_result.page_image, normalized_bbox)
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
        _ = bbox_normalized
        if not self._config.include_section_images:
            return ""
        return base64.b64encode(page_image.image_bytes).decode("ascii")

    def _normalize_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
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
        if page_type not in _ALLOWED_PAGE_TYPES:
            raise ValueError(f"Unsupported page_type: {page_type}")
        return {
            "page_index": page_index,
            "page_type": page_type,
            "page_sections": sections,
        }


def parse_bill_pdf(pdf_path: str, config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Parse bill PDF layout and return structured output."""
    parser = BillLayoutParser(ParserConfig.from_dict(config))
    return parser.parse(pdf_path)
