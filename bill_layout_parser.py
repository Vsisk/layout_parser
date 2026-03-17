"""Bill PDF layout parsing module.

This module exposes a single public API:

    parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict
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
    doclayout_url: str = ""
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
        headers = data.get("headers")
        return cls(
            doclayout_url=str(data.get("doclayout_url", "")),
            qwen_url=str(data.get("qwen_url", "")),
            qwen_model_name=str(data.get("qwen_model_name", "qwen3.5-35b")),
            api_key=str(data.get("api_key", "")),
            request_timeout_s=float(timeout_val),
            headers=headers if isinstance(headers, dict) else None,
            include_section_images=bool(data.get("include_section_images", True)),
        )


@dataclass(slots=True)
class Block:
    block_id: str
    text: str
    bbox: list[float]
    page_index: int
    source: str


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
    pass


class BillLayoutParser:
    def __init__(self, config: ParserConfig) -> None:
        self._config = config

    def parse(self, pdf_path: str) -> dict[str, list[dict[str, Any]]]:
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Input must be a .pdf file: {pdf_path}")

        pdf_doc = self._load_pdf(path)
        output: list[dict[str, Any]] = []
        try:
            for page_index in range(pdf_doc.page_count):
                page = pdf_doc.load_page(page_index)
                page_image = self._render_page(page, page_index)
                pdf_blocks = self._extract_pdf_blocks(page, page_index)
                doclayout_result = self._call_doclayout(page_image)
                merged_blocks = self._merge_pdf_and_ocr_blocks(pdf_blocks, doclayout_result.get("ocr_blocks", []))

                try:
                    qwen_result = self._call_qwen(page_image, merged_blocks, doclayout_result)
                    page_type = str(qwen_result.get("page_type", "bill_summary_page"))
                    section_candidates = qwen_result.get("sections", [])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Qwen call/parse failed, fallback enabled: %s", exc)
                    page_type = self._infer_page_type(merged_blocks)
                    section_candidates = []

                parsed = PageParseResult(
                    page_index=page_index,
                    page_type=page_type if page_type in _ALLOWED_PAGE_TYPES else self._infer_page_type(merged_blocks),
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=merged_blocks,
                    section_candidates=section_candidates,
                    page_image=page_image,
                )
                page_sections = self._resolve_sections(parsed)
                output.append(
                    {
                        "page_index": parsed.page_index,
                        "page_type": parsed.page_type,
                        "page_sections": page_sections,
                    }
                )
        finally:
            pdf_doc.close()
        return {"output_data": output}

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
            rgb_bytes=bytes(pix.samples),
            channels=int(getattr(pix, "n", 3) or 3),
        )

    def _extract_pdf_blocks(self, page: Any, page_index: int) -> list[Block]:
        out: list[Block] = []
        for idx, raw in enumerate(page.get_text("blocks")):
            if len(raw) < 5:
                continue
            x1, y1, x2, y2, text = raw[:5]
            t = str(text).strip()
            if not t:
                continue
            out.append(Block(f"p{page_index}_b{idx}", t, [float(x1), float(y1), float(x2), float(y2)], page_index, "pdf_text"))
        return out

    def _image_to_base64(self, page_image: PageImage) -> str:
        return base64.b64encode(page_image.image_bytes).decode("ascii")

    def _call_doclayout(self, page_image: PageImage) -> dict[str, list[Block] | list[dict[str, Any]]]:
        if not self._config.doclayout_url:
            return {"layout_blocks": [], "ocr_blocks": []}
        if requests is None:
            LOGGER.warning("requests not installed")
            return {"layout_blocks": [], "ocr_blocks": []}
        headers = {"Content-Type": "application/json", **(self._config.headers or {})}
        payload = {"image_base64": self._image_to_base64(page_image), "page_index": page_image.page_index}
        try:
            resp = requests.post(self._config.doclayout_url, json=payload, headers=headers, timeout=self._config.request_timeout_s)
            resp.raise_for_status()
            return self._parse_doclayout_response(resp.json(), page_image.page_index)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DocLayout call failed: %s", exc)
            return {"layout_blocks": [], "ocr_blocks": []}

    def _parse_doclayout_response(self, response_json: dict[str, Any] | list[Any] | str, page_index: int) -> dict[str, Any]:
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

        layout_raw = data.get("layout_blocks") or data.get("layout") or data.get("regions") or []
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
        for i, item in enumerate(ocr_raw if isinstance(ocr_raw, list) else []):
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox(item)
            txt = str(item.get("text", item.get("content", ""))).strip()
            if not bbox or not txt:
                continue
            ocr_blocks.append(Block(f"p{page_index}_ocr{i}", txt, bbox, page_index, "ocr"))
        return {"layout_blocks": layout_blocks, "ocr_blocks": ocr_blocks}

    def _build_qwen_prompt(self, page_index: int, blocks: list[Block], doclayout_result: dict[str, Any]) -> tuple[str, str]:
        system_prompt = "You are a bill parser. Return ONLY strict JSON with allowed labels."
        user_prompt = (
            "Output schema: {page_type, sections:[{region_type,structure_type,bbox,source_block_ids,confidence}]}. "
            f"Allowed page_type={sorted(_ALLOWED_PAGE_TYPES)}; structure_type={sorted(_ALLOWED_STRUCTURE_TYPES)}; "
            f"region_type={sorted(_ALLOWED_REGION_TYPES)}. bbox uses page coords [x1,y1,x2,y2]. "
            "Do not invent labels. Overlap is allowed. Ignore noise. "
            f"page_index={page_index}; blocks={json.dumps([{'block_id':b.block_id,'text':b.text[:120],'bbox':b.bbox} for b in blocks[:80]], ensure_ascii=False)}; "
            f"layout={json.dumps(doclayout_result.get('layout_blocks', [])[:50], ensure_ascii=False)}"
        )
        return system_prompt, user_prompt

    def _call_qwen(self, page_image: PageImage, blocks: list[Block], doclayout_result: dict[str, Any]) -> dict[str, Any]:
        if not self._config.qwen_url:
            return {"page_type": self._infer_page_type(blocks), "sections": []}
        if requests is None:
            raise RuntimeError("requests required")
        sp, up = self._build_qwen_prompt(page_image.page_index, blocks, doclayout_result)
        payload = {
            "model": self._config.qwen_model_name,
            "messages": [
                {"role": "system", "content": sp},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": up},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(page_image)}"}},
                    ],
                },
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json", **(self._config.headers or {})}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        resp = requests.post(self._config.qwen_url, json=payload, headers=headers, timeout=self._config.request_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = ""
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data.get("choices"):
            msg = data["choices"][0].get("message", {})
            raw = msg.get("content", "")
            if isinstance(raw, list):
                content = "\n".join(str(p.get("text", "")) for p in raw if isinstance(p, dict))
            else:
                content = str(raw)
        else:
            content = str(data.get("output_text", data.get("text", ""))) if isinstance(data, dict) else ""
        return self._parse_qwen_sections(self._extract_json_from_llm_response(content), page_image.page_index)

    def _extract_json_from_llm_response(self, llm_text: str) -> dict[str, Any]:
        text = llm_text.strip()
        if not text:
            raise QwenResponseParseError("empty")
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1] if "{" in text and "}" in text else text
        try:
            parsed = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            raise QwenResponseParseError("invalid json") from exc
        if not isinstance(parsed, dict):
            raise QwenResponseParseError("json object required")
        return parsed

    def _parse_qwen_sections(self, data: dict[str, Any], page_index: int) -> dict[str, Any]:
        page_type = str(data.get("page_type", "bill_summary_page"))
        if page_type not in _ALLOWED_PAGE_TYPES:
            page_type = "bill_summary_page"
        sections: list[SectionCandidate] = []
        for idx, sec in enumerate(data.get("sections", []) if isinstance(data.get("sections"), list) else [], start=1):
            if not isinstance(sec, dict):
                continue
            rt = str(sec.get("region_type", ""))
            st = str(sec.get("structure_type", ""))
            if rt not in _ALLOWED_REGION_TYPES or st not in _ALLOWED_STRUCTURE_TYPES:
                continue
            bbox = self._coerce_bbox(sec)
            if not bbox:
                continue
            ids = [str(x) for x in sec.get("source_block_ids", []) if isinstance(x, (str, int, float))]
            conf = max(0.0, min(1.0, float(sec.get("confidence", 0.6))))
            sections.append(SectionCandidate(f"p{page_index}_q{idx}", rt, st, bbox, ids, conf))
        return {"page_type": page_type, "sections": sections}

    def _merge_pdf_and_ocr_blocks(self, pdf_blocks: list[Block], ocr_blocks: list[Block]) -> list[Block]:
        merged = list(pdf_blocks)
        for ob in ocr_blocks:
            dup = any(self._bbox_iou(ob.bbox, pb.bbox) >= 0.7 and self._text_similarity(ob.text, pb.text) >= 0.8 for pb in merged)
            if not dup:
                merged.append(ob)
        return merged

    def _resolve_sections(self, page_result: PageParseResult) -> list[dict[str, Any]]:
        blocks_map = {b.block_id: b for b in page_result.blocks}
        candidates: list[SectionCandidate] = []

        for cand in page_result.section_candidates:
            fixed_bbox = cand.bbox
            if cand.source_block_ids:
                rebuilt = self._rebuild_bbox_from_source_blocks(cand.source_block_ids, blocks_map)
                if rebuilt:
                    fixed_bbox = rebuilt
            fixed_bbox = self._clip_bbox(fixed_bbox, page_result.width, page_result.height)
            adjusted = SectionCandidate(cand.section_id, cand.region_type, cand.structure_type, fixed_bbox, cand.source_block_ids, cand.confidence)
            if self._validate_section(adjusted):
                candidates.append(adjusted)

        candidates = self._deduplicate_sections(candidates, page_result.blocks)

        # special overlap handling: preserve small page number/footer and trim larger sections
        small_regions = [s for s in candidates if s.region_type in {"page_number", "page_footer"}]
        for s in candidates:
            if s.region_type in {"page_number", "page_footer"}:
                continue
            for sm in small_regions:
                if self._bbox_iou(s.bbox, sm.bbox) > 0.3:
                    s.bbox[3] = min(s.bbox[3], sm.bbox[1])  # trim bottom boundary away from footer/number
                    s.bbox = self._clip_bbox(s.bbox, page_result.width, page_result.height)

        candidates = [c for c in candidates if self._validate_section(c)]

        output: list[dict[str, Any]] = []
        ordered = sorted(candidates, key=lambda c: (c.bbox[1], c.bbox[0], c.region_type, c.structure_type))
        for idx, c in enumerate(ordered, start=1):
            norm_bbox = self._normalize_bbox(c.bbox, page_result.width, page_result.height)
            sid = self._generate_section_id(page_result.page_index, idx, c)
            output.append(
                {
                    "section_id": sid,
                    "region_type": c.region_type,
                    "structure_type": c.structure_type,
                    "bbox": norm_bbox,
                    "source_block_ids": list(c.source_block_ids),
                    "confidence": max(0.0, min(1.0, float(c.confidence))),
                    "image_base64": self._crop_section_image_base64(page_result.page_image, norm_bbox),
                }
            )
        return output

    def _deduplicate_sections(self, sections: list[SectionCandidate], blocks: list[Block]) -> list[SectionCandidate]:
        kept: list[SectionCandidate] = []
        blocks_map = {b.block_id: b for b in blocks}
        for sec in sorted(sections, key=lambda s: (-s.confidence, -len(s.source_block_ids))):
            duplicate = False
            for i, ex in enumerate(kept):
                if sec.region_type != ex.region_type:
                    continue
                if self._bbox_iou(sec.bbox, ex.bbox) <= 0.3:
                    continue
                score_sec = self._section_quality(sec, blocks_map)
                score_ex = self._section_quality(ex, blocks_map)
                if score_sec > score_ex:
                    kept[i] = sec
                duplicate = True
                break
            if not duplicate:
                kept.append(sec)
        return kept

    def _section_quality(self, section: SectionCandidate, blocks_map: dict[str, Block]) -> float:
        rebuilt = self._rebuild_bbox_from_source_blocks(section.source_block_ids, blocks_map)
        iou = self._bbox_iou(section.bbox, rebuilt) if rebuilt else 0.0
        return section.confidence * 2.0 + len(section.source_block_ids) * 0.05 + iou

    def _rebuild_bbox_from_source_blocks(self, source_block_ids: list[str], blocks_map: dict[str, Block]) -> list[float]:
        boxes = [blocks_map[bid].bbox for bid in source_block_ids if bid in blocks_map]
        if not boxes:
            return []
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

    def _normalize_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        clipped = self._clip_bbox(bbox, page_width, page_height)
        x1 = max(0.0, min(1.0, clipped[0] / page_width if page_width > 0 else 0.0))
        y1 = max(0.0, min(1.0, clipped[1] / page_height if page_height > 0 else 0.0))
        x2 = max(0.0, min(1.0, clipped[2] / page_width if page_width > 0 else 0.0))
        y2 = max(0.0, min(1.0, clipped[3] / page_height if page_height > 0 else 0.0))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _clip_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        if len(bbox) != 4:
            return [0.0, 0.0, min(1.0, page_width), min(1.0, page_height)]
        x1, y1, x2, y2 = [float(v) if isinstance(v, (int, float)) else 0.0 for v in bbox]
        if page_width <= 0 or page_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1 = max(0.0, min(page_width, x1))
        x2 = max(0.0, min(page_width, x2))
        y1 = max(0.0, min(page_height, y1))
        y2 = max(0.0, min(page_height, y2))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _crop_section_image_base64(self, page_image: PageImage, bbox_normalized: list[float]) -> str:
        if not self._config.include_section_images:
            return ""
        if len(bbox_normalized) != 4:
            return base64.b64encode(page_image.image_bytes).decode("ascii")

        x1 = int(max(0, min(page_image.width - 1, round(bbox_normalized[0] * page_image.width))))
        y1 = int(max(0, min(page_image.height - 1, round(bbox_normalized[1] * page_image.height))))
        x2 = int(max(0, min(page_image.width, round(bbox_normalized[2] * page_image.width))))
        y2 = int(max(0, min(page_image.height, round(bbox_normalized[3] * page_image.height))))
        if x2 <= x1 or y2 <= y1:
            return base64.b64encode(page_image.image_bytes).decode("ascii")
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return base64.b64encode(page_image.image_bytes).decode("ascii")

        row_stride = page_image.width * page_image.channels
        cropped_rows = []
        for y in range(y1, y2):
            start = y * row_stride + x1 * page_image.channels
            end = y * row_stride + x2 * page_image.channels
            cropped_rows.append(page_image.rgb_bytes[start:end])
        cropped_rgb = b"".join(cropped_rows)
        if not cropped_rgb:
            return base64.b64encode(page_image.image_bytes).decode("ascii")
        # Use raw rgb bytes for stable, lightweight encoding.
        return base64.b64encode(cropped_rgb).decode("ascii")

    def _generate_section_id(self, page_index: int, rank: int, section: SectionCandidate) -> str:
        _ = section
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
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _text_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

    def _coerce_bbox(self, item: dict[str, Any]) -> list[float]:
        bbox = item.get("bbox") or item.get("box") or item.get("rect")
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(v) for v in bbox]
        x1 = item.get("x1", item.get("left", item.get("x0")))
        y1 = item.get("y1", item.get("top", item.get("y0")))
        x2 = item.get("x2", item.get("right"))
        y2 = item.get("y2", item.get("bottom"))
        if all(v is not None for v in [x1, y1, x2, y2]):
            return [float(x1), float(y1), float(x2), float(y2)]
        return []

    def _infer_page_type(self, blocks: list[Block]) -> Literal["bill_summary_page", "bill_charge_page", "bill_detail_page"]:
        texts = [b.text.lower() for b in blocks if b.text.strip()]
        if not texts:
            return "bill_summary_page"
        counts: dict[str, int] = {}
        for t in texts:
            counts[t] = counts.get(t, 0) + 1
        if sum(1 for v in counts.values() if v >= 2) >= 3 or len(texts) >= 30:
            return "bill_detail_page"
        if any(k in t for t in texts for k in ["charge", "subtotal", "total", "adjustment"]):
            return "bill_charge_page"
        return "bill_summary_page"


def parse_bill_pdf(pdf_path: str, config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    parser = BillLayoutParser(ParserConfig.from_dict(config))
    return parser.parse(pdf_path)
