"""Bill PDF layout parser.

Public API:
    parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict
"""

from __future__ import annotations

import base64
import importlib
import json
import logging
import re
import tempfile
try:
    import tomllib
except Exception:  # noqa: BLE001
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Literal, Protocol

try:
    import fitz  # pymupdf
except Exception:  # noqa: BLE001
    fitz = None  # type: ignore[assignment]

try:  # AgentRunner is expected from runtime environment.
    from agent_runner import AgentRunner  # type: ignore
except Exception:  # noqa: BLE001
    AgentRunner = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

_ALLOWED_PAGE_TYPES: set[str] = {"bill_summary_page", "bill_charge_page", "bill_detail_page"}
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
class Block:
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
    page_source_type: str
    width: float
    height: float
    blocks: list[Block]
    section_candidates: list[SectionCandidate]
    page_image: PageImage


class LLMResponseParseError(ValueError):
    """Raised when LLM response cannot be converted to expected JSON object."""


def image_to_base64(img_path: str) -> str:
    return base64.b64encode(Path(img_path).read_bytes()).decode("ascii")


class BillLayoutParser:
    """Internal parser engine."""

    def __init__(self, config_overrides: dict[str, Any] | None = None) -> None:
        self._runtime_config = self._load_env_config()
        if isinstance(config_overrides, dict):
            self._runtime_config.update(config_overrides)

    def _load_env_config(self) -> dict[str, Any]:
        """Load config from env.config.toml.

        Supports either top-level keys or `[bill_layout_parser]` table.
        """
        cfg_path = Path("env.config.toml")
        if not cfg_path.exists():
            return {}
        try:
            with cfg_path.open("rb") as f:
                parsed = tomllib.load(f)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read env.config.toml: %s", exc)
            return {}

        section = parsed.get("bill_layout_parser")
        if isinstance(section, dict):
            return dict(section)
        return parsed if isinstance(parsed, dict) else {}

    def _resolve_obj_from_path(self, path_or_obj: Any) -> Any:
        if not isinstance(path_or_obj, str):
            return path_or_obj
        path = path_or_obj.strip()
        if not path:
            return None
        try:
            if ":" in path:
                mod_name, attr = path.split(":", 1)
            else:
                mod_name, attr = path.rsplit(".", 1)
            module = importlib.import_module(mod_name)
            return getattr(module, attr)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to resolve path '%s': %s", path, exc)
            return None

    def _get_ocr_processor(self) -> OCRProcessorProtocol | None:
        raw = self._runtime_config.get("ocr_processor")
        proc = self._resolve_obj_from_path(raw)
        if callable(proc) and not hasattr(proc, "process"):
            try:
                proc = proc()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to instantiate ocr_processor: %s", exc)
                return None
        return proc if hasattr(proc, "process") else None

    def _get_agent_runner_class(self) -> Any:
        raw = self._runtime_config.get("agent_runner_class")
        cls = self._resolve_obj_from_path(raw)
        return cls or AgentRunner

    def _include_section_images(self) -> bool:
        return bool(self._runtime_config.get("include_section_images", True))

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
                page_source_type = self._classify_page_source_type(page, pdf_blocks)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_img:
                    tmp_img.write(page_image.image_bytes)
                    tmp_img.flush()
                    img_path = tmp_img.name

                    if page_source_type == "image_based":
                        ocr_layout_blocks = self._call_ocr_processor(img_path, page_index)
                        llm_pdf_blocks: list[Block] = []
                    else:
                        ocr_layout_blocks = []
                        llm_pdf_blocks = pdf_blocks

                    try:
                        raw_blocks = self._build_raw_blocks(
                            page_index=page_index,
                            page_image=page_image,
                            pdf_text_blocks=llm_pdf_blocks,
                            ocr_layout_blocks=ocr_layout_blocks,
                            page_source_type=page_source_type,
                        )
                        llm_result = self._call_llm(raw_blocks=raw_blocks, img_path=img_path)
                        page_type = str(llm_result.get("page_type", "bill_summary_page"))
                        section_candidates = llm_result.get("sections", [])
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("LLM call/parse failed; fallback enabled: %s", exc)
                        primary_blocks = llm_pdf_blocks if page_source_type == "text_based" else ocr_layout_blocks
                        page_type = self._infer_page_type(primary_blocks)
                        section_candidates = []

                if page_type not in _ALLOWED_PAGE_TYPES:
                    primary_blocks = llm_pdf_blocks if page_source_type == "text_based" else ocr_layout_blocks
                    page_type = self._infer_page_type(primary_blocks)

                final_blocks = llm_pdf_blocks if page_source_type == "text_based" else ocr_layout_blocks

                parsed_page = PageParseResult(
                    page_index=page_index,
                    page_type=page_type,
                    page_source_type=page_source_type,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=final_blocks,
                    section_candidates=section_candidates,
                    page_image=page_image,
                )

                output_data.append(
                    {
                        "page_index": page_index,
                        "page_type": page_type,
                        "page_sections": self._resolve_sections(parsed_page),
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
            blocks.append(Block(f"p{page_index}_b{idx}", txt, [float(x1), float(y1), float(x2), float(y2)], page_index, "pdf_text", "text", 1.0))
        return blocks

    def _call_ocr_processor(self, img_path: str, page_index: int) -> list[Block]:
        processor = self._get_ocr_processor()
        if processor is None:
            return []

        try:
            raw_items = processor.process(img_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ocr_processor.process failed: %s", exc)
            return []

        if not isinstance(raw_items, list):
            return []

        blocks: list[Block] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            bbox = self._coerce_bbox_from_coordinate(item.get("coordinate"))
            if not bbox:
                continue
            label = str(item.get("label", "text")).strip().lower() or "text"
            score = max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0)))
            blocks.append(Block(f"p{page_index}_ocr{idx}", "", bbox, page_index, "ocr_layout", label, score))
        return blocks

    def _build_raw_blocks(
        self,
        page_index: int,
        page_image: PageImage,
        pdf_text_blocks: list[Block],
        ocr_layout_blocks: list[Block],
        page_source_type: Literal["text_based", "image_based"],
    ) -> dict[str, Any]:
        return {
            "page_index": page_index,
            "page_source_type": page_source_type,
            "page_size": {"width": page_image.width, "height": page_image.height},
            "pdf_text_blocks": [{"block_id": b.block_id, "text": b.text, "bbox": b.bbox} for b in pdf_text_blocks],
            "ocr_layout_blocks": [{"block_id": b.block_id, "bbox": b.bbox, "block_type": b.block_type, "confidence": b.confidence} for b in ocr_layout_blocks],
        }

    def _classify_page_source_type(self, page: Any, pdf_blocks: list[Block]) -> Literal["text_based", "image_based"]:
        non_empty_text_blocks = len(pdf_blocks)
        total_chars = sum(len(b.text.strip()) for b in pdf_blocks)
        text_coverage = self._estimate_text_coverage(pdf_blocks, float(page.rect.width), float(page.rect.height))

        if non_empty_text_blocks == 0:
            return "image_based"
        if total_chars <= 5 and text_coverage < 0.002:
            return "image_based"
        if total_chars < 30 and non_empty_text_blocks <= 2 and text_coverage < 0.008:
            return "image_based"
        if non_empty_text_blocks >= 8 or total_chars >= 120:
            return "text_based"
        if non_empty_text_blocks >= 3 and total_chars >= 40 and text_coverage >= 0.01:
            return "text_based"
        return "image_based"

    def _estimate_text_coverage(self, pdf_blocks: list[Block], page_width: float, page_height: float) -> float:
        page_area = max(page_width * page_height, 1.0)
        covered = 0.0
        for block in pdf_blocks:
            clipped = self._clip_bbox(block.bbox, page_width, page_height)
            w = max(0.0, clipped[2] - clipped[0])
            h = max(0.0, clipped[3] - clipped[1])
            covered += (w * h)
        return max(0.0, min(1.0, covered / page_area))

    def _call_llm(self, raw_blocks: dict[str, Any], img_path: str) -> dict[str, Any]:
        runner_class = self._get_agent_runner_class()
        if runner_class is None:
            raise RuntimeError("AgentRunner is not available in runtime")

        llm_response = runner_class("common").generate_result_by_llm(
            input_variables=["blocks"],
            prompt_template=["layoutDetection"],
            blocks=raw_blocks,
            llm_name="VL",
            image_url=image_to_base64(img_path),
        )
        return self._parse_llm_sections(self._extract_json_from_llm_response(llm_response), raw_blocks.get("page_index", 0))

    def _extract_json_from_llm_response(self, llm_response: Any) -> dict[str, Any]:
        if isinstance(llm_response, dict):
            return llm_response

        text = str(llm_response).strip()
        if not text:
            raise LLMResponseParseError("Empty llm response")

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = match.group(1) if match else text
        if not match:
            left, right = candidate.find("{"), candidate.rfind("}")
            if left != -1 and right != -1 and left < right:
                candidate = candidate[left : right + 1]

        try:
            parsed = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            raise LLMResponseParseError("Invalid JSON from llm response") from exc
        if not isinstance(parsed, dict):
            raise LLMResponseParseError("LLM response JSON must be an object")
        return parsed

    def _parse_llm_sections(self, data: dict[str, Any], page_index: int) -> dict[str, Any]:
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
            duplicate = any(self._bbox_iou(ocr.bbox, b.bbox) >= 0.7 and self._text_similarity(ocr.text, b.text) >= 0.8 for b in merged)
            if not duplicate:
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
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

    def _normalize_bbox(self, bbox: list[float], page_width: float, page_height: float) -> list[float]:
        clipped = self._clip_bbox(bbox, page_width, page_height)
        if page_width <= 0 or page_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = clipped
        return [max(0.0, min(1.0, x1 / page_width)), max(0.0, min(1.0, y1 / page_height)), max(0.0, min(1.0, x2 / page_width)), max(0.0, min(1.0, y2 / page_height))]

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
        if not self._include_section_images():
            return ""
        if len(bbox_normalized) != 4:
            return base64.b64encode(page_image.image_bytes).decode("ascii")
        x1 = int(max(0, min(page_image.width - 1, round(bbox_normalized[0] * page_image.width))))
        y1 = int(max(0, min(page_image.height - 1, round(bbox_normalized[1] * page_image.height))))
        x2 = int(max(0, min(page_image.width, round(bbox_normalized[2] * page_image.width))))
        y2 = int(max(0, min(page_image.height, round(bbox_normalized[3] * page_image.height))))
        if x2 <= x1 or y2 <= y1:
            return base64.b64encode(page_image.image_bytes).decode("ascii")

        channels = max(1, page_image.channels)
        row_stride = page_image.width * channels
        chunks: list[bytes] = []
        for y in range(y1, y2):
            start = y * row_stride + x1 * channels
            end = y * row_stride + x2 * channels
            chunks.append(page_image.rgb_bytes[start:end])
        cropped = b"".join(chunks)
        return base64.b64encode(cropped if cropped else page_image.image_bytes).decode("ascii")

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
            pts: list[tuple[float, float]] = []
            for p in coordinate:
                if isinstance(p, (list, tuple)) and len(p) >= 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                    pts.append((float(p[0]), float(p[1])))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return [min(xs), min(ys), max(xs), max(ys)]
        return []

    def _infer_page_type(self, blocks: list[Block]) -> Literal["bill_summary_page", "bill_charge_page", "bill_detail_page"]:
        texts = [b.text.strip().lower() for b in blocks if b.text.strip()]
        if not texts:
            return "bill_summary_page"
        freq: dict[str, int] = {}
        for t in texts:
            freq[t] = freq.get(t, 0) + 1
        repeated = sum(1 for c in freq.values() if c >= 2)
        if repeated >= 3 or len(texts) >= 30:
            return "bill_detail_page"
        if any(k in t for t in texts for k in ["charge", "subtotal", "total", "adjustment"]):
            return "bill_charge_page"
        return "bill_summary_page"


def parse_bill_pdf(pdf_path: str, config: dict | None = None) -> dict:
    """Parse bill PDF and return final structured dictionary."""
    parser = BillLayoutParser(config_overrides=config)
    return parser.parse(pdf_path)
