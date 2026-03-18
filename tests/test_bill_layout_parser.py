import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bill_layout_parser import (  # noqa: E402
    BillLayoutParser,
    SectionCandidate,
    _ALLOWED_PAGE_TYPES,
    _ALLOWED_REGION_TYPES,
    _ALLOWED_STRUCTURE_TYPES,
    parse_bill_pdf,
)


class _FakeProcessor:
    def __init__(self, items: list[dict[str, Any]], raise_error: bool = False) -> None:
        self.items = items
        self.raise_error = raise_error
        self.called_with: str | None = None

    def process(self, img_path: str) -> list[dict[str, Any]]:
        self.called_with = img_path
        if self.raise_error:
            raise RuntimeError("processor failed")
        return self.items


class _FakeRunner:
    last_kwargs: dict[str, Any] = {}
    response_text: Any = '{"page_type":"bill_summary_page","sections":[]}'
    should_raise: bool = False

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def generate_result_by_llm(self, **kwargs: Any) -> Any:
        _FakeRunner.last_kwargs = kwargs
        if _FakeRunner.should_raise:
            raise RuntimeError("llm failed")
        return _FakeRunner.response_text


class _FakePixmap:
    width = 1000
    height = 1000
    n = 3
    samples = b"\x80" * (1000 * 1000 * 3)

    def tobytes(self, fmt: str) -> bytes:
        assert fmt == "png"
        return b"fake-png"


class _FakePage:
    def __init__(self, blocks: list[tuple[float, float, float, float, str]]) -> None:
        self.rect = SimpleNamespace(width=1000.0, height=1000.0)
        self._blocks = blocks

    def get_pixmap(self, **kwargs: Any) -> _FakePixmap:
        _ = kwargs
        return _FakePixmap()

    def get_text(self, mode: str) -> list[tuple[float, float, float, float, str]]:
        assert mode == "blocks"
        return self._blocks


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages
        self.page_count = len(pages)

    def load_page(self, page_index: int) -> _FakePage:
        return self.pages[page_index]

    def close(self) -> None:
        return None


def _create_stub_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n")


def _patch_doc(monkeypatch: pytest.MonkeyPatch, pages: list[_FakePage]) -> None:
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: _FakeDoc(pages))


def test_invalid_pdf_path() -> None:
    with pytest.raises(FileNotFoundError):
        parse_bill_pdf("/not/exist/a.pdf")


def test_parse_minimal_pdf_with_mocked_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "a.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "subtotal")])])

    _FakeRunner.should_raise = False
    _FakeRunner.response_text = '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[10,10,300,200],"source_block_ids":["p0_b0"],"confidence":0.9}]}'
    result = parse_bill_pdf(
        str(pdf_path),
        {"ocr_processor": _FakeProcessor([{"coordinate": [100, 100, 200, 180], "label": "text", "score": 0.98}]), "agent_runner_class": _FakeRunner},
    )

    assert len(result["output_data"]) == 1
    call_args = _FakeRunner.last_kwargs
    assert call_args["input_variables"] == ["blocks"]
    assert call_args["prompt_template"] == ["layoutDetection"]
    assert call_args["llm_name"] == "VL"
    assert "blocks" in call_args


def test_blank_page_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "b.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([])])

    _FakeRunner.should_raise = True
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "agent_runner_class": _FakeRunner})
    page = result["output_data"][0]
    assert page["page_index"] == 0
    assert page["page_type"] in _ALLOWED_PAGE_TYPES
    assert isinstance(page["page_sections"], list)
    _FakeRunner.should_raise = False


def test_bbox_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "c.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "x")])])

    _FakeRunner.response_text = '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[-10,-20,2000,5000],"source_block_ids":[],"confidence":0.7}]}'
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "agent_runner_class": _FakeRunner})
    bbox = result["output_data"][0]["page_sections"][0]["bbox"]
    assert all(0.0 <= v <= 1.0 for v in bbox)


def test_image_base64_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "d.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "x")])])

    _FakeRunner.response_text = '{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,100,80],"source_block_ids":[],"confidence":0.5}]}'
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "agent_runner_class": _FakeRunner})
    assert "image_base64" in result["output_data"][0]["page_sections"][0]


def test_llm_code_fence_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "e.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "x")])])

    _FakeRunner.response_text = """```json
{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,100,80],"source_block_ids":[],"confidence":0.6}]}
```"""
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "agent_runner_class": _FakeRunner})
    assert len(result["output_data"][0]["page_sections"]) == 1


def test_invalid_llm_json_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "f.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "charge total")])])

    _FakeRunner.response_text = "invalid"
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "agent_runner_class": _FakeRunner})
    assert result["output_data"][0]["page_type"] in _ALLOWED_PAGE_TYPES
    assert result["output_data"][0]["page_sections"] == []


def test_doclayout_failure_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "g.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "x")])])

    _FakeRunner.response_text = '{"page_type":"bill_summary_page","sections":[]}'
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([], raise_error=True), "agent_runner_class": _FakeRunner})
    assert len(result["output_data"]) == 1


def test_section_deduplication() -> None:
    parser = BillLayoutParser()
    dedup = parser._deduplicate_sections(
        [
            SectionCandidate("a", "bill_charge_page.charge_items", "table", [0, 0, 100, 100], ["b1"], 0.2),
            SectionCandidate("b", "bill_charge_page.charge_items", "table", [5, 5, 95, 95], ["b1", "b2"], 0.9),
        ],
        [],
    )
    assert len(dedup) == 1


def test_allowed_labels_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "h.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([(10, 10, 200, 40, "x")])])

    _FakeRunner.response_text = '{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,100,80],"source_block_ids":[],"confidence":0.5}]}'
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "agent_runner_class": _FakeRunner})
    page = result["output_data"][0]
    assert page["page_type"] in _ALLOWED_PAGE_TYPES
    for section in page["page_sections"]:
        assert section["structure_type"] in _ALLOWED_STRUCTURE_TYPES
        assert section["region_type"] in _ALLOWED_REGION_TYPES
        assert re.match(r"^p0_s\d+$", section["section_id"])


def test_ocr_processor_output_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "i.pdf"
    _create_stub_pdf(pdf_path)
    _patch_doc(monkeypatch, [_FakePage([])])

    proc = _FakeProcessor([
        {"coordinate": [1, 2, 30, 40], "label": "image", "score": 0.98},
        {"coordinate": [[10, 10], [20, 10], [20, 20], [10, 20]], "label": "text", "score": 0.88},
    ])

    _FakeRunner.response_text = '{"page_type":"bill_summary_page","sections":[]}'
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": proc, "agent_runner_class": _FakeRunner})
    assert proc.called_with is not None
    assert len(result["output_data"]) == 1
