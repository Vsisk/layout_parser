import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bill_layout_parser import (  # noqa: E402
    BillLayoutParser,
    ParserConfig,
    SectionCandidate,
    _ALLOWED_PAGE_TYPES,
    _ALLOWED_REGION_TYPES,
    _ALLOWED_STRUCTURE_TYPES,
    parse_bill_pdf,
)


class _FakeProcessor:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.called_with: str | None = None

    def process(self, img_path: str) -> list[dict[str, Any]]:
        self.called_with = img_path
        return self.items


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self.payload


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


def test_invalid_pdf_path() -> None:
    with pytest.raises(FileNotFoundError):
        parse_bill_pdf("/not/exist/a.pdf")


def test_parse_minimal_pdf_with_mocked_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "a.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "subtotal")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    processor = _FakeProcessor(
        [{"coordinate": [100, 100, 200, 180], "label": "text", "score": 0.98}]
    )

    def _mock_post(url, json, headers, timeout):
        _ = (url, json, headers, timeout)
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[10,10,300,200],"source_block_ids":["p0_b0"],"confidence":0.9}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": processor, "qwen_url": "http://qwen"})
    assert "output_data" in result
    assert len(result["output_data"]) == 1


def test_blank_page_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "b.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    processor = _FakeProcessor([])

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(post=lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": "bad-json"}}]})),
    )

    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": processor, "qwen_url": "http://qwen"})
    page = result["output_data"][0]
    assert page["page_index"] == 0
    assert page["page_type"] in _ALLOWED_PAGE_TYPES
    assert isinstance(page["page_sections"], list)


def test_bbox_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "c.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "x")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(
            post=lambda *a, **k: _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[-10,-20,2000,5000],"source_block_ids":[],"confidence":0.7}]}'
                            }
                        }
                    ]
                }
            )
        ),
    )

    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "qwen_url": "http://qwen"})
    bbox = result["output_data"][0]["page_sections"][0]["bbox"]
    assert all(0.0 <= v <= 1.0 for v in bbox)


def test_image_base64_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "d.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "x")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(
            post=lambda *a, **k: _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,100,80],"source_block_ids":[],"confidence":0.5}]}'
                            }
                        }
                    ]
                }
            )
        ),
    )

    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "qwen_url": "http://qwen"})
    assert "image_base64" in result["output_data"][0]["page_sections"][0]


def test_llm_code_fence_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "e.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "x")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    fenced = """```json
{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,100,80],"source_block_ids":[],"confidence":0.6}]}
```"""
    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(post=lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": fenced}}]})),
    )
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "qwen_url": "http://qwen"})
    assert len(result["output_data"][0]["page_sections"]) == 1


def test_invalid_llm_json_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "f.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "charge total")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(post=lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": "invalid"}}]})),
    )
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "qwen_url": "http://qwen"})
    assert result["output_data"][0]["page_type"] in _ALLOWED_PAGE_TYPES
    assert result["output_data"][0]["page_sections"] == []


def test_doclayout_failure_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "g.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "x")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    class _FailProcessor:
        def process(self, img_path: str) -> list[dict[str, Any]]:
            _ = img_path
            raise RuntimeError("processor failed")

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(post=lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": '{"page_type":"bill_summary_page","sections":[]}'}}]})),
    )
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FailProcessor(), "qwen_url": "http://qwen"})
    assert len(result["output_data"]) == 1


def test_section_deduplication() -> None:
    parser = BillLayoutParser(ParserConfig())
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
    fake_doc = _FakeDoc([_FakePage([(10, 10, 200, 40, "x")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(
            post=lambda *a, **k: _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,100,80],"source_block_ids":[],"confidence":0.5}]}'
                            }
                        }
                    ]
                }
            )
        ),
    )

    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": _FakeProcessor([]), "qwen_url": "http://qwen"})
    page = result["output_data"][0]
    assert page["page_type"] in _ALLOWED_PAGE_TYPES
    for section in page["page_sections"]:
        assert section["structure_type"] in _ALLOWED_STRUCTURE_TYPES
        assert section["region_type"] in _ALLOWED_REGION_TYPES
        assert re.match(r"^p0_s\d+$", section["section_id"])


def test_ocr_processor_output_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "i.pdf"
    _create_stub_pdf(pdf_path)
    fake_doc = _FakeDoc([_FakePage([])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, p: fake_doc)

    proc = _FakeProcessor([
        {"coordinate": [1, 2, 30, 40], "label": "image", "score": 0.98},
        {"coordinate": [[10, 10], [20, 10], [20, 20], [10, 20]], "label": "text", "score": 0.88},
    ])

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(post=lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": '{"page_type":"bill_summary_page","sections":[]}'}}]})),
    )
    result = parse_bill_pdf(str(pdf_path), {"ocr_processor": proc, "qwen_url": "http://qwen"})
    assert proc.called_with is not None
    assert len(result["output_data"]) == 1
