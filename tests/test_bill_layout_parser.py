from typing import Any
import re
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _fitz_module() -> Any:
    return pytest.importorskip("fitz")


def _create_pdf(path: Path, pages_text: list[str | None]) -> None:
    fitz = _fitz_module()
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self._payload


def test_invalid_pdf_path() -> None:
    with pytest.raises(FileNotFoundError):
        parse_bill_pdf("/non/exist/path.pdf")


def test_parse_minimal_pdf_with_mocked_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "minimal.pdf"
    _create_pdf(pdf_path, ["Subtotal 100"])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[10,10,400,300],"source_block_ids":["p0_b0"],"confidence":0.9}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))

    result = parse_bill_pdf(
        str(pdf_path),
        config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"},
    )
    assert "output_data" in result
    assert isinstance(result["output_data"], list)
    assert len(result["output_data"]) == 1


def test_blank_page_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _create_pdf(pdf_path, [None])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse({"choices": [{"message": {"content": "invalid-json"}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})

    assert len(result["output_data"]) == 1
    page = result["output_data"][0]
    assert "page_index" in page and page["page_index"] == 0
    assert "page_type" in page and page["page_type"] in _ALLOWED_PAGE_TYPES
    assert "page_sections" in page and isinstance(page["page_sections"], list)


def test_bbox_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "bbox.pdf"
    _create_pdf(pdf_path, ["Total amount"])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[-10,20,2000,9000],"source_block_ids":[],"confidence":0.6}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})

    for section in result["output_data"][0]["page_sections"]:
        bbox = section["bbox"]
        assert len(bbox) == 4
        assert all(0.0 <= v <= 1.0 for v in bbox)


def test_image_base64_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "img.pdf"
    _create_pdf(pdf_path, ["hello"])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,200,120],"source_block_ids":[],"confidence":0.8}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})

    for section in result["output_data"][0]["page_sections"]:
        assert "image_base64" in section
        assert isinstance(section["image_base64"], str)


def test_llm_code_fence_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "fence.pdf"
    _create_pdf(pdf_path, ["hello"])

    fenced = """```json
{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,200,100],"source_block_ids":[],"confidence":0.8}]}
```"""

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse({"choices": [{"message": {"content": fenced}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})
    assert len(result["output_data"][0]["page_sections"]) >= 1


def test_invalid_llm_json_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "invalid-llm.pdf"
    _create_pdf(pdf_path, ["charge subtotal total"])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse({"choices": [{"message": {"content": "{bad-json"}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})
    assert result["output_data"][0]["page_type"] in _ALLOWED_PAGE_TYPES
    assert result["output_data"][0]["page_sections"] == []


def test_doclayout_failure_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "doclayout-fail.pdf"
    _create_pdf(pdf_path, ["line1"])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            raise RuntimeError("doclayout down")
        return _FakeResponse({"choices": [{"message": {"content": '{"page_type":"bill_summary_page","sections":[]}'}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})
    assert len(result["output_data"]) == 1


def test_section_deduplication() -> None:
    parser = BillLayoutParser(ParserConfig())
    sections = [
        SectionCandidate("a", "bill_charge_page.charge_items", "table", [0, 0, 100, 100], ["b1"], 0.4),
        SectionCandidate("b", "bill_charge_page.charge_items", "table", [5, 5, 95, 95], ["b1", "b2"], 0.9),
    ]
    deduped = parser._deduplicate_sections(sections, [])
    assert len(deduped) == 1
    assert deduped[0].section_id == "b"


def test_allowed_labels_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "labels.pdf"
    _create_pdf(pdf_path, ["x"])

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse(
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

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})

    page = result["output_data"][0]
    assert page["page_type"] in _ALLOWED_PAGE_TYPES
    for section in page["page_sections"]:
        assert section["structure_type"] in _ALLOWED_STRUCTURE_TYPES
        assert section["region_type"] in _ALLOWED_REGION_TYPES
        assert re.match(r"^p0_s\d+$", section["section_id"])


def test_doclayout_processor_coordinate_label_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "processor.pdf"
    _create_pdf(pdf_path, ["hello"])

    seen_payload: dict[str, Any] = {}

    def _mock_post(url, json, headers, timeout):
        _ = (headers, timeout)
        if "doclayout" in url:
            seen_payload.update(json)
            return _FakeResponse(
                {
                    "blocks": [
                        {
                            "coordinate": [[10, 10], [100, 10], [100, 40], [10, 40]],
                            "label": "text",
                            "score": 0.88,
                            "text": "ocr line",
                        }
                    ]
                }
            )
        return _FakeResponse({"choices": [{"message": {"content": '{"page_type":"bill_summary_page","sections":[]}'}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://x/doclayout", "qwen_url": "http://x/qwen"})

    assert "imgpath" in seen_payload
    assert len(result["output_data"]) == 1
