import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bill_layout_parser import BillLayoutParser, Block, ParserConfig, SectionCandidate, parse_bill_pdf


class _FakePixmap:
    width = 1000
    height = 1000
    n = 3
    samples = b"\x80" * (1000 * 1000 * 3)

    def tobytes(self, fmt: str) -> bytes:
        assert fmt == "png"
        return b"fake_png"


class _FakePage:
    def __init__(self, blocks: list[tuple[float, float, float, float, str]]):
        self.rect = SimpleNamespace(width=1000.0, height=1000.0)
        self._blocks = blocks

    def get_pixmap(self, **kwargs):
        _ = kwargs
        return _FakePixmap()

    def get_text(self, mode: str):
        assert mode == "blocks"
        return self._blocks


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]):
        self._pages = pages
        self.page_count = len(pages)

    def load_page(self, page_index: int) -> _FakePage:
        return self._pages[page_index]

    def close(self) -> None:
        return None


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


def test_invalid_pdf_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse_bill_pdf("/nonexistent/file.pdf")


def test_bbox_normalized_and_image_base64_and_section_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 60.0, "subtotal 100")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_charge_page","sections":[{"region_type":"bill_charge_page.charge_items","structure_type":"table","bbox":[10,20,500,700],"source_block_ids":["p0_b0"],"confidence":0.9}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen"})

    sec = result["output_data"][0]["page_sections"][0]
    assert all(0.0 <= v <= 1.0 for v in sec["bbox"])
    assert sec["bbox"][0] <= sec["bbox"][2] and sec["bbox"][1] <= sec["bbox"][3]
    assert isinstance(sec["image_base64"], str) and len(sec["image_base64"]) > 0
    assert re.match(r"^p0_s\d+$", sec["section_id"])


def test_deduplicate_conflict_sections() -> None:
    parser = BillLayoutParser(ParserConfig())
    blocks = {"b1": Block("b1", "A", [0, 0, 100, 100], 0, "pdf_text")}
    s1 = SectionCandidate("s1", "bill_charge_page.charge_items", "table", [0, 0, 100, 100], ["b1"], 0.4)
    s2 = SectionCandidate("s2", "bill_charge_page.charge_items", "table", [5, 5, 95, 95], ["b1"], 0.9)
    kept = parser._deduplicate_sections([s1, s2], list(blocks.values()))
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_missing_source_block_ids_still_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[10,10,120,80],"confidence":0.7}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))
    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen"})
    assert len(result["output_data"][0]["page_sections"]) == 1


def test_invalid_bbox_fixed_to_legal_range() -> None:
    parser = BillLayoutParser(ParserConfig())
    fixed = parser._normalize_bbox([-10, 2000, 5, -20], 1000, 1000)
    assert all(0.0 <= v <= 1.0 for v in fixed)
    assert fixed[0] <= fixed[2]
    assert fixed[1] <= fixed[3]


def test_qwen_invalid_json_fallback_page_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 50.0, "charge total")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(
            post=lambda url, json, headers, timeout: _FakeResponse({"ocr_blocks": []})
            if "doclayout" in url
            else _FakeResponse({"choices": [{"message": {"content": "not-json"}}]})
        ),
    )

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen"})
    assert result["output_data"][0]["page_type"] in {"bill_charge_page", "bill_summary_page", "bill_detail_page"}
    assert isinstance(result["output_data"][0]["page_sections"], list)
