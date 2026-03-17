import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bill_layout_parser import BillLayoutParser, Block, ParserConfig, parse_bill_pdf


class _FakePixmap:
    width = 1000
    height = 1000

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


def test_doclayout_success_response_parsed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")

    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 50.0, "pdf text")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    def _mock_post(url, json, headers, timeout):
        _ = (url, json, headers, timeout)
        return _FakeResponse(
            {
                "layout_blocks": [
                    {
                        "label": "bill_charge_page.charge_items",
                        "bbox": [0.0, 0.0, 100.0, 100.0],
                        "confidence": 0.95,
                    }
                ],
                "ocr_blocks": [
                    {
                        "text": "ocr extra",
                        "bbox": [200.0, 200.0, 320.0, 260.0],
                    }
                ],
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://local/layout", "timeout": 3})

    assert len(result["output_data"]) == 1
    sections = result["output_data"][0]["page_sections"]
    assert isinstance(sections, list)


def test_doclayout_failure_does_not_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")

    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 50.0, "pdf text")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)
    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(post=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network fail"))),
    )

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://local/layout"})
    assert len(result["output_data"]) == 1
    assert result["output_data"][0]["page_index"] == 0


def test_merge_pdf_and_ocr_blocks_deduplicates() -> None:
    parser = BillLayoutParser(ParserConfig())
    pdf_blocks = [
        Block("p0_b0", "Total Amount", [0.0, 0.0, 100.0, 20.0], 0, "pdf_text"),
    ]
    ocr_blocks = [
        Block("p0_ocr0", "Total Amount", [1.0, 0.0, 101.0, 20.0], 0, "ocr"),
        Block("p0_ocr1", "New Value", [150.0, 150.0, 300.0, 200.0], 0, "ocr"),
    ]

    merged = parser._merge_pdf_and_ocr_blocks(pdf_blocks, ocr_blocks)
    assert len(merged) == 2


def test_no_pdf_text_with_ocr_still_produces_sections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")

    fake_doc = _FakeDoc([_FakePage([])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    monkeypatch.setattr(
        "bill_layout_parser.requests",
        SimpleNamespace(
            post=lambda *args, **kwargs: _FakeResponse(
                {
                    "ocr": [
                        {"text": "scanned line", "bbox": [10.0, 20.0, 110.0, 60.0]},
                    ]
                }
            )
        ),
    )

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://local/layout"})
    assert len(result["output_data"]) == 1
    assert result["output_data"][0]["page_index"] == 0
    assert len(result["output_data"][0]["page_sections"]) >= 1
