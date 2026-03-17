import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bill_layout_parser import BillLayoutParser, ParserConfig, parse_bill_pdf


def _require_fitz() -> object:
    if importlib.util.find_spec("fitz") is None:
        pytest.skip("PyMuPDF (fitz) is not installed in this environment")
    import fitz

    return fitz


def _create_pdf(path: Path, text: str | None = None, pages: int = 1) -> None:
    fitz = _require_fitz()
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_invalid_pdf_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse_bill_pdf("/nonexistent/file.pdf")


def test_output_has_output_data_with_mocked_external_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path, text="hello bill")

    monkeypatch.setattr(BillLayoutParser, "_call_doclayout", lambda self, page_image: {"regions": []})
    monkeypatch.setattr(
        BillLayoutParser,
        "_call_qwen",
        lambda self, page_image, blocks, doclayout_result: {
            "page_type": "bill_charge_page",
            "default_structure_type": "text",
        },
    )

    result = parse_bill_pdf(str(pdf_path))

    assert "output_data" in result
    assert isinstance(result["output_data"], list)
    assert len(result["output_data"]) == 1
    assert result["output_data"][0]["page_index"] == 0


def test_bbox_is_list_of_four(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path, text="line a")

    monkeypatch.setattr(
        BillLayoutParser,
        "_call_doclayout",
        lambda self, page_image: {
            "regions": [
                {
                    "region_type": "bill_charge_page.charge_items",
                    "structure_type": "table",
                    "bbox": [10.0, 10.0, 120.0, 120.0],
                    "confidence": 0.9,
                }
            ]
        },
    )

    result = parse_bill_pdf(str(pdf_path))
    first_page = result["output_data"][0]
    first_section = first_page["page_sections"][0]

    bbox = first_section["bbox"]
    assert isinstance(bbox, list)
    assert len(bbox) == 4


def test_blank_page_still_returns_page_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _create_pdf(pdf_path, text=None)

    monkeypatch.setattr(BillLayoutParser, "_call_doclayout", lambda self, page_image: {"regions": []})
    monkeypatch.setattr(
        BillLayoutParser,
        "_call_qwen",
        lambda self, page_image, blocks, doclayout_result: {
            "page_type": "bill_summary_page",
            "default_structure_type": "text",
        },
    )

    result = parse_bill_pdf(str(pdf_path))
    assert len(result["output_data"]) == 1
    assert result["output_data"][0]["page_index"] == 0
    assert isinstance(result["output_data"][0]["page_sections"], list)


def test_extract_pdf_blocks_handles_empty_text_blocks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _create_pdf(pdf_path, text=None)

    parser = BillLayoutParser(ParserConfig())
    doc = parser._load_pdf(pdf_path)
    try:
        page = doc.load_page(0)
        blocks = parser._extract_pdf_blocks(page, page_index=0)
    finally:
        doc.close()

    assert isinstance(blocks, list)
    assert blocks == []
