import json
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


def test_qwen_success_valid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 40.0, "subtotal 100")])])
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
                            "content": json_module.dumps(
                                {
                                    "page_type": "bill_charge_page",
                                    "sections": [
                                        {
                                            "region_type": "bill_charge_page.charge_items",
                                            "structure_type": "table",
                                            "bbox": [10, 20, 500, 700],
                                            "source_block_ids": ["p0_b0"],
                                            "confidence": 0.9,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }
        )

    json_module = json
    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))

    result = parse_bill_pdf(
        str(pdf_path),
        config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen", "qwen_model_name": "qwen3.5-35b"},
    )
    page = result["output_data"][0]
    assert page["page_type"] in {"bill_summary_page", "bill_charge_page", "bill_detail_page"}
    assert len(page["page_sections"]) == 1
    assert page["page_sections"][0]["structure_type"] in {"table", "text", "kv", "image"}
    assert page["page_sections"][0]["region_type"] in {
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


def test_qwen_markdown_code_fence_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 40.0, "hello")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    fenced = """```json
{"page_type":"bill_summary_page","sections":[{"region_type":"bill_summary_page.account_information","structure_type":"kv","bbox":[1,2,300,200],"source_block_ids":["p0_b0"]}]}
```"""

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse({"choices": [{"message": {"content": fenced}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen"})
    sec = result["output_data"][0]["page_sections"][0]
    assert sec["structure_type"] in {"table", "text", "kv", "image"}
    assert sec["region_type"] in {
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


def test_qwen_invalid_json_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([(0.0, 0.0, 100.0, 40.0, "charge subtotal total")])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr_blocks": []})
        return _FakeResponse({"choices": [{"message": {"content": "not-json-response"}}]})

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen"})
    page = result["output_data"][0]
    assert page["page_type"] == "bill_charge_page"
    assert page["page_sections"] == []


def test_merge_pdf_and_ocr_blocks_deduplicates() -> None:
    parser = BillLayoutParser(ParserConfig())
    pdf_blocks = [Block("p0_b0", "Total Amount", [0.0, 0.0, 100.0, 20.0], 0, "pdf_text")]
    ocr_blocks = [
        Block("p0_ocr0", "Total Amount", [1.0, 0.0, 101.0, 20.0], 0, "ocr"),
        Block("p0_ocr1", "New Value", [150.0, 150.0, 300.0, 200.0], 0, "ocr"),
    ]
    merged = parser._merge_pdf_and_ocr_blocks(pdf_blocks, ocr_blocks)
    assert len(merged) == 2


def test_no_pdf_text_with_ocr_still_produces_section_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF")
    fake_doc = _FakeDoc([_FakePage([])])
    monkeypatch.setattr(BillLayoutParser, "_load_pdf", lambda self, path: fake_doc)

    def _mock_post(url, json, headers, timeout):
        _ = (json, headers, timeout)
        if "doclayout" in url:
            return _FakeResponse({"ocr": [{"text": "scanned", "bbox": [10, 10, 100, 50]}]})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"page_type":"bill_detail_page","sections":[{"region_type":"bill_detail_page.detail_record_display_content","structure_type":"text","bbox":[10,10,100,50],"source_block_ids":["p0_ocr0"]}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bill_layout_parser.requests", SimpleNamespace(post=_mock_post))

    result = parse_bill_pdf(str(pdf_path), config={"doclayout_url": "http://svc/doclayout", "qwen_url": "http://svc/qwen"})
    assert result["output_data"][0]["page_index"] == 0
    assert len(result["output_data"][0]["page_sections"]) >= 1
