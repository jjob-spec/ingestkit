"""Form field extractors.

Subpackage containing extraction backends:
- native_pdf: PyMuPDF-based form widget extraction
- ocr_overlay: Template overlay + per-field OCR extraction
- excel_cell: openpyxl cell value mapping extraction
"""

from ingestkit_forms.extractors.excel_cell import ExcelCellExtractor
from ingestkit_forms.extractors.native_pdf import NativePDFExtractor
from ingestkit_forms.extractors.ocr_overlay import OCROverlayExtractor
from ingestkit_forms.extractors.vlm_fallback import VLMFieldExtractor

__all__ = [
    "ExcelCellExtractor",
    "NativePDFExtractor",
    "OCROverlayExtractor",
    "VLMFieldExtractor",
]
