import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import fitz  # PyMuPDF
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.utils.locks import pymupdf_lock


if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument


_log = logging.getLogger(__name__)


def _page_geometry_from_pymupdf(
    page: fitz.Page,
    angle: float = 0.0,
    boundary_type: PdfPageBoundaryType = PdfPageBoundaryType.CROP_BOX,
) -> PdfPageGeometry:
    # PyMuPDF uses a top-left origin with y increasing downwards.
    # Build a generic geometry using the page rectangle for all boxes.
    with pymupdf_lock:
        rect = page.rect  # already the visible page rectangle

    width = float(rect.width)
    height = float(rect.height)

    # Build a bottom-left origin BoundingBox for geometry construction
    bbox = BoundingBox(
        l=0.0,
        b=0.0,
        r=width,
        t=height,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    return PdfPageGeometry(
        angle=angle,
        rect=BoundingRectangle.from_bounding_box(bbox),
        boundary_type=boundary_type,
        art_bbox=bbox,
        bleed_bbox=bbox,
        crop_bbox=bbox,
        media_bbox=bbox,
        trim_bbox=bbox,
    )


class PyMuPdfPageBackend(PdfPageBackend):
    def __init__(self, doc: fitz.Document, document_hash: str, page_no: int):
        self._doc = doc
        self._page = doc[page_no]
        self.document_hash = document_hash
        self.page_no = page_no
        self.valid = self._page is not None

    def _color_int_to_hex(self, color_int: int) -> str:
        """Convert PyMuPDF color integer to hex string."""
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"

    def _create_bounding_box(self, bbox_tuple: tuple, coord_origin: CoordOrigin = CoordOrigin.TOPLEFT) -> BoundingBox:
        """Create BoundingBox from tuple with consistent coordinate origin."""
        l, t, r, b = bbox_tuple
        return BoundingBox(
            l=float(l),
            t=float(t),
            r=float(r),
            b=float(b),
            coord_origin=coord_origin
        )

    def _create_font_metadata(self, span: dict, text: str) -> dict:
        """Create font metadata dictionary from a span."""
        font_name = span.get("font", "")
        font_size = span.get("size", 0.0)
        flags = span.get("flags", 0)
        color_int = span.get("color", 0)
        
        # Convert color to hex format
        color_hex = self._color_int_to_hex(color_int)
        
        # Determine font properties from flags
        is_bold = bool(flags & 1)
        is_italic = bool(flags & 2)
        is_monospaced = bool(flags & 4)
        
        # Extract alt_family_name (base font name without style indicators)
        alt_family_name = font_name
        if "-" in font_name:
            alt_family_name = font_name.split("-")[0]
        
        # Calculate bounding box height
        l, t, r, b = span.get("bbox", (0, 0, 0, 0))
        bbox_height = float(b) - float(t)
        
        return {
            "text": text,
            "font_family": font_name,
            "font_size": font_size,
            "name": font_name,
            "color": color_hex,
            "bbox": [float(l), float(t), float(r), float(b)],
            "bold": is_bold,
            "italic": is_italic,
            "monospaced": is_monospaced,
            "bbox_height": bbox_height,
            "alt_family_name": alt_family_name
        }

    def _create_text_cell(self, index: int, text: str, bbox: BoundingBox, 
                          font_metadata: List[dict], is_word_level: bool = False) -> TextCell:
        """Create a TextCell with consistent metadata structure."""
        if is_word_level:
            # Word-level specific metadata
            base_metadata = font_metadata[0] if font_metadata else {}
            metadata = [{
                **base_metadata,
                "subset": False,
                "weight": 700 if base_metadata.get("bold", False) else 400,
                "spaceAfter": 0
            }]
        else:
            metadata = font_metadata
        
        return TextCell(
            index=index,
            text=text,
            orig=text,
            rect=BoundingRectangle.from_bounding_box(bbox),
            from_ocr=False,
            font_metadata=metadata
        )

    def is_valid(self) -> bool:
        return self.valid

    def _compute_text_cells(self) -> List[TextCell]:
        if not self._page:
            return []

        with pymupdf_lock:
            # Use dict mode to iterate lines with their bounding boxes
            text_dict = self._page.get_text("dict")

        cells: List[TextCell] = []
        cell_counter = 0

        for block in text_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue  # skip non-text blocks
            for line in block.get("lines", []):
                # Merge the spans of a line into a single text cell
                spans = line.get("spans", [])
                if not spans:
                    continue
                text_content = "".join(span.get("text", "") for span in spans).strip()
                if text_content == "":
                    continue

                # Line bbox is provided in top-left origin by PyMuPDF
                bbox_tl = self._create_bounding_box(line.get("bbox", (0, 0, 0, 0)))

                # Extract font information directly from spans
                font_metadata = []
                for span in spans:
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    font_metadata.append(self._create_font_metadata(span, text))
                
                # Create a TextCell with the text content and font info
                cell = self._create_text_cell(cell_counter, text_content, bbox_tl, font_metadata)
                cells.append(cell)
                cell_counter += 1

        # Re-index from 1, as other backends do
        for i, cell in enumerate(cells, 1):
            cell.index = i

        return cells

    def _compute_word_cells(self) -> List[TextCell]:
        if not self._page:
            return []

        with pymupdf_lock:
            text_dict = self._page.get_text("dict")

        word_cells: List[TextCell] = []
        span_index = 0

        for block in text_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_content = span.get("text", "").strip()
                    if text_content == "":
                        continue

                    bbox_tl = self._create_bounding_box(span.get("bbox", (0, 0, 0, 0)))

                    font_metadata = [self._create_font_metadata(span, text_content)]

                    # Create a TextCell with the text content
                    cell = self._create_text_cell(span_index, text_content, bbox_tl, font_metadata, is_word_level=True)
                    
                    word_cells.append(cell)
                    span_index += 1

        # Re-index from 1
        for i, cell in enumerate(word_cells, 1):
            cell.index = i

        return word_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        AREA_THRESHOLD = 0  # same as other backends
        if not self._page:
            return []

        with pymupdf_lock:
            raw = self._page.get_text("rawdict")

        for block in raw.get("blocks", []):
            # In rawdict, images have type 1
            if block.get("type", 0) == 1:
                l, t, r, b = block.get("bbox", (0, 0, 0, 0))
                cropbox = BoundingBox(
                    l=float(l),
                    t=float(t),
                    r=float(r),
                    b=float(b),
                    coord_origin=CoordOrigin.TOPLEFT,
                ).scaled(scale=scale)

                if cropbox.area() > AREA_THRESHOLD:
                    yield cropbox

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        if not self._page:
            return ""

        # Ensure top-left origin for PyMuPDF
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            page_size = self.get_size()
            bbox = bbox.to_top_left_origin(page_height=page_size.height)

        l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b
        clip_rect = fitz.Rect(l, t, r, b)

        with pymupdf_lock:
            # Use textbox extraction limited by clip rectangle
            text = self._page.get_textbox(clip_rect)

        return text or ""

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        if not self.valid or not self._page:
            return None

        text_cells = self._compute_text_cells()
        word_cells = self._compute_word_cells()

        with pymupdf_lock:
            angle = float(getattr(self._page, "rotation", 0) or 0)

        dimension = _page_geometry_from_pymupdf(self._page, angle=angle)

        segmented_page = SegmentedPdfPage(
            dimension=dimension,
            textline_cells=text_cells,
            char_cells=[],
            word_cells=word_cells,
            has_textlines=len(text_cells) > 0,
            has_words=len(word_cells) > 0,
            has_chars=False,
        )
        
        return segmented_page

    def get_text_cells(self) -> Iterable[TextCell]:
        return self._compute_text_cells()

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        if not self._page:
            # Return a 1x1 transparent pixel as a safe fallback
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        page_size = self.get_size()

        if cropbox is None:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
        elif cropbox.coord_origin != CoordOrigin.TOPLEFT:
            cropbox = cropbox.to_top_left_origin(page_size.height)

        l, t, r, b = cropbox.l, cropbox.t, cropbox.r, cropbox.b
        clip_rect = fitz.Rect(l, t, r, b)

        with pymupdf_lock:
            pix = self._page.get_pixmap(
                matrix=fitz.Matrix(scale * 1.5, scale * 1.5),
                clip=clip_rect,
                alpha=False,
            )

        # Convert to PIL and then resize for sharper image, similar to pypdfium2 backend
        img = Image.open(BytesIO(pix.tobytes("png")))
        target_w = round(cropbox.width * scale)
        target_h = round(cropbox.height * scale)
        if target_w > 0 and target_h > 0:
            img = img.resize((target_w, target_h))
        return img

    def get_size(self) -> Size:
        if not self._page:
            return Size(width=0, height=0)
        with pymupdf_lock:
            rect = self._page.rect
            return Size(width=float(rect.width), height=float(rect.height))

    def unload(self):
        self._page = None


class PyMuPdfDocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)
        try:
            with pymupdf_lock:
                if isinstance(self.path_or_stream, (str, Path)):
                    self._doc = fitz.open(self.path_or_stream)  # type: ignore[arg-type]
                else:
                    # BytesIO stream
                    self._doc = fitz.open(stream=self.path_or_stream.read(), filetype="pdf")
        except Exception as e:
            raise RuntimeError(
                f"pymupdf could not load document with hash {self.document_hash}"
            ) from e

    def page_count(self) -> int:
        with pymupdf_lock:
            try:
                return int(self._doc.page_count)  # type: ignore[attr-defined]
            except AttributeError:
                return len(self._doc)

    def load_page(self, page_no: int) -> PyMuPdfPageBackend:
        with pymupdf_lock:
            return PyMuPdfPageBackend(self._doc, self.document_hash, page_no)

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        super().unload()
        with pymupdf_lock:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None


