import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

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
        self._background_color = None
        # Cache links for this page to reuse across computations
        self._links_cache: Optional[List[Dict[str, Any]]] = None

    def _color_int_to_hex(self, color_int: int) -> str:
        """Convert PyMuPDF color integer to hex string."""
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _get_background_color(self) -> str:
        """Extract the background color of the page.
        
        Returns:
            A hex string representing the most common color in the page,
            which is likely to be the background color.
        """
        if self._background_color is not None:
            return self._background_color
            
        if not self._page:
            return "#ffffff"  # Default white if page is invalid
            
        try:
            with pymupdf_lock:
                # Render the page at a lower resolution for performance
                # 72 dpi is usually sufficient to detect the background color
                pix = self._page.get_pixmap(matrix=fitz.Matrix(72/300, 72/300), alpha=False)
                
                # Get the pixel data
                width = pix.width
                height = pix.height
                
                # Sample pixels from the corners and center of the page
                # This is more efficient than analyzing all pixels
                sample_points = [
                    (0, 0),                    # Top-left
                    (width-1, 0),              # Top-right
                    (0, height-1),             # Bottom-left
                    (width-1, height-1),       # Bottom-right
                    (width//2, height//2),     # Center
                    (width//4, height//4),     # Upper-left quadrant
                    (3*width//4, height//4),   # Upper-right quadrant
                    (width//4, 3*height//4),   # Lower-left quadrant
                    (3*width//4, 3*height//4), # Lower-right quadrant
                ]
                
                colors = []
                for x, y in sample_points:
                    pixel = pix.pixel(x, y)
                    r, g, b = pixel[:3]  # Take first 3 values (RGB)
                    colors.append(f"#{r:02x}{g:02x}{b:02x}")
                
                # Count occurrences of each color
                from collections import Counter
                color_counts = Counter(colors)
                
                # Get the most common color
                self._background_color = color_counts.most_common(1)[0][0]
                
                return self._background_color
        except Exception as e:
            _log.warning(f"Error extracting background color: {e}")
            return "#ffffff"  # Default to white on error

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

    def _extract_region_background_color(self, bbox: tuple) -> str:
        """Extract background color for a specific region on the page.
        
        Args:
            bbox: Bounding box tuple (l, t, r, b)
            
        Returns:
            Background color as hex string
        """
        if not self._page:
            return "#ffffff"
            
        try:
            with pymupdf_lock:
                # Create a small pixmap for the region
                l, t, r, b = bbox
                width = r - l
                height = b - t
                
                # Skip if region is too small
                if width < 1 or height < 1:
                    return "#ffffff"
                    
                # Create a matrix for the region
                matrix = fitz.Matrix(1, 1)  # 1:1 scale
                clip_rect = fitz.Rect(l, t, r, b)
                
                # Get pixmap for the region
                pix = self._page.get_pixmap(matrix=matrix, clip=clip_rect, alpha=False)
                
                # Sample pixels from the region
                sample_points = [
                    (0, 0),                    # Top-left
                    (pix.width-1, 0),         # Top-right
                    (0, pix.height-1),        # Bottom-left
                    (pix.width-1, pix.height-1), # Bottom-right
                    (pix.width//2, pix.height//2), # Center
                ]
                
                colors = []
                for x, y in sample_points:
                    if 0 <= x < pix.width and 0 <= y < pix.height:
                        pixel = pix.pixel(x, y)
                        r, g, b = pixel[:3]  # Take first 3 values (RGB)
                        colors.append(f"#{r:02x}{g:02x}{b:02x}")
                
                if colors:
                    # Return the most common color
                    from collections import Counter
                    color_counts = Counter(colors)
                    return color_counts.most_common(1)[0][0]
                    
        except Exception as e:
            _log.debug(f"Error extracting region background color: {e}")
            
        return "#ffffff"  # Default to white on error

    def _load_links(self) -> List[Dict[str, Any]]:
        """Load and normalize link annotations for this page.
        Returns a list of dicts with keys: rect (l,t,r,b in TOPLEFT), kind, and either uri (str) or page (int), point (x,y), zoom (float|None).
        """
        if self._links_cache is not None:
            return self._links_cache
        links: List[Dict[str, Any]] = []
        if not self._page:
            self._links_cache = links
            return links
        try:
            with pymupdf_lock:
                # 1) Standard link annotations
                for lnk in self._page.get_links():
                    rect: fitz.Rect = lnk.get("from")
                    if not rect:
                        continue
                    l, t, r, b = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
                    info: Dict[str, Any] = {
                        "rect": (l, t, r, b),
                        "kind": lnk.get("kind") or "LINK",
                    }
                    uri = lnk.get("uri")
                    if uri:
                        info["uri"] = uri
                    page = lnk.get("page")
                    dest = lnk.get("to") or lnk.get("dest")
                    if page is not None or dest is not None:
                        info["page"] = int(page) if page is not None else None
                        if dest is not None and isinstance(dest, (list, tuple)) and len(dest) >= 2:
                            info["point"] = {"x": float(dest[0]), "y": float(dest[1])}
                        else:
                            info["point"] = None
                        zoom = lnk.get("zoom")
                        if zoom is not None:
                            info["zoom"] = float(zoom)
                    links.append(info)
        except Exception:
            # Best-effort: ignore link extraction errors
            pass

        # De-duplicate links based on (rect, uri/page/point)
        dedup: List[Dict[str, Any]] = []
        seen: set = set()
        for li in links:
            rect = tuple(li.get("rect", (0, 0, 0, 0)))
            key: tuple
            if "uri" in li and li["uri"]:
                key = (rect, "uri", str(li["uri"]))
            elif "page" in li or "point" in li:
                key = (rect, "dest", str(li.get("page")), str(li.get("point")))
            else:
                key = (rect, "other")
            if key in seen:
                continue
            seen.add(key)
            dedup.append(li)

        self._links_cache = dedup
        return dedup

    def get_links(self) -> List[Dict[str, Any]]:
        return self._load_links()

    def _rects_intersect(self, a: tuple, b: tuple) -> bool:
        al, at, ar, ab = a
        bl, bt, br, bb = b
        return not (ar <= bl or br <= al or ab <= bt or bb <= at)

    def _overlap_area(self, a: tuple, b: tuple) -> float:
        al, at, ar, ab = a
        bl, bt, br, bb = b
        ol = max(al, bl)
        ot = max(at, bt)
        or_ = min(ar, br)
        ob = min(ab, bb)
        return max(0.0, or_ - ol) * max(0.0, ob - ot)

    def _best_link_for_bbox(self, bbox: List[float], links: List[Dict[str, Any]]) -> Optional[Union[str, Dict[str, Any]]]:
        """Find the most likely link overlapping the given bbox using robust heuristics.
        Priority: center-in-rect; fallback to max overlap with minimum area ratio.
        Returns None, a URL string, or an internal link dict.
        """
        if not links or not bbox or len(bbox) != 4:
            return None
        bl, bt, br, bb = bbox
        cx = (bl + br) / 2.0
        cy = (bt + bb) / 2.0
        # First pass: center-in-rect
        for lnk in links:
            rl, rt, rr, rb = lnk["rect"]
            if rl <= cx <= rr and rt <= cy <= rb:
                if lnk.get("uri"):
                    return lnk["uri"]
                link_obj: Dict[str, Any] = {"type": "internal"}
                if "page" in lnk and lnk["page"] is not None:
                    link_obj["page"] = int(lnk["page"]) + 1
                if lnk.get("point"):
                    link_obj["point"] = {"x": lnk["point"]["x"], "y": lnk["point"]["y"]}
                if lnk.get("zoom") is not None:
                    link_obj["zoom"] = lnk["zoom"]
                return link_obj
        # Second pass: max overlap with threshold to avoid spurious large overlays
        span_area = max(1e-6, (br - bl) * (bb - bt))
        best = None
        best_area = 0.0
        for lnk in links:
            area = self._overlap_area((bl, bt, br, bb), lnk["rect"])
            if area > best_area:
                best_area = area
                best = lnk
        # Require at least 20% overlap of the span area
        if best and (best_area / span_area) >= 0.2:
            if best.get("uri"):
                return best["uri"]
            link_obj = {"type": "internal"}
            if "page" in best and best["page"] is not None:
                link_obj["page"] = int(best["page"]) + 1
            if best.get("point"):
                link_obj["point"] = {"x": best["point"]["x"], "y": best["point"]["y"]}
            if best.get("zoom") is not None:
                link_obj["zoom"] = best["zoom"]
            return link_obj
        return None

    def _create_font_metadata(self, span: dict, text: str, span_idx: int = 0, line: dict = None) -> dict:
        """Create font metadata dictionary from a span."""
        font_name = span.get("font", "")
        font_size = span.get("size", 0.0)
        flags = span.get("flags", 0)
        color_int = span.get("color", 0)
        
        # Format the integer color into a hex string
        color_hex = f"#{color_int:06x}"
        
        # PyMuPDF font flags - these are bit flags
        # Bit 0: superscript, Bit 1: italic, Bit 2: serifed, Bit 3: monospaced
        # Bit 4: bold, Bit 5: subset
        italic = bool(flags & (1 << 1))
        monospaced = bool(flags & (1 << 3))
        bold = bool(flags & (1 << 4))
        subset = bool(flags & (1 << 5))
        
        # Calculate weight based on bold flag and font name
        font_name_lower = font_name.lower()
        weight = 700 if bold or "bold" in font_name_lower else 400
        if "light" in font_name_lower:
            weight = 300
        elif "medium" in font_name_lower:
            weight = 500
        elif "semibold" in font_name_lower or "semi" in font_name_lower:
            weight = 600
        elif "heavy" in font_name_lower or "black" in font_name_lower or "extrabold" in font_name_lower:
            weight = 900
            
        # Calculate space after (distance to next span in same line)
        space_after = 0
        if line and span_idx < len(line.get("spans", [])) - 1:
            current_bbox = span.get("bbox", [])
            next_bbox = line.get("spans", [])[span_idx + 1].get("bbox", [])
            if len(current_bbox) == 4 and len(next_bbox) == 4:
                space_after = next_bbox[0] - current_bbox[2]  # left of next - right of current
                
        # Calculate line height from bbox
        l, t, r, b = span.get("bbox", (0, 0, 0, 0))
        line_height = float(b) - float(t)
        
        # Extract background color for this span's region
        background_color = self._extract_region_background_color((l, t, r, b))
        
        # Try to get alternative family name
        alt_family_name = None
        try:
            # Some fonts have alternative names stored in the font object
            if "+" in font_name:
                alt_family_name = font_name.split("+", 1)[1] if "+" in font_name else None
            elif "-" in font_name:
                # Sometimes fonts have variants like "Arial-Bold"
                base_name = font_name.split("-")[0]
                if base_name != font_name:
                    alt_family_name = base_name
        except:
            alt_family_name = None
            
        text_detail = {
            "text": text,
            "font_family": font_name,
            "font_size": round(font_size, 2),
            "name": font_name,
            "color": color_hex,
            "bbox": [round(float(l), 2), round(float(t), 2), round(float(r), 2), round(float(b), 2)],
            "italic": italic,
            "monospaced": monospaced,
            "subset": subset,
            "weight": weight,
            "line_height": round(line_height, 2),
            "space_after": round(space_after, 2),
            "background_color": background_color,
            # Add link placeholder; to be filled when intersecting a link rect
            "link": None,
        }
        
        # Only add alt_family_name if it exists and is different from font_family
        if alt_family_name and alt_family_name != font_name:
            text_detail["alt_family_name"] = alt_family_name
            
        return text_detail

    def _create_text_cell(self, index: int, text: str, bbox: BoundingBox, 
                          font_metadata: List[dict], is_word_level: bool = False) -> TextCell:
        """Create a TextCell with consistent metadata structure."""
        if is_word_level and font_metadata:
            # For word-level cells, we ensure all required fields are present
            # but we don't need to modify the metadata since it's already properly structured
            # by _create_font_metadata
            pass
        
        return TextCell(
            index=index,
            text=text,
            orig=text,
            rect=BoundingRectangle.from_bounding_box(bbox),
            from_ocr=False,
            font_metadata=font_metadata
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
        # Load links once
        links = self._load_links()

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
                for span_idx, span in enumerate(spans):
                    span_text = span.get("text", "")
                    if span_text:
                        meta = self._create_font_metadata(span, span_text.strip(), span_idx, line)
                        # Attach link information if span bbox overlaps a link
                        link_val = self._best_link_for_bbox(meta.get("bbox", []), links)
                        if link_val is not None:
                            meta["link"] = link_val
                        font_metadata.append(meta)

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
        links = self._load_links()

        for block in text_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_content = span.get("text", "").strip()
                    if text_content == "":
                        continue

                    bbox_tl = self._create_bounding_box(span.get("bbox", (0, 0, 0, 0)))

                    # For word cells, pass the span index and line for consistent metadata
                    meta = self._create_font_metadata(span, text_content, span_index % 10000, line)
                    # Attach link info to word-level metadata
                    link_val = self._best_link_for_bbox(meta.get("bbox", []), links)
                    if link_val is not None:
                        meta["link"] = link_val
                    font_metadata = [meta]

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
        
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'background_color': self._get_background_color()
        }

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

