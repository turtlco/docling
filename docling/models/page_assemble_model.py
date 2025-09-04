import logging
import re
from collections.abc import Iterable
from typing import List, Optional, Dict, Any

import numpy as np
from pydantic import BaseModel

from docling.datamodel.base_models import (
    AssembledUnit,
    ContainerElement,
    FigureElement,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BasePageModel
from docling.models.layout_model import LayoutModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class PageAssembleOptions(BaseModel):
    pass


class PageAssembleModel(BasePageModel):
    def __init__(self, options: PageAssembleOptions):
        self.options = options

    def normalize_characters(self, text):
        """Normalize special characters in a text string."""
        normalized = text
        normalized = normalized.replace("⁄", "/")  # noqa: RUF001
        normalized = normalized.replace("'", "'")  # noqa: RUF001
        normalized = normalized.replace("'", "'")  # noqa: RUF001
        normalized = normalized.replace("”", '"')
        normalized = normalized.replace("“", '"')
        normalized = normalized.replace("•", "·")
        return normalized
        
    def sanitize_font_metadata(self, metadata_list):
        """
        Sanitize text in font metadata to match the main text processing.
        Also ensure a 'link' field is present on each entry (None if no hyperlink).

        Args:
            metadata_list: List of font metadata dictionaries
            
        Returns:
            List of sanitized font metadata dictionaries
        """
        if not metadata_list:
            return []
            
        sanitized_metadata = []
        
        for meta in metadata_list:
            # Create a copy to avoid modifying the original
            meta_copy = dict(meta) if isinstance(meta, dict) else {}

            # Get the text from metadata
            text = meta_copy.get("text", "")
            
            # Remove hyphen at end of text if present
            if isinstance(text, str) and text.endswith("-"):
                text = text[:-1]
                
            # Normalize characters
            if isinstance(text, str):
                text = self.normalize_characters(text)

            # Update the text in the metadata copy
            meta_copy["text"] = text
            
            # Ensure link key exists with the required shape
            link_val = meta_copy.get("link", None)
            if link_val is None or isinstance(link_val, (str, dict)):
                # Accept None, URL string, or dict for internal link
                pass
            else:
                # Any other type -> reset to None
                link_val = None
            meta_copy["link"] = link_val

            sanitized_metadata.append(meta_copy)
            
        return sanitized_metadata
        
    def extract_background_color(self, cells):
        """
        Extract background color from a cluster of cells.
        
        Args:
            cells: List of TextCell objects
            
        Returns:
            Background color as hex string or None if not found
        """
        if not cells:
            return None
            
        # Try to get background color from the first cell's font metadata
        for cell in cells:
            if hasattr(cell, 'font_metadata') and cell.font_metadata:
                for meta in cell.font_metadata:
                    # Check if there's a background color in the metadata
                    if 'background_color' in meta:
                        return meta['background_color']
                        
        # If no background color found in font metadata, try to get from page
        # This would require access to the page object, so we'll return None for now
        # and let the calling code handle it
        return None
        
    def sanitize_text(self, lines):
        """Process a list of text lines, handling hyphenation and normalizing characters."""
        if len(lines) <= 1:
            return " ".join(lines)

        for ix, line in enumerate(lines[1:]):
            prev_line = lines[ix]

            if prev_line.endswith("-"):
                prev_words = re.findall(r"\b[\w]+\b", prev_line)
                line_words = re.findall(r"\b[\w]+\b", line)

                if (
                    len(prev_words)
                    and len(line_words)
                    and prev_words[-1].isalnum()
                    and line_words[0].isalnum()
                ):
                    lines[ix] = prev_line[:-1]
            else:
                lines[ix] += " "

        sanitized_text = "".join(lines)
        
        # Apply character normalization
        sanitized_text = self.normalize_characters(sanitized_text)

        return sanitized_text.strip()  # Strip any leading or trailing whitespace

    def _overlap_area(self, a: List[float], b: List[float]) -> float:
        al, at, ar, ab = a
        bl, bt, br, bb = b
        ol = max(al, bl)
        ot = max(at, bt)
        or_ = min(ar, br)
        ob = min(ab, bb)
        return max(0.0, or_ - ol) * max(0.0, ob - ot)

    def _best_link_for_bbox(self, bbox: List[float], links: List[Dict[str, Any]]) -> Optional[Optional[Dict[str, Any]]]:
        if not links or not bbox or len(bbox) != 4:
            return None
        bl, bt, br, bb = bbox
        cx = (bl + br) / 2.0
        cy = (bt + bb) / 2.0
        # First pass: center-in-rect
        for lnk in links:
            rl, rt, rr, rb = lnk.get("rect", (0, 0, 0, 0))
            if rl <= cx <= rr and rt <= cy <= rb:
                if lnk.get("uri"):
                    return lnk["uri"]
                link_obj: Dict[str, Any] = {"type": "internal"}
                if lnk.get("page") is not None:
                    link_obj["page"] = int(lnk["page"]) + 1
                if lnk.get("point"):
                    link_obj["point"] = {"x": lnk["point"]["x"], "y": lnk["point"]["y"]}
                if lnk.get("zoom") is not None:
                    link_obj["zoom"] = lnk["zoom"]
                return link_obj
        # Second pass: max overlap ratio
        span_area = max(1e-6, (br - bl) * (bb - bt))
        best = None
        best_area = 0.0
        for lnk in links:
            rl, rt, rr, rb = lnk.get("rect", (0, 0, 0, 0))
            area = self._overlap_area([bl, bt, br, bb], [rl, rt, rr, rb])
            if area > best_area:
                best_area = area
                best = lnk
        if best and (best_area / span_area) >= 0.2:
            if best.get("uri"):
                return best["uri"]
            link_obj: Dict[str, Any] = {"type": "internal"}
            if best.get("page") is not None:
                link_obj["page"] = int(best["page"]) + 1
            if best.get("point"):
                link_obj["point"] = {"x": best["point"]["x"], "y": best["point"]["y"]}
            if best.get("zoom") is not None:
                link_obj["zoom"] = best["zoom"]
            return link_obj
        return None

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "page_assemble"):
                    assert page.predictions.layout is not None

                    # assembles some JSON output page by page.

                    elements: List[PageElement] = []
                    headers: List[PageElement] = []
                    body: List[PageElement] = []

                    # Preload page links for assigning to figures/images
                    page_links = []
                    try:
                        page_links = page._backend.get_links() or []
                    except Exception:
                        page_links = []

                    for cluster in page.predictions.layout.clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
                            textlines = [
                                cell.text.replace("\x02", "-").strip()
                                for cell in cluster.cells
                                if len(cell.text.strip()) > 0
                            ]
                            text = self.sanitize_text(textlines)
                            
                            # Collect and sanitize font_metadata from cells
                            font_metadata = []
                            for cell in cluster.cells:
                                if hasattr(cell, 'font_metadata') and cell.font_metadata:
                                    # Apply the same sanitization to font metadata text
                                    sanitized_metadata = self.sanitize_font_metadata(cell.font_metadata)
                                    font_metadata.extend(sanitized_metadata)
                            
                            # Extract background color from cells
                            background_color = self.extract_background_color(cluster.cells)
                                    
                            text_el = TextElement(
                                label=cluster.label,
                                id=cluster.id,
                                text=text,
                                page_no=page.page_no,
                                cluster=cluster,
                                font_metadata=font_metadata,
                                background_color=background_color,
                            )
                            elements.append(text_el)

                            if cluster.label in LayoutModel.PAGE_HEADER_LABELS:
                                headers.append(text_el)
                            else:
                                body.append(text_el)
                        elif cluster.label in LayoutModel.TABLE_LABELS:
                            tbl = None
                            if page.predictions.tablestructure:
                                tbl = page.predictions.tablestructure.table_map.get(
                                    cluster.id, None
                                )
                            if not tbl:  # fallback: add table without structure, if it isn't present
                                tbl = Table(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    otsl_seq=[],
                                    table_cells=[],
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )

                            elements.append(tbl)
                            body.append(tbl)
                        elif cluster.label == LayoutModel.FIGURE_LABEL:
                            fig = None
                            if page.predictions.figures_classification:
                                fig = page.predictions.figures_classification.figure_map.get(
                                    cluster.id, None
                                )
                            if not fig:  # fallback: add figure without classification, if it isn't present
                                fig = FigureElement(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    data=None,
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )
                            # Assign link to figure based on cluster bbox and page link rects
                            try:
                                # cluster.bbox is TOPLEFT origin; links rects are TOPLEFT as well
                                l, t, r, b = cluster.bbox.as_tuple()
                                link_val = self._best_link_for_bbox([l, t, r, b], page_links)
                                if link_val is not None:
                                    fig.link = link_val
                            except Exception:
                                pass
                            elements.append(fig)
                            body.append(fig)
                        elif cluster.label in LayoutModel.CONTAINER_LABELS:
                            container_el = ContainerElement(
                                label=cluster.label,
                                id=cluster.id,
                                page_no=page.page_no,
                                cluster=cluster,
                            )
                            elements.append(container_el)
                            body.append(container_el)

                    page.assembled = AssembledUnit(
                        elements=elements, headers=headers, body=body
                    )

                yield page
