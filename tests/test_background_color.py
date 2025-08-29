import os
import pytest
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from docling.backend.pymupdf_backend import PyMuPdfDocumentBackend
from docling.datamodel.document import InputDocument
from docling.datamodel.base_models import InputFormat
from docling.datamodel.base_models import Page


def test_background_color_extraction():
    """Test that we can extract background color from a PDF page."""
    # Create a simple test document
    test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    test_pdf_path = test_dir / "data" / "minimal.pdf"
    
    # Skip if test file doesn't exist
    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF file not found: {test_pdf_path}")
    
    # Create input document
    input_doc = InputDocument(
        input_format=InputFormat.PDF,
        file_path=str(test_pdf_path),
    )
    
    # Create backend
    backend = PyMuPdfDocumentBackend(input_doc, test_pdf_path)
    
    # Load first page
    page = backend.load_page(0)
    
    # Get background color
    bg_color = page._get_background_color()
    
    # Check that we got a color in hex format
    assert isinstance(bg_color, str)
    assert bg_color.startswith("#")
    assert len(bg_color) == 7  # #RRGGBB format
    
    # Get segmented page
    segmented_page = page.get_segmented_page()
    
    # Check if we can access the background color from the segmented page
    # The background color should be stored in the metadata dictionary
    assert hasattr(segmented_page, 'metadata'), "SegmentedPdfPage should have a metadata dictionary"
    assert 'background_color' in segmented_page.metadata, "metadata should contain background_color"
    assert segmented_page.metadata['background_color'] == bg_color, "Background color in metadata should match"
    
    # For backward compatibility, it should also be accessible directly
    assert hasattr(segmented_page, 'background_color'), "SegmentedPdfPage should have background_color property"
    assert segmented_page.background_color == bg_color, "Direct background_color property should match"
    
    # Check if the Page object also has the background color in its metadata
    page_obj = Page(page_no=0, parsed_page=segmented_page)
    assert 'background_color' in page_obj.metadata, "Page.metadata should contain background_color"
    assert page_obj.metadata['background_color'] == bg_color, "Page metadata background_color should match"
    
    # Check that the computed property works
    assert page_obj.background_color == bg_color, "Page.background_color property should match"
        
    # Clean up
    page.unload()
    backend.unload()

