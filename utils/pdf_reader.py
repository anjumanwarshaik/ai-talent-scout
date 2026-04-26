# utils/pdf_reader.py
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None