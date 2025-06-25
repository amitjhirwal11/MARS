import PyPDF2
import docx
import pytesseract
from pdf2image import convert_from_path
import os

class ContentExtractor:
    def extract_text(self, file_path, file_type):
        file_type = file_type.lower()
        try:
            if file_type == 'pdf':
                return self._extract_pdf(file_path)
            elif file_type == 'docx':
                return self._extract_docx(file_path)
            elif file_type == 'txt':
                return self._extract_txt(file_path)
            else:
                return self._ocr_extraction(file_path)
        except Exception as e:
            raise ValueError(f"Extraction failed: {str(e)}")

    def extract_page_wise_text(self, file_path, file_type):
        file_type = file_type.lower()
        try:
            if file_type == 'pdf':
                return self._extract_pdf_page_wise(file_path)
            else:
                raise ValueError("Page-wise extraction is only supported for PDFs.")
        except Exception as e:
            raise ValueError(f"Page-wise extraction failed: {str(e)}")

    def _extract_pdf(self, path):
        text = ""
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _extract_pdf_page_wise(self, path):
        pages_text = []
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                pages_text.append(page_text if page_text else "")
        return pages_text

    def _extract_docx(self, path):
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _extract_txt(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def _ocr_extraction(self, path):
        images = convert_from_path(path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
