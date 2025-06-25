# MARS

### Project Overview

The system extracts content from documents, performs semantic analysis, and generates structured metadata including title, author, summary, keywords, and more. The final metadata is optionally compressed for efficient storage or transmission.

### Algorithms Used

- Gzip (gzip_compress.py)

Utilizes the gzip module to compress metadata with .gz output.
Gzip is a compression algorithm that uses the Deflate algorithm internally.

- Brotli (brotli_compress.py)

Uses Google’s Brotli algorithm for higher compression ratios.
Supports compression levels from 0 to 11

- Deflate (deflate_compress.py)

Implements the zlib library to apply Deflate compression.
Unlike Gzip, it doesn’t add headers/footers — just raw compressed data.

### Steps to Reproduce

1. Prepare Your Input: - Place your .pdf, .docx, or .txt files in the input_docs/ directory.
2. Run the Pipeline: - python main.py --file input_docs/sample.pdf --compress brotli
3. Check the Output: - uncompressed metadata: output/sample_metadata.json
4. Decompressing Compressed Output - using the algorithm gzip , brotli , deflate
5. Final Output - Support for different document types and compression standards

### Dependencies

PyPDF2, python-docx, nltk, spacy, gzip, brotli, zlib, json, argparse

### Use Cases

- Document management systems

- Knowledge indexing and search engines

- Academic paper processing

- Government/enterprise archives


### Future Enhancements

- Streamlit UI for drag-and-drop document uploads

- Support for scanned documents via OCR (Tesseract)

- Integration with document databases (e.g., MongoDB)

- Language detection and multilingual summarization

