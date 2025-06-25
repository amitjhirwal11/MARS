import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import uuid
import time
import logging
from flask import Flask, render_template, request, jsonify
from extractors import ContentExtractor
from core.semantic_processor import SemanticProcessor
from core.metadata_generator import MetadataGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

extractor = ContentExtractor()
processor = SemanticProcessor()
generator = MetadataGenerator()

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'pptx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Supported: PDF, DOCX, TXT, PPTX, PNG, JPG"}), 400
    
    try:
        file_id = str(uuid.uuid4())
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{file_ext}")
        file.save(file_path)
        
        # Page wise ke liye
        if file_ext == 'pdf':
            page_wise_data = extractor.extract_page_wise_text(file_path)
            full_content = "\n\n".join([page['text'] for page in page_wise_data])
        else:
            full_content = extractor.extract_text(file_path, file_ext)
            page_wise_data = [{'page_number': 1, 'text': full_content}]
        
        # Semantics
        semantic_data = processor.analyze_content(full_content)
        
        # for merging pgw + seman
        semantic_data['page_wise_analysis'] = processor.analyze_page_wise(page_wise_data)
        
        
        file_info = {
            'name': file.filename,
            'type': file_ext,
            'size': os.path.getsize(file_path),
            'path': file_path
        }
        metadata = generator.generate_metadata(file_info, semantic_data)
        
        metadata['processing_metrics'] = {
            'extraction_time': f"{time.time() - start_time:.2f} seconds",
            'content_size': f"{len(full_content)} characters"
        }
        
        os.remove(file_path)
        
        return jsonify(metadata)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

