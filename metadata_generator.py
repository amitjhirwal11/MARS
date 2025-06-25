from datetime import datetime
import hashlib
import mimetypes
import logging

logger = logging.getLogger(__name__)

class MetadataGenerator:
    def generate_metadata(self, file_info, semantic_data):
        try:
            return {
                "document_id": self._generate_id(file_info['name']),
                "file_metadata": {
                    "original_name": file_info['name'],
                    "file_type": self._get_file_type(file_info['name']),
                    "file_size": f"{file_info['size']/1024:.2f} KB",
                    "content_type": self._get_mime_type(file_info['path'])
                },
                "content_metadata": {
                    "summary": semantic_data['summary'],
                    "keywords": semantic_data['keywords'],
                    "entities": semantic_data['entities'],
                    "topics": semantic_data['topics'],
                    "readability": semantic_data['readability'],
                    "sentiment": semantic_data['sentiment']
                },
                "system_metadata": {
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                    "metadata_version": "2.1"
                },
                "visualizations": semantic_data['visualizations'],
                "page_wise_analysis": semantic_data.get('page_wise_analysis', [])
            }
        except Exception as e:
            logger.error(f"Metadata generation failed: {str(e)}")
            raise
    
    def _generate_id(self, filename):
        return hashlib.sha3_256(filename.encode()).hexdigest()[:16]
    
    def _get_file_type(self, filename):
        ext = filename.split('.')[-1].upper()
        mapping = {
            'PDF': 'Document',
            'DOCX': 'Document',
            'PPTX': 'Presentation',
            'TXT': 'Text',
            'PNG': 'Image',
            'JPG': 'Image',
            'JPEG': 'Image'
        }
        return mapping.get(ext, 'Unknown')
    
    def _get_mime_type(self, file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
