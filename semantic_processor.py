import re
import logging
import numpy as np
from transformers import pipeline
from keybert import KeyBERT
import spacy
from textblob import TextBlob
import textstat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)

class SemanticProcessor:
    def __init__(self):
        logger.info("Loading summarization model...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.kw_model = KeyBERT()
        logger.info("Loading NER model...")
        self.ner = spacy.load("en_core_web_sm")
    
    def analyze_content(self, text):
        try:
            # Identify meaningful sections
            meaningful_sections = self.identify_meaningful_sections(text)
            
            # Extract summary from meaningful sections
            logger.info("Generating summary...")
            summary = self.summarizer(
                meaningful_sections, 
                max_length=150, 
                min_length=30, 
                do_sample=False
            )[0]['summary_text']
            
            # Extract keywords
            logger.info("Extracting keywords...")
            keywords = self.kw_model.extract_keywords(
                meaningful_sections,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=10
            )
            
            # Extract entities
            logger.info("Extracting entities...")
            entities = self.extract_entities(meaningful_sections)
            
            # Topic modeling
            logger.info("Performing topic modeling...")
            # For single document, n_docs=1
            topics = self.extract_topics(text, n_docs=1)
            
            # Readability and sentiment
            readability = textstat.flesch_reading_ease(text)
            sentiment = TextBlob(text).sentiment
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            wordcloud_img = self.generate_wordcloud(text)
            keyword_plot = self.generate_keyword_plot(keywords)
            
            return {
                "summary": summary,
                "keywords": [kw[0] for kw in keywords],
                "entities": entities,
                "topics": topics,
                "readability": readability,
                "sentiment": {
                    "polarity": sentiment.polarity,
                    "subjectivity": sentiment.subjectivity
                },
                "visualizations": {
                    "wordcloud": wordcloud_img,
                    "keyword_plot": keyword_plot
                }
            }
        except Exception as e:
            logger.error(f"Semantic analysis failed: {str(e)}")
            raise
    
    def analyze_page_wise(self, page_data):
        """Analyze each page individually for key content"""
        logger.info("Performing page-wise analysis")
        page_analysis = []
        
        for page in page_data:
            text = page['text']
            if not text.strip():
                continue
                
            # Extract page-specific features
            summary = self._summarize_page(text)
            keywords = self._extract_page_keywords(text)
            entities = self._extract_page_entities(text)
            
            page_analysis.append({
                'page_number': page['page_number'],
                'char_count': page.get('char_count', len(text)),
                'word_count': page.get('word_count', len(text.split())),
                'summary': summary,
                'keywords': keywords,
                'entities': entities
            })
        
        return page_analysis
    
    def identify_meaningful_sections(self, text):
        """Extract headings and high-content paragraphs using advanced heuristics"""
        # Extract headings (lines in ALL CAPS or Title Case)
        headings = re.findall(r'^([A-Z][A-Za-z\s]+[A-Za-z])$', text, re.MULTILINE)
        
        # Extract dense paragraphs (over 50 words)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.split()) > 50]
        
        # Extract bullet points
        bullet_points = re.findall(r'^\s*[\u2022\u2023\u25E6\u2043\u2219]\s*(.+)$', text, re.MULTILINE)
        
        return '\n\n'.join(headings + paragraphs + bullet_points)
    
    def extract_entities(self, text):
        doc = self.ner(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        return entities
    
    def _summarize_page(self, text):
        """Generate a concise summary for a single page"""
        try:
            summary = self.summarizer(
                text, 
                max_length=100, 
                min_length=20, 
                do_sample=False
            )[0]['summary_text']
            return summary
        except:
            return text[:150] + '...' if len(text) > 150 else text
    
    def _extract_page_keywords(self, text):
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=5
        )
        return [kw[0] for kw in keywords]
    
    def _extract_page_entities(self, text):
        doc = self.ner(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        return entities
    
    def extract_topics(self, text, n_docs=1, n_topics=3, n_words=5):
        """
        Safe topic modeling with automatic parameter adjustment to avoid
        min_df/max_df errors for single or multiple documents.
        """
        # Dynamically set min_df based on document count
        min_df_val = 1 if n_docs == 1 else 2
        
        vectorizer = TfidfVectorizer(
            max_df=1.0,        # Always allow terms in all docs
            min_df=min_df_val,  # Adjust based on doc count
            stop_words='english'
        )
        tfidf = vectorizer.fit_transform([text])
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=5,
            random_state=42
        )
        lda.fit(tfidf)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append({
                "topic_id": topic_idx + 1,
                "keywords": top_features
            })
        return topics

    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white',
                              collocations=False).generate(text)
        img = BytesIO()
        wordcloud.to_image().save(img, 'PNG')
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def generate_keyword_plot(self, keywords):
        words = [kw[0] for kw in keywords]
        scores = [kw[1] for kw in keywords]
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(words))
        plt.barh(y_pos, scores, color='skyblue')
        plt.yticks(y_pos, words)
        plt.xlabel('Relevance Score')
        plt.title('Top Keywords')
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')
