"""
Fake News Detector - Model Architecture
Uses TF-IDF + Logistic Regression (fast, interpretable, works great for text)
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Download nltk stopwords
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class FakeNewsClassifier:
    def __init__(self):
        self.pipeline = None
        self.vectorizer = None

    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits (keep some punctuation)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _create_pipeline(self):
        """Create the ML pipeline"""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,        # Reduced for memory efficiency
                ngram_range=(1, 2),       # Unigrams + Bigrams
                min_df=5,                 # At least 5 documents
                max_df=0.85,              # Remove very common words
                sublinear_tf=True,        # Apply sublinear TF scaling
                strip_accents='unicode'
            )),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',  # Handle imbalanced data
                solver='lbfgs',
                n_jobs=-1,
                random_state=42
            ))
        ])
        return pipeline

    def build(self):
        """Build the pipeline"""
        self.pipeline = self._create_pipeline()
        return self

    def fit(self, X, y):
        """Train the model"""
        # Clean texts
        X_clean = [self._clean_text(text) for text in X]
        self.pipeline.fit(X_clean, y)
        return self

    def predict(self, X):
        """Predict labels"""
        X_clean = [self._clean_text(text) for text in X]
        return self.pipeline.predict(X_clean)

    def predict_proba(self, X):
        """Predict probabilities"""
        X_clean = [self._clean_text(text) for text in X]
        return self.pipeline.predict_proba(X_clean)

    def save(self, path='model/fake_news_model.pkl'):
        """Save model to disk"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    def load(self, path='model/fake_news_model.pkl'):
        """Load model from disk"""
        self.pipeline = joblib.load(path)
        return self