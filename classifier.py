# classifier.py
import re
import numpy as np
import pandas as pd
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Try to download NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

# Feature extraction patterns
ipv4_simple = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
ipv6_simple = r"\b[0-9A-Fa-f:]{2,}\b"
ip_pattern_simple = f"(?:{ipv4_simple}|{ipv6_simple})"

def add_url_features(X):
    """Feature extraction function for URLs"""
    df_temp = pd.DataFrame({"url": X})
    
    df_temp["https"] = df_temp["url"].str.contains("https").astype(int)
    df_temp["length"] = df_temp["url"].str.len()
    df_temp["num_digits"] = df_temp["url"].str.count(r"\d")
    df_temp["subdomain_count"] = df_temp["url"].str.count(r"\.") - 1
    df_temp["path_length"] = df_temp["url"].str.split("/", n=3).str[-1].str.len()
    df_temp["ip_in_url"] = df_temp["url"].str.contains(ip_pattern_simple, regex=True).astype(int)
    df_temp["dots"] = df_temp["url"].str.count(r"\.")
    df_temp["at_count"] = df_temp["url"].str.count("@")
    df_temp["question_count"] = df_temp["url"].str.count(r"\?")
    df_temp["hyphen_count"] = df_temp["url"].str.count("-")
    df_temp["equal_count"] = df_temp["url"].str.count("=")
    df_temp["hash_count"] = df_temp["url"].str.count("#")
    df_temp["percent_count"] = df_temp["url"].str.count("%")
    df_temp["plus_count"] = df_temp["url"].str.count(r"\+")
    df_temp["dollar_count"] = df_temp["url"].str.count(r"\$")
    df_temp["exclaim_count"] = df_temp["url"].str.count("!")
    df_temp["star_count"] = df_temp["url"].str.count(r"\*")
    df_temp["comma_count"] = df_temp["url"].str.count(",")
    df_temp["param_count"] = df_temp["url"].str.count("&")
    
    return df_temp.drop(columns=["url"])

class URLTextClassifier:
    """Unified classifier for both URLs and text messages"""
    
    def __init__(self, url_model_path='models/url_model.pkl', 
                 text_model_path='models/text_model.pkl', 
                 auto_train=True, url_dataset_path='datasets/project_dataset.csv',
                 force_retrain=False):  # Added force_retrain parameter
        
        self.url_model_loaded = False
        self.text_model_loaded = False
        self.url_pipeline = None
        self.text_model = None
        self.tfidf_vectorizer = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load URL model
        if not force_retrain:  # Only try to load if not forcing retrain
            try:
                if os.path.exists(url_model_path):
                    # Use a custom unpickler that can find the function
                    from joblib import load
                    self.url_pipeline = load(url_model_path)
                    self.url_model_loaded = True
                    print(f"✅ URL model loaded from {url_model_path}")
                else:
                    print(f"⚠️ URL model not found at {url_model_path}")
                    if auto_train:
                        self._train_url_model(url_dataset_path, url_model_path)
                        self.url_model_loaded = True
            except Exception as e:
                print(f"⚠️ Error loading URL model: {e}")
                if auto_train:
                    try:
                        self._train_url_model(url_dataset_path, url_model_path)
                        self.url_model_loaded = True
                    except Exception as e2:
                        print(f"❌ Failed to train URL model: {e2}")
        else:
            print("🔄 Force retraining URL model...")
            self._train_url_model(url_dataset_path, url_model_path)
            self.url_model_loaded = True
        
        # Try to load text model
        if not force_retrain:  # Only try to load if not forcing retrain
            try:
                if os.path.exists(text_model_path):
                    model_data = joblib.load(text_model_path)
                    self.text_model = model_data['classifier']
                    self.tfidf_vectorizer = model_data['vectorizer']
                    self.text_model_loaded = True
                    print(f"✅ Text model loaded from {text_model_path}")
                else:
                    print(f"⚠️ Text model not found at {text_model_path}")
                    if auto_train:
                        self._train_text_model(save_path=text_model_path)
                        self.text_model_loaded = True
            except Exception as e:
                print(f"⚠️ Error loading text model: {e}")
                if auto_train:
                    try:
                        self._train_text_model(save_path=text_model_path)
                        self.text_model_loaded = True
                    except Exception as e2:
                        print(f"❌ Failed to train text model: {e2}")
        else:
            print("🔄 Force retraining text model...")
            self._train_text_model(save_path=text_model_path)
            self.text_model_loaded = True
    
    def _train_url_model(self, dataset_path, model_path):
        """Train URL model and save it"""
        print("🔄 Training URL model...")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset not found at {dataset_path}")
            print("Creating a sample dataset for demonstration...")
            self._create_sample_url_dataset(dataset_path)
        
        # Load and train
        df = pd.read_csv(dataset_path)
        X = df[["url"]]
        y = df["result"]
        
        feature_transformer = FunctionTransformer(add_url_features, validate=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("tfidf", HashingVectorizer(
                    analyzer="char",
                    ngram_range=(3, 3),
                    n_features=400,
                    alternate_sign=False,
                    binary=True,
                    dtype=np.float32
                ), "url"),
                ("custom", feature_transformer, "url")
            ]
        )
        
        pipeline = Pipeline([
            ("features", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=42))
        ])
        
        # Train the model
        pipeline.fit(X, y)
        
        # Save the model
        joblib.dump(pipeline, model_path)
        print(f"✅ URL model trained and saved to {model_path}")
        
        # Verify the model was saved
        if os.path.exists(model_path):
            print(f"✅ Verification: Model file exists at {model_path} (Size: {os.path.getsize(model_path)} bytes)")
        else:
            print(f"❌ ERROR: Model file was not saved properly!")
        
        self.url_pipeline = pipeline
        return pipeline
    
    def _create_sample_url_dataset(self, dataset_path):
        """Create a sample dataset for demonstration"""
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        sample_data = {
            'url': [
                # Benign URLs
                'https://www.google.com',
                'https://www.github.com',
                'https://stackoverflow.com',
                'https://www.python.org',
                'https://docs.python.org',
                # Malicious URLs
                'http://malicious-site.com/phishing',
                'https://fake-bank-verify.com/login',
                'http://suspicious-download.net/virus.exe',
                'https://paypal-verify-account.com',
                'http://free-bitcoin-generator.xyz'
            ],
            'result': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(dataset_path, index=False)
        print(f"✅ Sample URL dataset created at {dataset_path}")
    
    def _train_text_model(self, save_path):
        """Train text model and save it"""
        print("🔄 Training text model...")
        
        dataset_path = 'datasets/spam.csv'
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset not found at {dataset_path}")
            print("Creating a sample dataset for demonstration...")
            self._create_sample_text_dataset(dataset_path)
        
        try:
            df = pd.read_csv(dataset_path, encoding='latin1')
        except:
            df = pd.read_csv(dataset_path, encoding='utf-8')
        
        # Handle different CSV formats
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
            df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        df.drop_duplicates(inplace=True)
        
        ps = PorterStemmer()
        
        def preprocess_text(text):
            try:
                text = text.lower()
                tokens = nltk.word_tokenize(text)
                tokens = [t for t in tokens if t.isalnum()]
                stop_words = set(stopwords.words('english'))
                tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
                tokens = [ps.stem(t) for t in tokens]
                return ' '.join(tokens)
            except:
                return text.lower()
        
        print("🔄 Preprocessing text...")
        df["transformed_text"] = df["text"].apply(preprocess_text)
        
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=2,
            max_features=4000
        )
        
        X = tfidf_vectorizer.fit_transform(df['transformed_text']).toarray()
        y = df['target'].values
        
        text_model = MultinomialNB()
        text_model.fit(X, y)
        
        # Save the model with vectorizer
        model_data = {
            'classifier': text_model,
            'vectorizer': tfidf_vectorizer
        }
        joblib.dump(model_data, save_path)
        print(f"✅ Text model trained and saved to {save_path}")
        
        # Verify the model was saved
        if os.path.exists(save_path):
            print(f"✅ Verification: Model file exists at {save_path} (Size: {os.path.getsize(save_path)} bytes)")
        else:
            print(f"❌ ERROR: Model file was not saved properly!")
        
        self.text_model = text_model
        self.tfidf_vectorizer = tfidf_vectorizer
        
        return text_model, tfidf_vectorizer
    
    def _create_sample_text_dataset(self, dataset_path):
        """Create a sample text dataset for demonstration"""
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        sample_data = {
            'v1': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
            'v2': [
                'Hello, how are you?',
                'Meeting at 3pm tomorrow',
                'Congratulations! You won $1000! Click here to claim',
                'See you at the party',
                'URGENT: Your account has been compromised, verify now',
                'Thanks for your email',
                'FREE iPhone giveaway! Click the link to get yours',
                'I will call you later',
                'You have been selected for a prize, send your details',
                'The report is ready for review'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(dataset_path, index=False)
        print(f"✅ Sample text dataset created at {dataset_path}")
    
    def detect_input_type(self, text):
        """Detect whether input is a URL or regular text message"""
        text = text.strip().lower()
        
        url_patterns = [
            r'^https?://',
            r'^www\.',
            r'\.com$',
            r'\.org$',
            r'\.net$',
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        ]
        
        for pattern in url_patterns:
            if re.search(pattern, text):
                return 'url'
        
        short_url_patterns = [r'bit\.ly/', r'tinyurl\.com/', r'goo\.gl/', r't\.co/']
        for pattern in short_url_patterns:
            if re.search(pattern, text):
                return 'url'
        
        return 'text'
    
    def preprocess_text(self, text):
        """Preprocess text message for classification"""
        try:
            ps = PorterStemmer()
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            tokens = [t for t in tokens if t.isalnum()]
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
            tokens = [ps.stem(t) for t in tokens]
            return ' '.join(tokens)
        except:
            return text.lower()
    
    def classify_url(self, url):
        """Classify a URL as benign or malicious"""
        if not self.url_model_loaded:
            raise ValueError("URL model not loaded")
        
        df_url = pd.DataFrame({'url': [url]})
        prediction = self.url_pipeline.predict(df_url)
        probabilities = self.url_pipeline.predict_proba(df_url)
        
        return {
            'input_type': 'url',
            'url': url,
            'prediction': int(prediction[0]),
            'label': 'malicious' if prediction[0] == 1 else 'benign',
            'confidence': float(max(probabilities[0]))
        }
    
    def classify_text(self, text):
        """Classify text message as ham or spam"""
        if not self.text_model_loaded:
            raise ValueError("Text model not loaded")
        
        processed_text = self.preprocess_text(text)
        X = self.tfidf_vectorizer.transform([processed_text]).toarray()
        prediction = self.text_model.predict(X)
        probabilities = self.text_model.predict_proba(X)
        
        return {
            'input_type': 'text',
            'original_text': text,
            'processed_text': processed_text,
            'prediction': int(prediction[0]),
            'label': 'spam' if prediction[0] == 1 else 'ham',
            'confidence': float(max(probabilities[0]))
        }
    
    def classify(self, user_input):
        """Main classification method"""
        input_type = self.detect_input_type(user_input)
        
        if input_type == 'url':
            return self.classify_url(user_input)
        else:
            return self.classify_text(user_input)
    
    def classify_batch(self, inputs):
        """Classify multiple inputs"""
        return [self.classify(user_input) for user_input in inputs]
    
    def retrain_all_models(self):
        """Force retrain and save both models"""
        print("🔄 Retraining all models from scratch...")
        self._train_url_model('datasets/project_dataset.csv', 'models/url_model.pkl')
        self._train_text_model('models/text_model.pkl')
        print("✅ All models retrained and saved successfully!")


# Example usage to test saving
if __name__ == "__main__":
    # Test the classifier
    print("=" * 50)
    print("Testing classifier with model saving...")
    print("=" * 50)
    
    # Force retrain to ensure models are saved
    classifier = URLTextClassifier(auto_train=True, force_retrain=True)
    
    # Test classification
    test_inputs = [
        "https://www.google.com",
        "Congratulations! You won a prize!",
        "Hello, how are you?"
    ]
    
    for input_text in test_inputs:
        result = classifier.classify(input_text)
        print(f"\nInput: {input_text}")
        print(f"Type: {result['input_type']}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}")