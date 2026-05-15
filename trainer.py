# trainer.py
import pandas as pd
import numpy as np
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

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

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def train_url_model(self, dataset_path, model_path='models/url_model.pkl'):
        """Train URL classification model"""
        print("\n Training URL Classification Model...")
        
        # Load dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        print(f" Loaded {len(df)} rows from {dataset_path}")
        
        # Prepare data
        X = df[["url"]]
        y = df["result"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create feature transformer
        feature_transformer = FunctionTransformer(add_url_features, validate=False)
        
        # Create preprocessor
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
        
        # Create pipeline
        pipeline = Pipeline([
            ("features", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=42))
        ])
        
        # Train model
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" F1 Score: {metrics['f1_score']:.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f" Model saved to {model_path}")
        
        return {
            'model': pipeline,
            'metrics': metrics,
            'test_size': len(y_test)
        }
    
    def train_text_model(self, dataset_path, model_path='models/text_model.pkl'):
        """Train text classification model"""
        print("\nTraining Text Classification Model...")
        
        # Load dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path, encoding='latin1')
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        
        # Encode labels
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        df.drop_duplicates(inplace=True)
        
        print(f"Loaded {len(df)} rows from {dataset_path}")
        
        # Preprocess function
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
        
        # Apply preprocessing
        print("Preprocessing text...")
        df["transformed_text"] = df["text"].apply(preprocess_text)
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=2,
            max_features=4000
        )
        
        # Transform data
        X = tfidf_vectorizer.fit_transform(df['transformed_text']).toarray()
        y = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print(" Training Naive Bayes model...")
        text_model = MultinomialNB()
        text_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = text_model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" F1 Score: {metrics['f1_score']:.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data = {
            'classifier': text_model,
            'vectorizer': tfidf_vectorizer
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        return {
            'model': text_model,
            'vectorizer': tfidf_vectorizer,
            'metrics': metrics,
            'test_size': len(y_test)
        }
    
    def create_sample_url_dataset(self, dataset_path='datasets/project_dataset.csv'):
        """Create a sample URL dataset for testing"""
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        sample_data = {
            'url': [
                # Benign URLs
                'https://www.google.com',
                'https://www.github.com',
                'https://stackoverflow.com',
                'https://www.python.org',
                'https://docs.python.org',
                'https://www.microsoft.com',
                'https://www.amazon.com',
                'https://www.youtube.com',
                'https://www.facebook.com',
                'https://www.twitter.com',
                # Malicious URLs
                'http://malicious-site.com/phishing',
                'https://fake-bank-verify.com/login',
                'http://suspicious-download.net/virus.exe',
                'https://paypal-verify-account.com',
                'http://free-bitcoin-generator.xyz',
                'https://secure-verify-apple.com',
                'http://update-windows-defender.com',
                'https://confirm-account-instagram.com',
                'http://claim-your-prize-now.net',
                'https://verify-payment-info.com'
            ],
            'result': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(dataset_path, index=False)
        print(f"Sample URL dataset created at {dataset_path}")
        return dataset_path
    
    def create_sample_text_dataset(self, dataset_path='datasets/spam.csv'):
        """Create a sample text dataset for testing"""
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        sample_data = {
            'v1': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 
                   'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
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
                'The report is ready for review',
                'WINNER! You have been selected for a free vacation',
                'Can we reschedule the meeting?',
                'Your Netflix subscription has expired, update now',
                'Looking forward to the event',
                'Bank account verification required immediately',
                'The project is on track',
                'Claim your $500 Amazon gift card now',
                'Let me know your thoughts',
                'Security alert: Unusual login detected',
                'Please find attached the document'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(dataset_path, index=False)
        print(f"Sample text dataset created at {dataset_path}")
        return dataset_path