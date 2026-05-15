# app.py - Complete working version
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import re 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Try to download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'
#ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')

# ============================================
# Feature extraction functions
# ============================================

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

# ============================================
# URL Text Classifier
# ============================================

class URLTextClassifier:
    def __init__(self):
        self.url_model_loaded = False
        self.text_model_loaded = False
        self.url_pipeline = None
        self.text_model = None
        self.tfidf_vectorizer = None
        
        # Try to load existing models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists('models/url_model.pkl'):
                self.url_pipeline = joblib.load('models/url_model.pkl')
                self.url_model_loaded = True
                print("URL model loaded")
        except Exception as e:
            print(f"Could not load URL model: {e}")
        
        try:
            if os.path.exists('models/text_model.pkl'):
                model_data = joblib.load('models/text_model.pkl')
                self.text_model = model_data['classifier']
                self.tfidf_vectorizer = model_data['vectorizer']
                self.text_model_loaded = True
                print("Text model loaded")
        except Exception as e:
            print(f"Could not load text model: {e}")
    
    def detect_input_type(self, text):
        """Detect if input is URL or text"""
        text = text.strip().lower()
        if re.match(r'^https?://', text) or re.match(r'^www\.', text) or re.search(r'\.(com|org|net|edu|gov|io|co)', text):
            return 'url'
        return 'text'
    
    def classify(self, user_input):
        """Classify user input"""
        input_type = self.detect_input_type(user_input)
        
        if input_type == 'url':
            return self.classify_url(user_input)
        else:
            return self.classify_text(user_input)
    
    def classify_url(self, url):
        """Classify URL as benign or malicious"""
        if not self.url_model_loaded:
            return {
                'input_type': 'url',
                'url': url,
                'label': 'unknown',
                'confidence': 0.5,
                'message': 'URL model not trained yet. Please train the model in admin panel.'
            }
        
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
        """Classify text as ham or spam"""
        if not self.text_model_loaded:
            return {
                'input_type': 'text',
                'original_text': text,
                'label': 'unknown',
                'confidence': 0.5,
                'message': 'Text model not trained yet. Please train the model in admin panel.'
            }
        
        # Simple preprocessing
        processed_text = text.lower()
        X = self.tfidf_vectorizer.transform([processed_text]).toarray()
        prediction = self.text_model.predict(X)
        probabilities = self.text_model.predict_proba(X)
        
        return {
            'input_type': 'text',
            'original_text': text,
            'prediction': int(prediction[0]),
            'label': 'spam' if prediction[0] == 1 else 'ham',
            'confidence': float(max(probabilities[0]))
        }

# Initialize classifier
classifier = URLTextClassifier()

# ============================================
# Helper Functions
# ============================================

def create_sample_url_dataset():
    """Create sample URL dataset for training"""
    os.makedirs('datasets', exist_ok=True)
    
    sample_data = {
        'url': [
            'https://www.google.com', 'https://www.github.com', 'https://stackoverflow.com',
            'https://www.python.org', 'https://docs.python.org', 'https://www.microsoft.com',
            'https://www.amazon.com', 'https://www.youtube.com', 'https://www.facebook.com',
            'http://malicious-site.com/phishing', 'https://fake-bank-verify.com/login',
            'http://suspicious-download.net/virus.exe', 'https://paypal-verify-account.com',
            'http://free-bitcoin-generator.xyz', 'https://secure-verify-apple.com'
        ],
        'result': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('datasets/project_dataset.csv', index=False)
    print(" Sample URL dataset created")

def create_sample_text_dataset():
    """Create sample text dataset for training"""
    os.makedirs('datasets', exist_ok=True)
    
    sample_data = {
        'v1': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
        'v2': [
            'Hello, how are you?', 'Meeting at 3pm tomorrow',
            'Congratulations! You won $1000! Click here to claim',
            'See you at the party', 'URGENT: Your account has been compromised',
            'Thanks for your email', 'FREE iPhone giveaway! Click the link',
            'I will call you later', 'You have been selected for a prize',
            'The report is ready for review'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('datasets/spam.csv', index=False)
    print(" Sample text dataset created")

# Create datasets if they don't exist
if not os.path.exists('datasets/project_dataset.csv'):
    create_sample_url_dataset()
if not os.path.exists('datasets/spam.csv'):
    create_sample_text_dataset()

# ============================================
# Admin decorator
# ============================================

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            if request.is_json:
                return jsonify({'error': 'Admin access required'}), 401
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# Public Routes
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_input():
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        
        if not user_input:
            return jsonify({'error': 'Please enter some text or URL'}), 400
        
        result = classifier.classify(user_input)
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Please provide a list of texts/URLs'}), 400
        
        results = []
        for text in texts:
            if text.strip():
                result = classifier.classify(text.strip())
                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                results.append(result)
        
        stats = {
            'total': len(results),
            'malicious_spam': sum(1 for r in results if r.get('label') in ['malicious', 'spam']),
            'benign_ham': sum(1 for r in results if r.get('label') in ['benign', 'ham']),
            'urls': sum(1 for r in results if r.get('input_type') == 'url'),
            'texts': sum(1 for r in results if r.get('input_type') == 'text')
        }
        
        return jsonify({'results': results, 'statistics': stats})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'url_model_loaded': classifier.url_model_loaded,
        'text_model_loaded': classifier.text_model_loaded,
        'models_ready': classifier.url_model_loaded and classifier.text_model_loaded
    })

# ============================================
# Admin Routes
# ============================================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid credentials')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin.html')

@app.route('/admin/status', methods=['GET'])
@admin_required
def admin_status():
    return jsonify({
        'models_ready': classifier.url_model_loaded and classifier.text_model_loaded,
        'url_model': classifier.url_model_loaded,
        'text_model': classifier.text_model_loaded
    })

@app.route('/admin/train/url', methods=['POST'])
@admin_required
def train_url_model():
    """Train URL classification model"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'datasets/project_dataset.csv')
        
        print(f"\n🔗 Training URL model with dataset: {dataset_path}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            return jsonify({'success': False, 'error': f'Dataset not found at {dataset_path}'}), 400
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} rows")
        
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
            'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" F1 Score: {metrics['f1_score']:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, 'models/url_model.pkl')
        print(f" Model saved to models/url_model.pkl")
        
        # Update classifier
        classifier.url_pipeline = pipeline
        classifier.url_model_loaded = True
        
        # Log training
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'train_url_model',
            'metrics': metrics,
            'dataset': dataset_path
        }
        
        logs_file = 'models/training_logs.json'
        if os.path.exists(logs_file):
            with open(logs_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        with open(logs_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'URL model trained successfully!',
            'metrics': metrics
        })
    
    except Exception as e:
        print(f" Error training URL model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/train/text', methods=['POST'])
@admin_required
def train_text_model():
    """Train text classification model"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'datasets/spam.csv')
        
        print(f"\nTraining text model with dataset: {dataset_path}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            return jsonify({'success': False, 'error': f'Dataset not found at {dataset_path}'}), 400
        
        # Load dataset
        df = pd.read_csv(dataset_path, encoding='latin1')
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        
        # Encode labels
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        df.drop_duplicates(inplace=True)
        
        print(f"Loaded {len(df)} rows")
        
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
        print("Training Naive Bayes model...")
        text_model = MultinomialNB()
        text_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = text_model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred))
        }
        
        print(f" Accuracy: {metrics['accuracy']:.4f}")
        print(f" Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f" F1 Score: {metrics['f1_score']:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_data = {
            'classifier': text_model,
            'vectorizer': tfidf_vectorizer
        }
        joblib.dump(model_data, 'models/text_model.pkl')
        print(f"Model saved to models/text_model.pkl")
        
        # Update classifier
        classifier.text_model = text_model
        classifier.tfidf_vectorizer = tfidf_vectorizer
        classifier.text_model_loaded = True
        
        # Log training
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'train_text_model',
            'metrics': metrics,
            'dataset': dataset_path
        }
        
        logs_file = 'models/training_logs.json'
        if os.path.exists(logs_file):
            with open(logs_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        with open(logs_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Text model trained successfully!',
            'metrics': metrics
        })
    
    except Exception as e:
        print(f" Error training text model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/performance', methods=['GET'])
@admin_required
def get_performance():
    """Get model performance metrics"""
    try:
        url_metrics = {'error': 'Model not trained yet'}
        text_metrics = {'error': 'Model not trained yet'}
        
        # Evaluate URL model if exists
        if os.path.exists('models/url_model.pkl') and os.path.exists('datasets/project_dataset.csv'):
            try:
                model = joblib.load('models/url_model.pkl')
                df = pd.read_csv('datasets/project_dataset.csv')
                
                X = df[["url"]]
                y = df["result"]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                y_pred = model.predict(X_test)
                
                url_metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted')),
                    'recall': float(recall_score(y_test, y_pred, average='weighted')),
                    'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
                }
            except Exception as e:
                url_metrics = {'error': str(e)}
        
        # Evaluate text model if exists
        if os.path.exists('models/text_model.pkl') and os.path.exists('datasets/spam.csv'):
            try:
                model_data = joblib.load('models/text_model.pkl')
                model = model_data['classifier']
                vectorizer = model_data['vectorizer']
                
                df = pd.read_csv('datasets/spam.csv', encoding='latin1')
                df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
                df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
                
                le = LabelEncoder()
                y = le.fit_transform(df['target'])
                
                # Simple preprocessing
                X = vectorizer.transform(df['text'].str.lower()).toarray()
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                y_pred = model.predict(X_test)
                
                text_metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred)),
                    'recall': float(recall_score(y_test, y_pred)),
                    'f1_score': float(f1_score(y_test, y_pred))
                }
            except Exception as e:
                text_metrics = {'error': str(e)}
        
        return jsonify({
            'url_model': url_metrics,
            'text_model': text_metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/logs', methods=['GET'])
@admin_required
def get_logs():
    """Get training logs"""
    try:
        logs_file = 'models/training_logs.json'
        if os.path.exists(logs_file):
            with open(logs_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        return jsonify({'logs': logs})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/retrain', methods=['POST'])
@admin_required
def retrain_models():
    """Retrain both models"""
    try:
        # Retrain URL model
        url_result = train_url_model_internal()
        
        # Retrain text model
        text_result = train_text_model_internal()
        
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully',
            'url_metrics': url_result,
            'text_metrics': text_result
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def train_url_model_internal():
    """Internal function to train URL model"""
    dataset_path = 'datasets/project_dataset.csv'
    
    if not os.path.exists(dataset_path):
        create_sample_url_dataset()
    
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
    
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'models/url_model.pkl')
    
    classifier.url_pipeline = pipeline
    classifier.url_model_loaded = True
    
    return {'status': 'trained'}

def train_text_model_internal():
    """Internal function to train text model"""
    dataset_path = 'datasets/spam.csv'
    
    if not os.path.exists(dataset_path):
        create_sample_text_dataset()
    
    df = pd.read_csv(dataset_path, encoding='latin1')
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
    df.drop_duplicates(inplace=True)
    
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=1.0,
        min_df=2,
        max_features=4000
    )
    
    X = tfidf_vectorizer.fit_transform(df['text'].str.lower()).toarray()
    y = df['target'].values
    
    text_model = MultinomialNB()
    text_model.fit(X, y)
    
    model_data = {
        'classifier': text_model,
        'vectorizer': tfidf_vectorizer
    }
    joblib.dump(model_data, 'models/text_model.pkl')
    
    classifier.text_model = text_model
    classifier.tfidf_vectorizer = tfidf_vectorizer
    classifier.text_model_loaded = True
    
    return {'status': 'trained'}

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*60)
    print("URL & Text Classification System")
    print("="*60)
    print("\n Public Interface: http://127.0.0.1:5000")
    print(" Admin Interface: http://127.0.0.1:5000/admin/login")
    print(" Admin Credentials: admin / admin123")
    print("\n Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)