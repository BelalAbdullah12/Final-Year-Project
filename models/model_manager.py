# models/model_manager.py
import os
import json
import shutil
from datetime import datetime
import joblib

class ModelManager:
    """Manages model operations: backup, restore, logging"""
    
    def __init__(self, model_dir='models', backup_dir='models/backups', log_file='models/training_logs.json'):
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        self.log_file = log_file
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)
    
    def backup_models(self):
        """Backup current models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f'backup_{timestamp}')
        os.makedirs(backup_path, exist_ok=True)
        
        # Backup URL model
        url_model_path = os.path.join(self.model_dir, 'url_model.pkl')
        if os.path.exists(url_model_path):
            shutil.copy(url_model_path, os.path.join(backup_path, 'url_model.pkl'))
        
        # Backup text model
        text_model_path = os.path.join(self.model_dir, 'text_model.pkl')
        if os.path.exists(text_model_path):
            shutil.copy(text_model_path, os.path.join(backup_path, 'text_model.pkl'))
        
        # Backup log file
        if os.path.exists(self.log_file):
            shutil.copy(self.log_file, os.path.join(backup_path, 'training_logs.json'))
        
        return backup_path
    
    def restore_models(self, backup_timestamp):
        """Restore models from backup"""
        backup_path = os.path.join(self.backup_dir, f'backup_{backup_timestamp}')
        
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup {backup_timestamp} not found")
        
        # Restore URL model
        backup_url = os.path.join(backup_path, 'url_model.pkl')
        if os.path.exists(backup_url):
            shutil.copy(backup_url, os.path.join(self.model_dir, 'url_model.pkl'))
        
        # Restore text model
        backup_text = os.path.join(backup_path, 'text_model.pkl')
        if os.path.exists(backup_text):
            shutil.copy(backup_text, os.path.join(self.model_dir, 'text_model.pkl'))
        
        return True
    
    def log_training(self, log_entry):
        """Log training activity"""
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_training_logs(self, limit=50):
        """Get training logs"""
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        return logs[-limit:]
    
    def get_model_info(self):
        """Get information about current models"""
        info = {
            'url_model': {
                'exists': False,
                'size_mb': 0,
                'modified': None
            },
            'text_model': {
                'exists': False,
                'size_mb': 0,
                'modified': None
            }
        }
        
        url_path = os.path.join(self.model_dir, 'url_model.pkl')
        if os.path.exists(url_path):
            info['url_model']['exists'] = True
            info['url_model']['size_mb'] = round(os.path.getsize(url_path) / (1024 * 1024), 2)
            info['url_model']['modified'] = datetime.fromtimestamp(
                os.path.getmtime(url_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        
        text_path = os.path.join(self.model_dir, 'text_model.pkl')
        if os.path.exists(text_path):
            info['text_model']['exists'] = True
            info['text_model']['size_mb'] = round(os.path.getsize(text_path) / (1024 * 1024), 2)
            info['text_model']['modified'] = datetime.fromtimestamp(
                os.path.getmtime(text_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        
        return info