#!/usr/bin/env python3
"""
Advanced Diabetes Detection Model Trainer
Multiple ML algorithms with hyperparameter tuning and cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class DiabetesModelTrainer:
    """
    Advanced trainer for diabetes detection models with multiple algorithms
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        print("\n" + "="*70)
        print("🏥 ADVANCED DIABETES DETECTION MODEL TRAINER")
        print("="*70)
    
    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess diabetes dataset
        """
        print("\n📊 Loading dataset...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Separate features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """
        Split data and apply scaling
        """
        print("\n📈 Splitting and scaling data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Training set: {X_train_scaled.shape}")
        print(f"✅ Testing set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_svm_model(self, X_train, y_train, X_test, y_test):
        """
        Train SVM with hyperparameter tuning
        """
        print("\n🤖 Training SVM with hyperparameter tuning...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        svm = SVC(probability=True, random_state=self.random_state)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        self.models['SVM'] = best_svm
        
        y_pred = best_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Best SVM Accuracy: {accuracy:.4f}")
        print(f"   Best params: {grid_search.best_params_}")
        
        return best_svm, accuracy
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple ensemble models
        """
        results = {}
        
        # Random Forest
        print("\n🌲 Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        self.models['RandomForest'] = rf
        results['RandomForest'] = rf_acc
        print(f"✅ Random Forest Accuracy: {rf_acc:.4f}")
        
        # Gradient Boosting
        print("\n🚀 Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_acc = accuracy_score(y_test, gb_pred)
        self.models['GradientBoosting'] = gb
        results['GradientBoosting'] = gb_acc
        print(f"✅ Gradient Boosting Accuracy: {gb_acc:.4f}")
        
        # AdaBoost
        print("\n⚡ Training AdaBoost...")
        ab = AdaBoostClassifier(n_estimators=100, random_state=self.random_state)
        ab.fit(X_train, y_train)
        ab_pred = ab.predict(X_test)
        ab_acc = accuracy_score(y_test, ab_pred)
        self.models['AdaBoost'] = ab
        results['AdaBoost'] = ab_acc
        print(f"✅ AdaBoost Accuracy: {ab_acc:.4f}")
        
        return results
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression with hyperparameter tuning
        """
        print("\n📊 Training Logistic Regression...")
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        self.models['LogisticRegression'] = best_lr
        
        y_pred = best_lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Logistic Regression Accuracy: {accuracy:.4f}")
        
        return best_lr, accuracy
    
    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive evaluation of all models
        """
        print("\n" + "="*70)
        print("📈 MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            # Track best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
        
        return results
    
    def save_model(self, filename='best_diabetes_model.pkl'):
        """
        Save the best model
        """
        if self.best_model is not None:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"\n✅ Best model saved as {filename}")
        else:
            print("\n❌ No model trained yet!")


if __name__ == "__main__":
    # Example usage
    trainer = DiabetesModelTrainer()
    print("\n✅ Diabetes Model Trainer initialized successfully!")
    print("\nAvailable methods:")
    print("  - load_and_preprocess_data()")
    print("  - split_and_scale_data()")
    print("  - train_svm_model()")
    print("  - train_ensemble_models()")
    print("  - train_logistic_regression()")
    print("  - evaluate_models()")
    print("  - save_model()")
    print("\n" + "="*70)
