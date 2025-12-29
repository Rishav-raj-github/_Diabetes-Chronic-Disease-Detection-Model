#!/usr/bin/env python3
"""
Diabetes Prediction Module - Real-time predictions with interpretability
"""

import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    """
    Advanced predictor for diabetes detection with confidence scores and risk assessment
    """
    
    def __init__(self, model_path='best_diabetes_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.risk_thresholds = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
            print("\n🏥 DIABETES PREDICTOR INITIALIZED")
            print(f"Model loaded from: {model_path}")
        except FileNotFoundError:
            print(f"\u274c Model file not found: {model_path}")
            self.model = None
    
    def predict_single(self, features):
        """
        Predict diabetes for a single patient
        features: list or array of patient medical indicators
        """
        if self.model is None:
            return None
        
        features_array = np.array(features).reshape(1, -1)
        
        try:
            prediction = self.model.predict(features_array)[0]
            probability = max(self.model.predict_proba(features_array)[0])
            
            result = {
                'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                'confidence': float(probability),
                'risk_level': self._assess_risk(probability),
                'timestamp': datetime.now().isoformat()
            }
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, features_list):
        """
        Batch prediction for multiple patients
        """
        if self.model is None:
            return None
        
        results = []
        for i, features in enumerate(features_list):
            result = self.predict_single(features)
            result['patient_id'] = i + 1
            results.append(result)
        
        return results
    
    def _assess_risk(self, confidence):
        """
        Assess risk level based on confidence score
        """
        if confidence < self.risk_thresholds['low']:
            return 'Very Low Risk'
        elif confidence < self.risk_thresholds['medium']:
            return 'Low Risk'
        elif confidence < self.risk_thresholds['high']:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def get_feature_importance(self):
        """
        Get feature importance if available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_.tolist()
        return None
    
    def generate_report(self, features, feature_names=None):
        """
        Generate comprehensive prediction report
        """
        prediction = self.predict_single(features)
        
        if not feature_names:
            feature_names = [f'Feature_{i+1}' for i in range(len(features))]
        
        report = {
            'prediction': prediction,
            'feature_values': dict(zip(feature_names, features)),
            'model_type': type(self.model).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def export_predictions(self, results, filename='predictions.json'):
        """
        Export predictions to JSON file
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Predictions exported to {filename}")


class RiskAssessment:
    """
    Advanced risk assessment and recommendations
    """
    
    @staticmethod
    def get_recommendations(prediction_result):
        """
        Get personalized recommendations based on prediction
        """
        risk_level = prediction_result.get('risk_level', 'Unknown')
        
        recommendations = {
            'Very Low Risk': [
                'Continue regular health checkups',
                'Maintain healthy lifestyle',
                'Regular exercise (30 mins daily)',
                'Balanced diet'
            ],
            'Low Risk': [
                'Increase monitoring frequency',
                'Lifestyle modifications recommended',
                'Regular exercise program',
                'Dietary changes for diabetes prevention'
            ],
            'Medium Risk': [
                'Consult healthcare professional',
                'Diabetes screening recommended',
                'Implement lifestyle changes immediately',
                'Regular blood glucose monitoring'
            ],
            'High Risk': [
                'URGENT: Consult endocrinologist',
                'Immediate diabetes screening required',
                'Medical intervention recommended',
                'Hospitalization may be needed',
                'Continuous monitoring essential'
            ]
        }
        
        return recommendations.get(risk_level, [])


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏥 DIABETES PREDICTOR - READY FOR USE")
    print("="*70)
    print("\nAvailable components:")
    print("  - DiabetesPredictor: Single/batch predictions")
    print("  - RiskAssessment: Personalized recommendations")
    print("  - Feature importance analysis")
    print("  - Comprehensive reporting")
    print("\n" + "="*70)
