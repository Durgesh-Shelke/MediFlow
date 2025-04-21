import pandas as pd
import numpy as np
import datetime
import uuid
import json
import os
from math import fabs
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load

class HospitalRecommendationAI:
    def __init__(self):
        self.urgency_classifier = None
        self.wait_time_predictor = None
        self.scaler = StandardScaler()
        self.urgency_levels = ["Low Priority", "Medium Priority", "High Priority", "Emergency"]
        
    def train_urgency_classifier(self, training_data):
        """
        Train the urgency classifier using historical patient data
        
        Parameters:
        training_data: DataFrame with columns for vitals and labeled urgency levels
        """
        # Extract features (vitals) and target (urgency)
        X = training_data[['Temp', 'Pulse', 'RespRate', 'SysBP', 'DiaBP', 'SpO2', 'PainScore']]
        y = training_data['UrgencyLevel']
        
        # Create and train the classifier
        self.urgency_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.urgency_classifier.fit(X, y)
        print("Urgency classifier trained successfully")
        
    def train_wait_time_predictor(self, historical_data):
        """
        Train a model to predict wait times based on hospital metrics
        
        Parameters:
        historical_data: DataFrame with hospital load, time of day, and actual wait times
        """
        # Extract features and target
        X = historical_data[['TotalBeds', 'EmergencyBeds', 'CurrentLoad', 'TimeOfDay', 'DayOfWeek']]
        y = historical_data['ActualWaitTime']
        
        # Create and train the predictor
        self.wait_time_predictor = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.wait_time_predictor.fit(X, y)
        print("Wait time predictor trained successfully")
        
    def predict_urgency(self, vitals):
        """
        Predict urgency level based on patient vitals
        
        Parameters:
        vitals: dict containing vital signs
        
        Returns:
        urgency_level: str representing the urgency classification
        confidence: float representing the confidence in the classification
        """
        if self.urgency_classifier is None:
            # Fallback to rule-based logic if model isn't trained
            return self._rule_based_urgency(vitals)
        
        # Extract features in the correct order
        features = np.array([[
            vitals['Temp'], 
            vitals['Pulse'], 
            vitals['RespRate'], 
            vitals['SysBP'], 
            vitals['DiaBP'], 
            vitals['SpO2'], 
            vitals['PainScore']
        ]])
        
        # Make prediction
        urgency_level = self.urgency_classifier.predict(features)[0]
        urgency_probs = self.urgency_classifier.predict_proba(features)[0]
        
        # Add confidence metric
        confidence = max(urgency_probs)
        
        return urgency_level, confidence
    
    def _rule_based_urgency(self, vitals):
        """Fallback method using rules for urgency classification"""
        # Extract vital signs
        temp = vitals['Temp']
        pulse = vitals['Pulse']
        resp_rate = vitals['RespRate']
        sys_bp = vitals['SysBP']
        dia_bp = vitals['DiaBP']
        spo2 = vitals['SpO2']
        pain_score = vitals['PainScore']
        
        # Emergency conditions
        if (temp > 102 or temp < 95 or 
            pulse > 120 or pulse < 50 or 
            resp_rate > 30 or resp_rate < 8 or 
            sys_bp < 90 or sys_bp > 180 or 
            dia_bp > 120 or 
            spo2 < 90 or 
            pain_score >= 9):
            return "Emergency", 0.9
        
        # High Priority conditions
        elif (temp > 100.5 or temp < 96 or 
              pulse > 110 or pulse < 55 or 
              resp_rate > 24 or resp_rate < 10 or 
              sys_bp < 100 or sys_bp > 160 or 
              dia_bp > 100 or 
              spo2 < 92 or 
              pain_score >= 7):
            return "High Priority", 0.8
        
        # Medium Priority conditions
        elif (temp > 99.5 or temp < 97 or 
              pulse > 100 or pulse < 60 or 
              resp_rate > 20 or resp_rate < 12 or 
              sys_bp < 110 or sys_bp > 140 or 
              dia_bp > 90 or 
              spo2 < 94 or 
              pain_score >= 5):
            return "Medium Priority", 0.7
        
        # Low Priority (normal vitals)
        else:
            return "Low Priority", 0.7

    def predict_wait_time(self, hospital_data, urgency_level, time_features=None):
        """
        Predict wait time based on hospital metrics and urgency
        
        Parameters:
        hospital_data: dict with hospital capacity information
        urgency_level: str representing patient urgency
        time_features: dict with time-related features (optional)
        
        Returns:
        estimated_wait: float representing estimated wait time in minutes
        """
        if self.wait_time_predictor is None or time_features is None:
            # Use heuristic approach if model isn't trained or no time features
            return self._heuristic_wait_time(hospital_data, urgency_level)
        
        # Prepare features for prediction
        features = np.array([[
            hospital_data['TotalBeds'],
            hospital_data['EmergencyBeds'],
            hospital_data['CurrentLoad'],
            time_features['TimeOfDay'],
            time_features['DayOfWeek']
        ]])
        
        # Apply scaling
        scaled_features = self.scaler.transform(features)
        
        # Make prediction and adjust based on urgency
        base_wait = self.wait_time_predictor.predict(scaled_features)[0]
        
        # Adjust wait time based on urgency level
        urgency_multipliers = {
            "Emergency": 0.0,  # No wait for emergencies
            "High Priority": 0.25,  # 25% of predicted wait
            "Medium Priority": 0.5,  # 50% of predicted wait
            "Low Priority": 1.0  # Full predicted wait
        }
        
        adjusted_wait = base_wait * urgency_multipliers.get(urgency_level, 1.0)
        return adjusted_wait
    
    def _heuristic_wait_time(self, hospital_data, urgency_level):
        """Fallback method using heuristics for wait time estimation"""
        total_beds = hospital_data['TotalBeds']
        emergency_beds = hospital_data['EmergencyBeds']
        
        # Calculate wait time based on hospital load
        general_load = max(0, total_beds - emergency_beds)
        
        # Adjust wait time based on urgency level
        if urgency_level == "Emergency":
            queue_time = 0  # Emergency patients get immediate attention
        elif urgency_level == "High Priority":
            queue_time = min(15, general_load * 15 / 4)  # 1/4 of the normal wait
        elif urgency_level == "Medium Priority":
            queue_time = min(30, general_load * 15 / 2)  # 1/2 of the normal wait
        else:  # Low Priority
            queue_time = min(60, general_load * 15)  # Full wait time
            
        return queue_time
    
    def manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two points"""
        return fabs(x1 - x2) + fabs(y1 - y2)
    
    def recommend_hospitals(self, patient_data, hospitals_df, patient_location):
        """
        Main function to recommend hospitals based on patient data
        
        Parameters:
        patient_data: dict containing patient information and vitals
        hospitals_df: DataFrame with hospital information
        patient_location: tuple (x, y) with patient coordinates
        
        Returns:
        top_hospitals: list of dicts with hospital recommendations
        urgency_level: str representing determined urgency level
        """
        # Extract patient vitals
        vitals = {
            'Temp': patient_data['Temp'],
            'Pulse': patient_data['Pulse'],
            'RespRate': patient_data['RespRate'],
            'SysBP': patient_data['SysBP'],
            'DiaBP': patient_data['DiaBP'],
            'SpO2': patient_data['SpO2'],
            'PainScore': patient_data['PainScore']
        }
        
        # Determine urgency level using AI model
        urgency_level, confidence = self.predict_urgency(vitals)
        
        # Get current time features
        now = datetime.datetime.now()
        time_features = {
            'TimeOfDay': now.hour + now.minute/60,
            'DayOfWeek': now.weekday()
        }
        
        # Calculate metrics for each hospital
        hospital_metrics = []
        x_patient, y_patient = patient_location
        
        for _, hospital in hospitals_df.iterrows():
            name = hospital['Name']
            total_beds = hospital['TotalBeds']
            emergency_beds = hospital['EmergencyBeds']
            x_hospital = hospital['X']
            y_hospital = hospital['Y']
            
            # Calculate distance and travel time
            distance = self.manhattan_distance(x_patient, y_patient, x_hospital, y_hospital)
            travel_time = distance  # Speed = 1 unit/min
            
            # Use AI to predict wait time
            hospital_data = {
                'TotalBeds': total_beds,
                'EmergencyBeds': emergency_beds,
                'CurrentLoad': total_beds - emergency_beds  # Simple load estimate
            }
            
            queue_time = self.predict_wait_time(hospital_data, urgency_level, time_features)
            total_delay = travel_time + queue_time
            
            # Check if hospital has capacity
            has_capacity = True
            if urgency_level == "Emergency" and emergency_beds <= 0:
                has_capacity = False
            elif total_beds <= 0:
                has_capacity = False
            
            if has_capacity:
                hospital_metrics.append({
                    'Hospital Name': name,
                    'Distance': distance,
                    'Travel Time': travel_time,
                    'Queue Time': queue_time,
                    'Total Delay': total_delay,
                    'UrgencyConfidence': confidence
                })
        
        # Rank hospitals using a weighted approach
        for hospital in hospital_metrics:
            # Create a composite score
            urgency_weight = 4 if urgency_level == "Emergency" else \
                            3 if urgency_level == "High Priority" else \
                            2 if urgency_level == "Medium Priority" else 1
                            
            # Weighted score: lower is better
            hospital['Score'] = (hospital['Total Delay'] / urgency_weight) + \
                              (10 * (1 - hospital['UrgencyConfidence']))  # Penalize low confidence
        
        # Sort hospitals by score (lower is better)
        hospital_metrics.sort(key=lambda x: x['Score'])
        
        # Get top 3 hospitals (or fewer if not enough available)
        top_hospitals = hospital_metrics[:min(3, len(hospital_metrics))]
        
        return top_hospitals, urgency_level
    
    def recommend_from_kiosk(self, patient_data, hospitals_df, kiosk_hospital_name):
        """
        Modified recommendation function for in-hospital kiosks
        
        Parameters:
        patient_data: dict containing patient information and vitals
        hospitals_df: DataFrame with hospital information
        kiosk_hospital_name: str name of the hospital where kiosk is located
        
        Returns:
        top_hospitals: list of dicts with hospital recommendations
        urgency_level: str representing determined urgency level
        """
        # Extract patient vitals
        vitals = {
            'Temp': patient_data['Temp'],
            'Pulse': patient_data['Pulse'],
            'RespRate': patient_data['RespRate'],
            'SysBP': patient_data['SysBP'],
            'DiaBP': patient_data['DiaBP'],
            'SpO2': patient_data['SpO2'],
            'PainScore': patient_data['PainScore']
        }
        
        # Determine urgency level using AI model
        urgency_level, confidence = self.predict_urgency(vitals)
        
        # Get current time features
        now = datetime.datetime.now()
        time_features = {
            'TimeOfDay': now.hour + now.minute/60,
            'DayOfWeek': now.weekday()
        }
        
        # Get kiosk hospital information
        kiosk_hospital = hospitals_df[hospitals_df['Name'] == kiosk_hospital_name].iloc[0]
        kiosk_x, kiosk_y = kiosk_hospital['X'], kiosk_hospital['Y']
        
        # Calculate metrics for each hospital
        hospital_metrics = []
        
        for _, hospital in hospitals_df.iterrows():
            name = hospital['Name']
            total_beds = hospital['TotalBeds']
            emergency_beds = hospital['EmergencyBeds']
            x_hospital = hospital['X']
            y_hospital = hospital['Y']
            
            # Calculate distance from current hospital
            distance = self.manhattan_distance(kiosk_x, kiosk_y, x_hospital, y_hospital)
            travel_time = distance  # Speed = 1 unit/min
            
            # If this is the current hospital, set travel time to 0
            if name == kiosk_hospital_name:
                distance = 0
                travel_time = 0
            
            # Use AI to predict wait time
            hospital_data = {
                'TotalBeds': total_beds,
                'EmergencyBeds': emergency_beds,
                'CurrentLoad': total_beds - emergency_beds  # Simple load estimate
            }
            
            queue_time = self.predict_wait_time(hospital_data, urgency_level, time_features)
            total_delay = travel_time + queue_time
            
            # Check if hospital has capacity
            has_capacity = True
            if urgency_level == "Emergency" and emergency_beds <= 0:
                has_capacity = False
            elif total_beds <= 0:
                has_capacity = False
            
            if has_capacity:
                hospital_metrics.append({
                    'Hospital Name': name,
                    'Distance': distance,
                    'Travel Time': travel_time,
                    'Queue Time': queue_time,
                    'Total Delay': total_delay,
                    'UrgencyConfidence': confidence,
                    'IsCurrentHospital': (name == kiosk_hospital_name)
                })
        
        # Rank hospitals - with strong preference for current hospital in non-emergency cases
        for hospital in hospital_metrics:
            # Create a composite score with bonus for current hospital
            urgency_weight = 4 if urgency_level == "Emergency" else \
                            3 if urgency_level == "High Priority" else \
                            2 if urgency_level == "Medium Priority" else 1
                            
            # Current hospital bonus (smaller for higher urgency cases)
            current_hospital_bonus = 0
            if hospital['IsCurrentHospital']:
                if urgency_level == "Emergency":
                    current_hospital_bonus = 0  # No bonus for emergencies - go to best hospital
                elif urgency_level == "High Priority":
                    current_hospital_bonus = 5  # Small bonus
                else:
                    current_hospital_bonus = 15  # Larger bonus for lower urgency
            
            # Weighted score: lower is better
            hospital['Score'] = (hospital['Total Delay'] / urgency_weight) - current_hospital_bonus + \
                              (10 * (1 - hospital['UrgencyConfidence']))  # Penalize low confidence
        
        # Sort hospitals by score (lower is better)
        hospital_metrics.sort(key=lambda x: x['Score'])
        
        # Get top 3 hospitals (or fewer if not enough available)
        top_hospitals = hospital_metrics[:min(3, len(hospital_metrics))]
        
        return top_hospitals, urgency_level
    
    def generate_token(self, patient_name, selected_hospital, urgency_level, estimated_wait_time):
        """Generate a token for the selected hospital"""
        token_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        token = {
            'TokenID': token_id,
            'PatientName': patient_name,
            'SelectedHospital': selected_hospital,
            'UrgencyLevel': urgency_level,
            'Timestamp': timestamp,
            'EstimatedWaitTime': estimated_wait_time
        }
        
        return token
    
    def save_token(self, token, filename='token_queue.csv'):
        """Save token to CSV file"""
        token_df = pd.DataFrame([token])
        
        try:
            # Check if file exists
            existing_df = pd.read_csv(filename)
            updated_df = pd.concat([existing_df, token_df], ignore_index=True)
            updated_df.to_csv(filename, index=False)
        except FileNotFoundError:
            # Create new file if it doesn't exist
            token_df.to_csv(filename, index=False)
    
    def save_models(self, path="models/"):
        """Save trained models to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.urgency_classifier is not None:
            dump(self.urgency_classifier, f"{path}urgency_classifier.joblib")
        
        if self.wait_time_predictor is not None:
            dump(self.wait_time_predictor, f"{path}wait_time_predictor.joblib")
            
        print(f"Models saved to {path}")
    
    def load_models(self, path="models/"):
        """Load trained models from disk"""
        try:
            self.urgency_classifier = load(f"{path}urgency_classifier.joblib")
            self.wait_time_predictor = load(f"{path}wait_time_predictor.joblib")
            print("Models loaded successfully")
            return True
        except FileNotFoundError:
            print("Models not found, using fallback methods")
            return False
    
    def generate_training_data(self, num_samples=1000):
        """
        Generate synthetic training data for model development
        Used for initial model training when historical data is limited
        """
        # Feature ranges based on medical standards
        temp_range = (95.0, 104.0)  # °F
        pulse_range = (40, 180)  # bpm
        resp_range = (6, 40)  # breaths/min
        sys_bp_range = (70, 200)  # mmHg
        dia_bp_range = (40, 130)  # mmHg
        spo2_range = (80, 100)  # %
        pain_range = (0, 10)  # scale
        
        synthetic_data = []
        
        for _ in range(num_samples):
            # Generate random vitals
            temp = np.random.uniform(*temp_range)
            pulse = np.random.uniform(*pulse_range)
            resp_rate = np.random.uniform(*resp_range)
            sys_bp = np.random.uniform(*sys_bp_range)
            dia_bp = np.random.uniform(*dia_bp_range)
            spo2 = np.random.uniform(*spo2_range)
            pain_score = np.random.randint(*pain_range)
            
            # Create vitals dict
            vitals = {
                'Temp': temp,
                'Pulse': pulse,
                'RespRate': resp_rate,
                'SysBP': sys_bp,
                'DiaBP': dia_bp,
                'SpO2': spo2,
                'PainScore': pain_score
            }
            
            # Get urgency using rule-based method
            urgency, _ = self._rule_based_urgency(vitals)
            
            # Create training example
            example = {
                **vitals,
                'UrgencyLevel': urgency
            }
            
            synthetic_data.append(example)
        
        # Convert to DataFrame
        df = pd.DataFrame(synthetic_data)
        
        # Option to save the synthetic data
        df.to_csv("synthetic_training_data.csv", index=False)
        
        return df


class KioskManager:
    def __init__(self, config_file="kiosk_config.json"):
        self.config_file = config_file
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "kiosk_id": "KIOSK_001",
                "hospital_name": "General Hospital",
                "location": {
                    "building": "Main",
                    "floor": "1",
                    "area": "Emergency Department"
                }
            }
            self.save_config()
    
    def save_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)
    
    def get_hospital_name(self):
        return self.config["hospital_name"]


# Flask application integration functions
def setup_hospital_ai():
    """Initialize and prepare the Hospital Recommendation AI"""
    hospital_ai = HospitalRecommendationAI()
    models_loaded = hospital_ai.load_models()
    
    if not models_loaded:
        # Generate synthetic data and train models
        training_data = hospital_ai.generate_training_data(num_samples=1000000)
        hospital_ai.train_urgency_classifier(training_data)
        hospital_ai.save_models()
    
    return hospital_ai