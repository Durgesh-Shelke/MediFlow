from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import csv
import datetime
import math
import pandas as pd

app = Flask(__name__)

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HOSPITALS_CSV = os.path.join(DATA_DIR, "hospitals.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

from model import HospitalRecommendationAI

# Initialize the AI system globally
hospital_ai = HospitalRecommendationAI()
hospital_ai.load_models()

@app.route('/recommend_hospital', methods=['POST'])
def recommend_hospital():
    # Get patient ID from form
    patient_id = request.form['patient_id']
    
    # Read patient data from CSV
    patients_df = pd.read_csv('patients.csv')
    patient_data = patients_df[patients_df['Name'] == patient_id].iloc[0]
    
    # Read hospital data
    hospitals_df = pd.read_csv('hospitals.csv')
    
    # Get current hospital name
    current_hospital = patient_data['Hospital name']
    
    # Get recommendations
    top_hospitals, urgency_level = hospital_ai.recommend_from_kiosk(
        patient_data=patient_data,
        hospitals_df=hospitals_df,
        kiosk_hospital_name=current_hospital
    )
    
    # Return recommendation page
    return render_template(
        'recommendations.html',
        patient=patient_data,
        hospitals=top_hospitals,
        urgency=urgency_level
    )

# Initialize CSV files if they don't exist
def initialize_csv_files():
    # Initialize hospitals.csv with headers if it doesn't exist
    if not os.path.exists(HOSPITALS_CSV):
        with open(HOSPITALS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'TotalBeds', 'EmergencyBeds', 'X', 'Y'])
    
    # Initialize patients.csv with headers if it doesn't exist
    if not os.path.exists(PATIENTS_CSV):
        with open(PATIENTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp','Name','Hospital name', 'Temp', 'Pulse', 'RespRate', 'SysBP', 'DiaBP', 'SpO2', 'PainScore'])

# Calculate Manhattan distance between two points
def manhattan_distance(x1, y1, x2, y2):
    return abs(float(x1) - float(x2)) + abs(float(y1) - float(y2))

# Update hospital distances
def update_hospital_distances():
    # Read all hospitals
    hospitals = []
    try:
        with open(HOSPITALS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Get headers
            base_headers = headers[:5]  # Name, TotalBeds, EmergencyBeds, X, Y
            
            for row in reader:
                hospitals.append(row)
    except FileNotFoundError:
        return False
    
    # Calculate distances between all hospitals
    distance_headers = []
    for i, hospital in enumerate(hospitals):
        hospital_name = hospital[0]
        distance_headers.append(f"Dist_to_{hospital_name}")
    
    # Create new headers with distance columns
    new_headers = base_headers + distance_headers
    
    # Calculate distances and create new rows
    new_data = []
    for i, hospital1 in enumerate(hospitals):
        new_row = hospital1[:5].copy()  # Base data
        
        # Add distance to each hospital
        for j, hospital2 in enumerate(hospitals):
            if i == j:  # Same hospital
                new_row.append("0")
            else:
                dist = manhattan_distance(
                    hospital1[3], hospital1[4],  # X1, Y1
                    hospital2[3], hospital2[4]   # X2, Y2
                )
                new_row.append(str(dist))
        
        new_data.append(new_row)
    
    # Write updated data back to CSV
    with open(HOSPITALS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(new_data)
    
    return True

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_hospital', methods=['GET', 'POST'])
def register_hospital():
    if request.method == 'POST':
        try:
            hospital_name = request.form['hospital_name']
            total_beds = request.form['total_beds']
            emergency_beds = request.form['emergency_beds']
            x_coord = request.form['x_coord']
            y_coord = request.form['y_coord']
            
            # Validate inputs
            if not hospital_name or not total_beds or not emergency_beds or not x_coord or not y_coord:
                return render_template('hospital.html', error="All fields are required")
            
            # Add hospital to CSV
            with open(HOSPITALS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([hospital_name, total_beds, emergency_beds, x_coord, y_coord])
            
            # Update distances
            update_hospital_distances()
            
            return render_template('hospital.html', success="Hospital registered and distances updated!")
        except Exception as e:
            return render_template('hospital.html', error=f"Error: {str(e)}")
    
    return render_template('hospital.html')

def load_hospital_names():
    with open(HOSPITALS_CSV, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row['Name'].strip().lower() for row in reader]
    
@app.route('/register_patient', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        hospital_name = request.form['hospital_name'].strip().lower()
        valid_hospitals = load_hospital_names()

        if hospital_name not in valid_hospitals:
            return render_template('patient.html', error="Hospital name not found in registry.")
        
        try:
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get form data
            name = request.form['name']
            hospital_name = request.form['hospital_name']
            temperature = request.form['temperature']
            pulse = request.form['pulse']
            resp_rate = request.form['resp_rate']
            sys_bp = request.form['sys_bp']
            dia_bp = request.form['dia_bp']
            spo2 = request.form['spo2']
            pain_score = request.form['pain_score']
            
            # Validate inputs
            if not all([name, hospital_name, temperature, pulse, resp_rate, sys_bp, dia_bp, spo2, pain_score]):
                return render_template('patient.html', error="All fields are required")
            
            # Add patient to CSV
            with open(PATIENTS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name, hospital_name, temperature, pulse, resp_rate, sys_bp, dia_bp, spo2, pain_score])
            
            return render_template('patient.html', success="Patient data recorded!")
        except Exception as e:
            return render_template('patient.html', error=f"Error: {str(e)}")
    
    return render_template('patient.html')

@app.route('/view_data')
def view_data():
    try:
        patients = []
        headers = []
        
        with open(PATIENTS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Get headers
            for row in reader:
                patients.append(row)
        
        # Get hospital data as well
        hospitals = []
        hospital_headers = []
        
        try:
            with open(HOSPITALS_CSV, 'r', newline='') as f:
                reader = csv.reader(f)
                hospital_headers = next(reader)  # Get headers
                for row in reader:
                    hospitals.append(row)
        except FileNotFoundError:
            hospitals = []
            hospital_headers = []
        
        return render_template('view_data.html', 
                               patients=patients, 
                               patient_headers=headers,
                               hospitals=hospitals,
                               hospital_headers=hospital_headers)
    except FileNotFoundError:
        return render_template('view_data.html', error="No data found")
    except Exception as e:
        return render_template('view_data.html', error=f"Error: {str(e)}")

@app.route('/api/patients')
def api_patients():
    try:
        patients = []
        with open(PATIENTS_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patients.append(row)
        return jsonify(patients)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/hospitals')
def api_hospitals():
    try:
        hospitals = []
        with open(HOSPITALS_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hospitals.append(row)
        return jsonify(hospitals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_csv_files()
    app.run(debug=True)