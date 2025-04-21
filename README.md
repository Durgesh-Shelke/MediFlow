# Hospital & Patient Registry Web Application

A full-stack web application that manages hospital and patient records with spatial distance calculations between hospitals.

## Features

- Register hospitals with location coordinates
- Automatically calculate Manhattan distances between all hospitals
- Record patient vital signs
- View and filter patient and hospital data
- Responsive design for various screen sizes

## Tech Stack

- **Backend:** Python 3.x with Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **UI Framework:** Bootstrap 5
- **Data Visualization:** DataTables for interactive tables
- **Data Storage:** CSV files

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hospital-registry.git
   cd hospital-registry
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the navigation sidebar to:
   - Register new hospitals
   - Record patient data
   - View all records

## Project Structure

```
hospital_registry/
│
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── index.html      # Main page with dashboard
│   ├── hospital.html   # Hospital registration form
│   ├── patient.html    # Patient registration form 
│   └── view_data.html  # Data viewing page
│
├── static/             # Static assets
│   ├── css/            # Stylesheets
│   │   └── main.css    # Main stylesheet
│   └── js/             # JavaScript files
│       └── main.js     # Main JavaScript file
│
├── data/               # Data storage
│   ├── hospitals.csv   # Hospital records with distances
│   └── patients.csv    # Patient records
│
└── README.md           # Project documentation
```

## Data Format

### hospitals.csv
Initial columns:
- Name (text)
- TotalBeds (integer)
- EmergencyBeds (integer)
- X (float) - X coordinate
- Y (float) - Y coordinate

Additional columns are dynamically added for distances:
- Dist_to_[HospitalName] (float) - Manhattan distance to other hospitals

### patients.csv
- Timestamp (datetime)
- Temp (float) - Temperature in °C
- Pulse (integer) - Pulse rate in bpm
- RespRate (integer) - Respiratory rate
- SysBP (integer) - Systolic blood pressure
- DiaBP (integer) - Diastolic blood pressure
- SpO2 (integer) - Oxygen saturation percentage
- PainScore (integer) - Pain score on a scale of 1-10

## Spatial Distance Calculation

The application uses Manhattan distance to calculate distances between hospitals:
```
Manhattan Distance = |X₁ - X₂| + |Y₁ - Y₂|
```

This calculation is performed automatically whenever a new hospital is registered.

## Dependencies

- Flask==2.2.3
- Bootstrap 5.3.0
- jQuery 3.6.0
- DataTables 1.11.5

## Error Handling

- Form validation for all inputs
- File existence checks for CSV data files
- Exception handling for data processing operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.