from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import folium
import requests
from geopy.geocoders import Nominatim
import random
import time

app = Flask(__name__)

# Global variables
API_KEY = '5b3ce3597851110001cf6248eb38bd3e8a4748269c41294631f48c0e'
ORS_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
df = None

# Load hospital data and preprocess
def load_hospital_data():
    hospitals = pd.read_csv('data.csv')
    locator = Nominatim(user_agent="myGeocoder")
    lat, lon = [], []

    def findloc(address):
        try:
            location = locator.geocode(address)
            time.sleep(1)  # To avoid hitting the geocoding API rate limit
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except Exception as e:
            print(f"Error geocoding address '{address}': {e}")
            return None, None

    # Geocode each hospital's address
    for _, row in hospitals.iterrows():
        latitude, longitude = findloc(row["address"])
        lat.append(latitude)
        lon.append(longitude)

    # Add latitude and longitude to the DataFrame
    hospitals['Latitude'] = lat
    hospitals['Longitude'] = lon

    # Drop rows where geocoding failed
    hospitals = hospitals.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    return hospitals

# Generate clustered ambulance data
def generate_clustered_ambulances(hospitals, num_ambulances=75):
    data = []
    for _ in range(num_ambulances):
        numberplate = f"KA-{random.randint(1, 99):02d}-{random.choice(['A', 'B', 'C', 'D'])}{random.randint(1000, 9999)}"
        hospital = hospitals.sample().iloc[0]
        lat, lon = hospital['Latitude'], hospital['Longitude']
        lat += random.uniform(-0.01, 0.01)
        lon += random.uniform(-0.01, 0.01)
        data.append({"Ambulance_Numberplate": numberplate, "Latitude": lat, "Longitude": lon})
    df = pd.DataFrame(data)

    ambulance_types = ['ALS', 'BLS', 'Non-Emergency']
    probabilities = [0.30, 0.50, 0.20]
    df['ambulance_type'] = np.random.choice(ambulance_types, size=len(df), p=probabilities)
    df['Ventilator'] = df['ambulance_type'].apply(lambda x: 1 if x == 'ALS' else 0)
    df['Oxygen'] = df['ambulance_type'].apply(lambda x: 1 if x in ['ALS', 'BLS'] else 0)
    df['Cardiac'] = df['ambulance_type'].apply(lambda x: 1 if x in ['ALS', 'BLS'] else 0)
    return df

# Compute score for ambulances
def gaussian(x, mu, sigma=1):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def compute_score(row, severity):
    score = 0
    centers = {'Non-Emergency': 2, 'BLS': 5.5, 'ALS': 8}
    sigma = 2

    if 1 <= severity <= 4:
        score += gaussian(severity, centers['Non-Emergency'], sigma)
        score -= gaussian(severity, centers['ALS'], sigma)
        score -= gaussian(severity, centers['BLS'], sigma)
    elif 4 < severity <= 7:
        score += gaussian(severity, centers['BLS'], sigma)
        score -= gaussian(severity, centers['Non-Emergency'], sigma)
        score -= gaussian(severity, centers['ALS'], sigma)
    elif 7 < severity <= 10:
        score += gaussian(severity, centers['ALS'], sigma)
        score -= gaussian(severity, centers['Non-Emergency'], sigma)
        score -= gaussian(severity, centers['BLS'], sigma)
    return score

def bonus_scoring(row, severity, cardiac, oxygen, ventilation):
    score = row['Score']
    if cardiac and row['Cardiac']:
        score += 1
    if oxygen and row['Oxygen']:
        score += 1
    if ventilation and row['Ventilator']:
        score += 1
    normalized_distance = 1 / (row['Distance'] + 1)
    score += normalized_distance * 5
    score += (1 / (row['ETA'] + 1)) * 5
    return score

def batch_eta(locations):
    payload = {"locations": locations, "metrics": ["duration"]}
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    response = requests.post(ORS_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('durations', [])
    else:
        return None

def score_generation(lat, lon, severity, cardiac, oxygen, ventilation):
    locations = df[['Longitude', 'Latitude']].values.tolist()
    origin = [float(lon), float(lat)]
    all_coordinates = [[loc, origin] for loc in locations]
    etas = []

    for batch in all_coordinates:
        duration_matrix = batch_eta(batch)
        if duration_matrix:
            etas.append(duration_matrix[0][1] // 60)
            time.sleep(1)
    df['ETA'] = etas

    distances = [geopy.distance.distance([lat, lon], (row.Latitude, row.Longitude)).km for row in df.itertuples()]
    df['Distance'] = distances

    df['Score'] = df.apply(lambda row: compute_score(row, int(severity)), axis=1)
    df['Score'] = df.apply(lambda row: bonus_scoring(row, int(severity), int(cardiac), int(oxygen), int(ventilation)), axis=1)
    best_ambulance = df[df['Score'] == df['Score'].max()]
    return best_ambulance

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_ambulance', methods=['POST'])
def get_ambulance():
    lat = request.form.get('latitude')
    lon = request.form.get('longitude')
    severity = request.form.get('severity')
    cardiac = request.form.get('cardiac')
    oxygen = request.form.get('oxygen')
    ventilation = request.form.get('ventilation')

    result = score_generation(lat, lon, severity, cardiac, oxygen, ventilation)
    return result.to_html(classes='table table-striped')

@app.route('/results')
def view_results():
    try:
        results = pd.read_csv('ambulance_result.csv')
        return results.to_html(classes='table table-striped')
    except Exception as e:
        return f"Error loading results: {e}"

# Initialize data before app starts
hospitals = load_hospital_data()
df = generate_clustered_ambulances(hospitals)

if __name__ == '__main__':
    app.run(debug=True)
