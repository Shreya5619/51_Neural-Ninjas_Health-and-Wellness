import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import geopy.distance
import requests
import time

app = Flask(__name__)

# Global configurations
API_KEY = '5b3ce3597851110001cf6248eb38bd3e8a4748269c41294631f48c0e'
ORS_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"

def gaussian(x, mu, sigma=1):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def compute_score(row, severity):
    score = 0
    centers = {'Non-Emergency': 2, 'BLS': 5.5, 'ALS': 8}
    sigma = 2
    if severity >= 1 and severity <= 4:
        score += gaussian(severity, centers['Non-Emergency'], sigma)
        score -= gaussian(severity, centers['ALS'], sigma)
        score -= gaussian(severity, centers['BLS'], sigma)
    elif severity >= 4 and severity <= 7:
        score += gaussian(severity, centers['BLS'], sigma)
        score -= gaussian(severity, centers['Non-Emergency'], sigma)
        score -= gaussian(severity, centers['ALS'], sigma)
    elif severity >= 7 and severity <= 10:
        score += gaussian(severity, centers['ALS'], sigma)
        score -= gaussian(severity, centers['Non-Emergency'], sigma)
        score -= gaussian(severity, centers['BLS'], sigma)
    return score

def bonus_scoring(row, severity, cardiac, oxygen, ventilation):
    score = row['Score']
    if cardiac and row['Cardiac'] == 1:
        score += 1
    if oxygen and row['Oxygen'] == 1:
        score += 1
    if ventilation and row['Ventilator'] == 1:
        score += 1
    normalized_distance = 1 / (row['Distance'] + 1)
    score += normalized_distance * 5
    score += (1 / (row['ETA'] + 1)) * 5
    return score

def batch_eta(locations, API_KEY, url):
    payload = {"locations": locations, "metrics": ["duration"]}
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get('durations', [])
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def score_generation(lat, lon, severity, cardiac, oxygen, ventilation):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(BASE_DIR, 'ambulance.csv')
    df = pd.read_csv(csv_file_path)
    locations = df[['Longitude', 'Latitude']].values.tolist()
    origin = [float(lon), float(lat)]
    all_coordinates = [[loc, origin] for loc in locations]
    etas = []
    for batch in all_coordinates:
        duration_matrix = batch_eta(batch, API_KEY, ORS_URL)
        if duration_matrix:
            print(f"Duration matrix for batch {batch}: {duration_matrix}")
            etas.append(duration_matrix[0][1] // 60)
    df['ETA'] = etas
    distance_km = [
        geopy.distance.distance((lat, lon), (row.Latitude, row.Longitude)).km
        for row in df.itertuples(index=False)
    ]
    df['Distance'] = distance_km
    df['Score'] = df.apply(lambda row: compute_score(row, severity), axis=1)
    df['Score'] = df.apply(lambda row: bonus_scoring(row, severity, cardiac, oxygen, ventilation), axis=1)
    return df[df['Score'] == df['Score'].max()]

@app.route('/recommend_ambulance', methods=['POST'])
def recommend_ambulance():
    try:
        data = request.json
        lat = data['latitude']
        lon = data['longitude']
        severity = int(data['severity'])
        cardiac = data.get('cardiac', 0)
        oxygen = data.get('oxygen', 0)
        ventilation = data.get('ventilation', 0)
        result = score_generation(lat, lon, severity, cardiac, oxygen, ventilation)
        return jsonify(result.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
