from flask import Flask, request, jsonify, render_template
import pandas as pd
import math

app = Flask(__name__)

# Load the hospital data
hospital_data = pd.read_csv('Bangalore_Hospitals.csv')

def haversine(lat1, lon1, lat2, lon2):
    # Calculate distance between two lat/lon pairs using the Haversine formula
    R = 6371  # Radius of Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_hospitals', methods=['POST'])
def find_hospitals():
    user_location = request.json
    user_lat = user_location['latitude']
    user_lon = user_location['longitude']

    # Calculate distances
    hospital_data['Distance'] = hospital_data.apply(
        lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']),
        axis=1
    )
    # Get the 5 nearest hospitals
    nearest_hospitals = hospital_data.nsmallest(5, 'Distance')
    return jsonify(nearest_hospitals.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
