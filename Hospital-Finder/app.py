from flask import Flask, request, jsonify, render_template
import pandas as pd
import math

app = Flask(__name__)

# Load the hospital data
hospital_data = pd.read_csv('Bangalore_Hospitals.csv')

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth (latitude and longitude) using the Haversine formula.
    """
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route('/')
def home():
    """
    Render the HTML page where users can input their location.
    """
    return render_template('index.html')

@app.route('/find_hospitals', methods=['POST'])
def find_hospitals():
    """
    API endpoint to find the 5 nearest hospitals based on user location.
    """
    try:
        # Get user's latitude and longitude from the frontend
        user_location = request.json
        user_lat = float(user_location.get('latitude'))
        user_lon = float(user_location.get('longitude'))

        # Calculate distances for all hospitals
        hospital_data['Distance'] = hospital_data.apply(
            lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']),
            axis=1
        )

        # Get the 5 nearest hospitals
        nearest_hospitals = hospital_data.nsmallest(5, 'Distance')[['Hospital_Name', 'Specialty', 'Rating', 'Latitude', 'Longitude', 'Distance']]

        # Convert the data to JSON format
        return jsonify(nearest_hospitals.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
