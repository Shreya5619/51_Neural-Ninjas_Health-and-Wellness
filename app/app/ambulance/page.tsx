"use client";
import { useEffect, useState } from 'react';
import MapPage from './map/page';

const AmbulanceForm = () => {
  const [latitude, setLatitude] = useState(12.985332);
  const [longitude, setLongitude] = useState(77.9544343);
  const [latitude1, setLatitude1] = useState(12.985332);
  const [longitude1, setLongitude1] = useState(77.9544343);
  const [error, setError] = useState(null);
  const [severity, setSeverity] = useState<number>(0);
  const [cardiac, setCardiac] = useState<boolean>(false);
  const [oxygen, setOxygen] = useState<boolean>(false);
  const [ventilation, setVentilation] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();


    const requestData = {
      latitude: (latitude),
      longitude: (longitude),
      severity: severity,
      cardiac: cardiac ? 1 : 0,
      oxygen: oxygen ? 1 : 0,
      ventilation: ventilation ? 1 : 0,
    };

    fetch('http://localhost:5000/recommend_ambulance', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    })
    .then((response) => {
      console.log("Response:", response); // Logs the raw Response object
      console.log("Type of response:", typeof response); // Logs 'object', since the response is an object
      return response.json(); // Parse the JSON from the response
    })
    .then((data) => {
      console.log("Response data:", data); // Logs the parsed JSON array
      if (data.length > 0) {
        const firstAmbulance = data[0]; 
        setLatitude1(firstAmbulance.Latitude)
        setLongitude1(firstAmbulance.Longitude)// Access the first object in the array
        console.log("Latitude:", firstAmbulance.Latitude); // Logs the latitude
        console.log("Longitude:", firstAmbulance.Longitude); // Logs the longitude
      } else {
        console.warn("No ambulance data received.");
      }
  
      setResult(data); // Save the data to your state
    })
      .catch((error) => {
        console.error('Error or network issue:', error); // Handle any errors or network issues
      });
  };
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLatitude(position.coords.latitude);
          setLongitude(position.coords.longitude);
        },
        (err) => {
          setError(err.message);
        }
      );
    } else {
      setError("Geolocation is not supported by your browser.");
    }
  }, []);

  return (
    <div>
      <h2>Ambulance Recommendation</h2>
      <form onSubmit={handleSubmit}>
        <label>
          Latitude:
          <input
            type="number"
            step="any"
            value={latitude}
            onChange={(e) => setLatitude(parseFloat(e.target.value))}
            required
          />
        </label>
        <br />
        <label>
          Longitude:
          <input
            type="number"
            step="any"
            value={longitude}
            onChange={(e) => setLongitude(parseFloat(e.target.value))}
            required
          />
        </label>
        <br />
        <label>
          Severity (1-10):
          <input
            type="number"
            min="1"
            max="10"
            value={severity}
            onChange={(e) => setSeverity(Number(e.target.value))}
            required
          />
        </label>
        <br />
        <label>
          Cardiac:
          <input
            type="checkbox"
            checked={cardiac}
            onChange={() => setCardiac(!cardiac)}
          />
        </label>
        <br />
        <label>
          Oxygen:
          <input
            type="checkbox"
            checked={oxygen}
            onChange={() => setOxygen(!oxygen)}
          />
        </label>
        <br />
        <label>
          Ventilation:
          <input
            type="checkbox"
            checked={ventilation}
            onChange={() => setVentilation(!ventilation)}
          />
        </label>
        <br />
        <button type="submit">Get Ambulance</button>
      </form>

      {result && (
        <div>
          <h3>Recommended Ambulance</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
        
      )}

{result && (
        <div>
          <h3>Recommended Ambulance</h3>
          <pre><MapPage lat1={latitude1} lon1={longitude1} lat2={latitude} lon2={longitude}/></pre>
          getCurrentPosition
        </div>
)}
    </div>
  );
};

export default AmbulanceForm;
