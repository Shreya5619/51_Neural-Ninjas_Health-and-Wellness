"use client";

import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet'; // Import Leaflet for custom icon classes
import { useEffect, useState } from 'react';

// Define the interface for props
interface MapPageProps {
  lat1: number;
  lon1: number;
  lat2: number;
  lon2: number;
}

const MapPage = ({ lat1, lon1, lat2, lon2 }: MapPageProps) => {
  const [map, setMap] = useState<any>(null);

  const position: [number, number] = [lat1, lon1]; // Current location from props
  const secondLocation: [number, number] = [lat2, lon2]; // Second location from props

  // Define Leaflet's default icon
  const location1Icon = new L.Icon({
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png', // Default Leaflet icon
    iconSize: [32, 32], // Icon size
    iconAnchor: [16, 32], // Anchor point
    popupAnchor: [0, -32], // Popup position relative to the icon
    className: 'leaflet-div-icon', // Custom class for CSS styling
  });

  const location2Icon = new L.Icon({
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png', // Default Leaflet icon
    iconSize: [32, 32], // Icon size
    iconAnchor: [16, 32], // Anchor point
    popupAnchor: [0, -32], // Popup position relative to the icon
    className: 'leaflet-div-icon', // Custom class for CSS styling
  });

  // Adjust map center when the coordinates change
  useEffect(() => {
    if (map) {
      map.setView(position, map.getZoom()); // Update map center
    }
  }, [lat1, lon1, map]); // Trigger when lat1 or lon1 changes

  return (
    <div style={{ height: '100vh' }}>
      <h2>Selected Ambulance</h2>
      {/* Initialize MapContainer once */}
      <MapContainer
        center={position}
        zoom={12}
        style={{ height: '100%', width: '100%' }}
        whenReady={(mapInstance: any) => setMap(mapInstance)} // Store map instance
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {/* Marker for the first location */}
        <Marker position={position} icon={location1Icon}>
          <Popup>Ambulance Location</Popup>
        </Marker>
        {/* Marker for the second location */}
        <Marker position={secondLocation} icon={location2Icon}>
          <Popup>My Location</Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};

export default MapPage;
