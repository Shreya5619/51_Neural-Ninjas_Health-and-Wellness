"use client";

import { useState, useEffect } from "react";
import { Container, Row, Col, Form, Button, Card, Alert, Spinner } from "react-bootstrap";
import RecommendedAmbulance from "./recommendAmbulance";

const AmbulanceForm = () => {
  const [latitude, setLatitude] = useState(12.985332);
  const [longitude, setLongitude] = useState(77.9544343);
  const [severity, setSeverity] = useState<number>(0);
  const [cardiac, setCardiac] = useState<boolean>(false);
  const [oxygen, setOxygen] = useState<boolean>(false);
  const [ventilation, setVentilation] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const requestData = {
      latitude,
      longitude,
      severity,
      cardiac: cardiac ? 1 : 0,
      oxygen: oxygen ? 1 : 0,
      ventilation: ventilation ? 1 : 0,
    };

    fetch("http://localhost:5000/recommend_ambulance", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        setResult(data);
      })
      .catch((err) => {
        setLoading(false);
        setError("Failed to fetch ambulance recommendations. Please try again.");
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
          setError("Could not retrieve your location.");
        }
      );
    } else {
      setError("Geolocation is not supported by your browser.");
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-r from-blue-400 via-teal-300 to-green-300 p-5 flex items-center">
      <Container>
        <Row className="justify-content-center">
          <Col md={8}>
            <Card className="shadow-lg">
              <Card.Body>
                <h2 className="text-center text-3xl font-bold mb-4 text-blue-700">
                  🚑 Ambulance Recommendation
                </h2>
                <Form onSubmit={handleSubmit}>
                  <Row>
                    <Col md={6}>
                      <Form.Group controlId="latitude" className="mb-3">
                        <Form.Label>Latitude</Form.Label>
                        <Form.Control
                          type="number"
                          step="any"
                          value={latitude}
                          onChange={(e) => setLatitude(parseFloat(e.target.value))}
                          required
                        />
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group controlId="longitude" className="mb-3">
                        <Form.Label>Longitude</Form.Label>
                        <Form.Control
                          type="number"
                          step="any"
                          value={longitude}
                          onChange={(e) => setLongitude(parseFloat(e.target.value))}
                          required
                        />
                      </Form.Group>
                    </Col>
                  </Row>
                  <Form.Group controlId="severity" className="mb-3">
                    <Form.Label>Severity (1-10)</Form.Label>
                    <Form.Control
                      type="number"
                      min="1"
                      max="10"
                      value={severity}
                      onChange={(e) => setSeverity(Number(e.target.value))}
                      required
                    />
                  </Form.Group>
                  <Row>
                    <Col md={4}>
                      <Form.Check
                        type="checkbox"
                        id="cardiac"
                        label="Cardiac Support"
                        checked={cardiac}
                        onChange={() => setCardiac(!cardiac)}
                      />
                    </Col>
                    <Col md={4}>
                      <Form.Check
                        type="checkbox"
                        id="oxygen"
                        label="Oxygen Support"
                        checked={oxygen}
                        onChange={() => setOxygen(!oxygen)}
                      />
                    </Col>
                    <Col md={4}>
                      <Form.Check
                        type="checkbox"
                        id="ventilation"
                        label="Ventilation Support"
                        checked={ventilation}
                        onChange={() => setVentilation(!ventilation)}
                      />
                    </Col>
                  </Row>
                  <div className="mt-4 text-center">
                    <Button
                      type="submit"
                      variant="primary"
                      className="px-4 py-2"
                      disabled={loading}
                    >
                      {loading ? (
                        <>
                          <Spinner animation="border" size="sm" /> Submitting...
                        </>
                      ) : (
                        "Get Ambulance"
                      )}
                    </Button>
                  </div>
                </Form>
                {error && (
                  <Alert variant="danger" className="mt-4">
                    {error}
                  </Alert>
                )}
               {result && result.length > 0 && (
  <RecommendedAmbulance ambulance={result[0]} />
)}
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default AmbulanceForm;
