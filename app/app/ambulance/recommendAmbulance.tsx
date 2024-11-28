import React from "react";
import { Card, Row, Col, Badge } from "react-bootstrap";

const RecommendedAmbulance = ({ ambulance }: { ambulance: any }) => {
  return (
    <Card className="shadow-lg mt-4">
      <Card.Body>
        <h4 className="text-center text-blue-600 font-bold mb-3">🚑 Recommended Ambulance</h4>
        <Row className="mb-3">
          <Col>
            <h5>
              <Badge bg="info">Ambulance Type:</Badge> {ambulance.ambulance_type}
            </h5>
          </Col>
          <Col>
            <h5>
              <Badge bg="secondary">Number Plate:</Badge> {ambulance.Ambulance_Numberplate}
            </h5>
          </Col>
        </Row>
        <Row>
          <Col md={6}>
            <h6>
              <Badge bg="success">Distance:</Badge> {ambulance.Distance.toFixed(2)} km
            </h6>
          </Col>
          <Col md={6}>
            <h6>
              <Badge bg="warning">ETA:</Badge> {ambulance.ETA} minutes
            </h6>
          </Col>
        </Row>
        <Row className="mt-3">
          <Col>
            <h6>
              <Badge bg="danger">Cardiac Support:</Badge>{" "}
              {ambulance.Cardiac ? "Yes" : "No"}
            </h6>
          </Col>
          <Col>
            <h6>
              <Badge bg="primary">Oxygen Support:</Badge>{" "}
              {ambulance.Oxygen ? "Yes" : "No"}
            </h6>
          </Col>
          <Col>
            <h6>
              <Badge bg="dark">Ventilator:</Badge>{" "}
              {ambulance.Ventilator ? "Yes" : "No"}
            </h6>
          </Col>
        </Row>
        <div className="mt-3 text-center">
          <h6>
            <Badge bg="light" text="dark">Score:</Badge>{" "}
            <strong className="text-blue-500">{ambulance.Score.toFixed(2)}</strong>
          </h6>
        </div>
        <div className="mt-4">
          <h6 className="text-center font-medium text-green-700">
            🚩 Location: {ambulance.Latitude}, {ambulance.Longitude}
          </h6>
        </div>
      </Card.Body>
    </Card>
  );
};

export default RecommendedAmbulance;
