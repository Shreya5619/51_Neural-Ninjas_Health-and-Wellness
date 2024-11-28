"use client";
import { Button, Container } from 'react-bootstrap'; // Import React Bootstrap components
import { useEffect, useRef, useState } from 'react'; // Import useState for managing file state
import { FaTrash } from 'react-icons/fa'; 
export default function Page() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    // Request microphone access
    const getMicrophonePermission = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);

        mediaRecorderRef.current.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorderRef.current.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          const audioUrl = URL.createObjectURL(audioBlob);
          setAudioUrl(audioUrl);  // Store the URL to play or download
          audioChunksRef.current = [];  // Clear the audio chunks for the next recording
        };
      } catch (err) {
        console.error('Error accessing microphone:', err);
      }
    };

    getMicrophonePermission();
  }, []);

  const startRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.start();
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const [file, setFile] = useState<File | null>(null); // File state for file input

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <Container className="mt-5">
      <h1 className="text-center mb-4" style={{ color: '#007bff' }}>MedAssist</h1>
      <h4 className='text-center mb-5'>Get Rapid Assistance during Emergency</h4>

      <div className="mb-4">
        <h4 htmlFor="exampleFormControlTextarea1" className="form-label">
          Enter your symptoms
        </h4>
        
        <div className="position-relative">
          <textarea
            className="form-control pe-5"
            id="exampleFormControlTextarea1"
            rows="5"
            placeholder="Describe your symptoms here..."
          ></textarea>

          {/* Voice Input Section */}
          
<div className="container mt-5">
  <h4>Tell us about your symptoms through voice input</h4>
  <div className="d-flex justify-content-center align-items-center">
    <Button 
      variant={isRecording ? "danger" : "primary"} 
      onClick={isRecording ? stopRecording : startRecording}
    >
      {isRecording ? "Stop Recording 🎤" : "Start Recording 🎤"}
    </Button>
  </div>

  {audioUrl && (
    <div className="mt-4">
      <h5>Recorded Audio</h5>
      <audio controls src={audioUrl}></audio>
      <div className="mt-2 d-flex align-items-center">
        {/* Delete Button */}
        <Button
          variant="danger"
          size="sm"
          className="ms-2 d-flex align-items-center"
          onClick={() => setAudioUrl(null)} // Reset audioUrl state to null
        >
          <FaTrash className="me-1" /> Delete
        </Button>
      </div>
    </div>
  )}
</div>
        </div>
      </div>

      {/* Upload File Section */}
    
<div className="mb-4">
  <h4 htmlFor="fileUpload" className="form-label">
    Upload an image of the symptoms
  </h4>
  <input
    type="file"
    className="form-control"
    id="fileUpload"
    onChange={handleFileUpload}
  />
  {file && (
    <div className="mt-2">
      <small className="text-muted">File uploaded: {file.name}</small>
      {/* Delete Button with Icon */}
      <Button
        variant="danger"
        size="sm"
        className="ms-2"
        onClick={() => setFile(null)} // Reset the file state to null
      >
        <FaTrash /> {/* Displaying the trash icon */}
      </Button>
    </div>
  )}
</div>
      {/* Submit Button */}
      <div className="col-12 mt-4">
        <Button variant="primary" type="submit">Submit</Button>
      </div>

      {/* Decorative Footer */}
      <footer className="text-center mt-5">
        <p className="text-muted">Powered by MedAssist</p>
      </footer>
    </Container>
  );
}