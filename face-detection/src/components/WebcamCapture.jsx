import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import { Button, Card, CardContent } from "@mui/material";
import './Webcam.css';

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [image, setImage] = useState(null);
  const [frameCount, setFrameCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setFrameCount((prev) => prev + 1);
    }, 100); 

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (frameCount % 5 === 0 && frameCount > 0) {
      capture();
    }
  }, [frameCount]);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImage(imageSrc);
    sendImageToBackend(imageSrc);
  };

  const sendImageToBackend = async (imageSrc) => {
    const formData = new FormData();
    const base64Response = await fetch(imageSrc);
    const blob = await base64Response.blob();

    // Set the filename as .jpg
    const file = new File([blob], "image.jpg", { type: "image/jpeg" });
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      console.log("Image sent successfully", result);
    } catch (error) {
      console.error("Error sending image", error);
    }
};


  return (
    <div className="webcam-container">
      <Card className="webcam-card">
        <CardContent className="flex flex-col items-center p-6">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="rounded-lg shadow-lg w-full max-w-xs"
          />
          <br />
          <div className="capturebutton">
            <Button
              onClick={capture}
              variant="contained"
              color="primary"
            >
              Capture Now
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default WebcamCapture;
