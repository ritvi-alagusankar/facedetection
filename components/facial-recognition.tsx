"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, Loader2, User } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function FacialRecognition() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef(null);
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionResult, setDetectionResult] = useState<string | null>(null)
  const [activeModel, setActiveModel] = useState("traditional")

  // Start webcam
  const startWebcam = async () => {
    try {
      setIsLoading(true)
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }

      setStream(mediaStream)
      setIsLoading(false)
    } catch (error) {
      console.error("Error accessing webcam:", error)
      setIsLoading(false)
    }
  }

  const postFrameToAPI = async (frame) => {
    const formData = new FormData();
    formData.append("file", frame);
  
    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,  // Ensure this contains the file you want to upload
      });
      if (response.ok) {
        const data = await response.json(); // Get JSON response
        console.log("Image uploaded successfully", data);
        drawBoundingBoxes(data.bounding, data.names); // Call function to draw bounding boxes
      } else {
        console.error("Error uploading image", response.statusText);
      }
    } catch (error) {
      console.error("Error posting frame to API", error);
    }
  };
  
  const captureFrame = () => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
  
    // Set canvas size to match video dimensions
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
  
    // Draw the current video frame onto the canvas
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
  
    // Convert the canvas to a Blob (image file)
    canvas.toBlob((blob) => {
      if (blob) {
        postFrameToAPI(blob);
      }
    }, "image/jpeg");
  };
  
  const drawBoundingBoxes = (boxes, names) => {
    if (!canvasRef.current || !videoRef.current || !boxes || boxes.length === 0) return;
  
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
  
    // Set canvas size to match video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
  
    // Clear previous drawings
    context.clearRect(0, 0, canvas.width, canvas.height);
  
    // Draw bounding boxes
    context.strokeStyle = "red";
    context.lineWidth = 2;
    context.font = "16px Arial";
    context.fillStyle = "red";
  
    boxes.forEach((box, index) => {
      console.log(box);
      const [x, y, width, height] = box;
      context.strokeRect(x, y, width, height);
      context.fillText(names[index], x, y - 5);
    });
  };
  

  // Stop webcam
  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
      if (canvasRef.current) {
        const context = canvasRef.current.getContext("2d")
        context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
    }
    setIsDetecting(false)
    setDetectionResult(null)
  }

  // Detect face
  const detectFace = () => {
    setIsDetecting(true)

    // Simulate detection process
    setTimeout(() => {
      const results =
        activeModel === "traditional"
          ? ["John Doe", "Jane Smith", "Unknown"]
          : ["John Doe (98%)", "Jane Smith (95%)", "Alex Johnson (87%)"]

      const randomResult = results[Math.floor(Math.random() * results.length)]
      setDetectionResult(randomResult)
      setIsDetecting(false)
    }, 2000)
  }

  // Clean up on unmount
  useEffect(() => {
    const interval = setInterval(captureFrame, 100); // Capture every 100ms

    return () => {
      // Stop webcam stream
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      // Clear the interval to stop frame capture
      clearInterval(interval);
    } 
  }, [stream])

  return (
    <div className="w-full max-w-4xl mx-auto">
      <Tabs defaultValue="traditional" className="w-full" onValueChange={setActiveModel}>
        <TabsList className="grid w-full grid-cols-2 mb-8">
          <TabsTrigger value="traditional">Traditional Methods</TabsTrigger>
          <TabsTrigger value="deep-learning">Deep Learning Methods</TabsTrigger>
        </TabsList>

        <TabsContent value="traditional" className="mt-0">
          <Card className="border-2">
            <CardHeader>
              <CardTitle>Traditional Facial Recognition</CardTitle>
              <CardDescription>
                Using classical computer vision techniques like Eigenfaces, Fisherfaces, and LBPH.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center space-y-4">
              <div className="relative w-full aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center">
                {!stream && !isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full">
                    <Camera className="h-12 w-12 text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">Camera not active</p>
                  </div>
                )}
                {isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full">
                    <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
                    <p className="text-muted-foreground">Starting camera...</p>
                  </div>
                )}
                <div className="relative w-full h-full">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover ${!stream ? "hidden" : ""}`}
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
              />
              {isDetecting && stream && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <div className="flex flex-col items-center">
                    <Loader2 className="h-8 w-8 animate-spin text-white mb-2" />
                    <p className="text-white">Analyzing with deep learning...</p>
                  </div>
                </div>
              )}

            </div>
              </div>

              {detectionResult && (
                <div className="w-full p-4 bg-muted rounded-lg flex items-center">
                  <User className="h-6 w-6 mr-2 text-primary" />
                  <div>
                    <p className="font-medium">Detected: {detectionResult}</p>
                    <p className="text-sm text-muted-foreground">Using traditional methods</p>
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              {!stream ? (
                <Button onClick={startWebcam} disabled={isLoading}>
                  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Start Camera
                </Button>
              ) : (
                <div className="flex space-x-2">
                  <Button variant="outline" onClick={stopWebcam}>
                    Stop Camera
                  </Button>
                  <Button onClick={detectFace} disabled={isDetecting}>
                    {isDetecting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Detect Face
                  </Button>
                </div>
              )}
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="deep-learning" className="mt-0">
          <Card className="border-2">
            <CardHeader>
              <CardTitle>Deep Learning Facial Recognition</CardTitle>
              <CardDescription>
                Using neural networks like CNN, FaceNet, and DeepFace for more accurate recognition.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center space-y-4">
              <div className="relative w-full aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center">
                {!stream && !isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full">
                    <Camera className="h-12 w-12 text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">Camera not active</p>
                  </div>
                )}
                {isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full">
                    <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
                    <p className="text-muted-foreground">Starting camera...</p>
                  </div>
                )}
                <div className="relative w-full h-full">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover ${!stream ? "hidden" : ""}`}
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
              />
              {isDetecting && stream && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <div className="flex flex-col items-center">
                    <Loader2 className="h-8 w-8 animate-spin text-white mb-2" />
                    <p className="text-white">Analyzing with deep learning...</p>
                  </div>
                </div>
              )}

            </div>
              </div>

              {detectionResult && (
                <div className="w-full p-4 bg-muted rounded-lg flex items-center">
                  <User className="h-6 w-6 mr-2 text-primary" />
                  <div>
                    <p className="font-medium">Detected: {detectionResult}</p>
                    <p className="text-sm text-muted-foreground">Using deep learning methods</p>
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              {!stream ? (
                <Button onClick={startWebcam} disabled={isLoading}>
                  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Start Camera
                </Button>
              ) : (
                <div className="flex space-x-2">
                  <Button variant="outline" onClick={stopWebcam}>
                    Stop Camera
                  </Button>
                  <Button onClick={detectFace} disabled={isDetecting}>
                    {isDetecting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Detect Face
                  </Button>
                </div>
              )}
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

