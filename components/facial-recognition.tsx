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
  
  const drawBoundingBoxes = (boxes: number[][], names: string[]) => {
    if (!canvasRef.current || !videoRef.current || !boxes || boxes.length === 0) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    // Set canvas size to match video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Clear previous drawings
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Define colors based on active model
    const colors = activeModel === "traditional" 
      ? {
          primary: 'rgba(0, 255, 255, 0.8)', // Cyan
          accent: 'rgba(0, 255, 255, 1)', // Solid Cyan
          text: 'rgba(0, 255, 255, 1)', // Cyan Text
        }
      : {
          primary: 'rgba(255, 0, 255, 0.8)', // Magenta
          accent: 'rgba(255, 0, 255, 1)', // Solid Magenta
          text: 'rgba(255, 0, 255, 1)', // Magenta Text
        };

    // Draw bounding boxes
    boxes.forEach((box, index) => {
      const [x, y, width, height] = box;
      const name = names[index];

      // Draw the main box
      context.strokeStyle = colors.primary;
      context.lineWidth = 2;
      context.strokeRect(x, y, width, height);

      // Draw corner markers
      const cornerSize = 15;
      const cornerWidth = 3;
      context.strokeStyle = colors.accent;
      context.lineWidth = cornerWidth;
      
      // Top-left corner
      context.beginPath();
      context.moveTo(x, y + cornerSize);
      context.lineTo(x, y);
      context.lineTo(x + cornerSize, y);
      context.stroke();

      // Top-right corner
      context.beginPath();
      context.moveTo(x + width - cornerSize, y);
      context.lineTo(x + width, y);
      context.lineTo(x + width, y + cornerSize);
      context.stroke();

      // Bottom-right corner
      context.beginPath();
      context.moveTo(x + width, y + height - cornerSize);
      context.lineTo(x + width, y + height);
      context.lineTo(x + width - cornerSize, y + height);
      context.stroke();

      // Bottom-left corner
      context.beginPath();
      context.moveTo(x + cornerSize, y + height);
      context.lineTo(x, y + height);
      context.lineTo(x, y + height - cornerSize);
      context.stroke();

      // Draw name text with glow effect
      context.font = 'bold 14px "JetBrains Mono", monospace';
      context.textBaseline = 'middle';
      
      // Draw glow effect
      context.shadowColor = colors.primary;
      context.shadowBlur = 15;
      context.fillStyle = colors.text;
      context.fillText(name, x + 5, y - 10);
      
      // Reset shadow
      context.shadowColor = 'transparent';
      context.shadowBlur = 0;
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
    <div className="w-full max-w-4xl mx-auto p-4">
      <Tabs defaultValue="traditional" className="w-full" onValueChange={(value) => {
        setActiveModel(value);
        stopWebcam(); // Stop camera when switching tabs
      }}>
        <TabsList className="grid w-full grid-cols-2 mb-8 bg-black/20 backdrop-blur-sm p-1 rounded-lg border border-white/10">
          <TabsTrigger 
            value="traditional" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:shadow-[0_0_15px_rgba(34,211,238,0.3)]"
          >
            Traditional Methods
          </TabsTrigger>
          <TabsTrigger 
            value="deep-learning" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500/20 data-[state=active]:to-pink-500/20 data-[state=active]:text-purple-400 data-[state=active]:shadow-[0_0_15px_rgba(192,132,252,0.3)]"
          >
            Deep Learning Methods
          </TabsTrigger>
        </TabsList>

        <TabsContent value="traditional" className="mt-0">
          <Card className="border border-white/10 bg-black/20 backdrop-blur-sm shadow-[0_0_30px_rgba(0,0,0,0.3)]">
            <CardHeader className="space-y-2">
              <CardTitle className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Traditional Facial Recognition
              </CardTitle>
              <CardDescription className="text-base text-white/70">
                Using classical computer vision techniques like Eigenfaces, Fisherfaces, and LBPH.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center space-y-6">
              <div className="relative w-full aspect-video bg-black/30 rounded-xl overflow-hidden flex items-center justify-center border border-white/10 shadow-[0_0_20px_rgba(34,211,238,0.1)]">
                {!stream && !isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full bg-black/20 backdrop-blur-sm">
                    <Camera className="h-16 w-16 text-cyan-400 mb-4" />
                    <p className="text-white/70 text-lg">Camera not active</p>
                  </div>
                )}
                {isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full bg-black/20 backdrop-blur-sm">
                    <Loader2 className="h-12 w-12 animate-spin text-cyan-400 mb-4" />
                    <p className="text-white/70 text-lg">Starting camera...</p>
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
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center">
                      <div className="flex flex-col items-center">
                        <Loader2 className="h-12 w-12 animate-spin text-cyan-400 mb-4" />
                        <p className="text-white text-lg font-medium">Analyzing with deep learning...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {detectionResult && (
                <div className="w-full p-6 bg-black/30 rounded-xl flex items-center space-x-4 border border-white/10 backdrop-blur-sm">
                  <div className="bg-cyan-500/20 p-3 rounded-full">
                    <User className="h-6 w-6 text-cyan-400" />
                  </div>
                  <div>
                    <p className="font-semibold text-lg text-white">Detected: {detectionResult}</p>
                    <p className="text-sm text-white/50">Using traditional methods</p>
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between p-6">
              {!stream ? (
                <Button 
                  onClick={startWebcam} 
                  disabled={isLoading} 
                  className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.3)]"
                >
                  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Start Camera
                </Button>
              ) : (
                <div className="flex space-x-4 w-full">
                  <Button 
                    variant="outline" 
                    onClick={stopWebcam} 
                    className="flex-1 border-cyan-400 text-cyan-400 hover:bg-cyan-400/10"
                  >
                    Stop Camera
                  </Button>
                </div>
              )}
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="deep-learning" className="mt-0">
          <Card className="border border-white/10 bg-black/20 backdrop-blur-sm shadow-[0_0_30px_rgba(0,0,0,0.3)]">
            <CardHeader className="space-y-2">
              <CardTitle className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Deep Learning Facial Recognition
              </CardTitle>
              <CardDescription className="text-base text-white/70">
                Using neural networks like CNN, FaceNet, and DeepFace for more accurate recognition.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center space-y-6">
              <div className="relative w-full aspect-video bg-black/30 rounded-xl overflow-hidden flex items-center justify-center border border-white/10 shadow-[0_0_20px_rgba(192,132,252,0.1)]">
                {!stream && !isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full bg-black/20 backdrop-blur-sm">
                    <Camera className="h-16 w-16 text-purple-400 mb-4" />
                    <p className="text-white/70 text-lg">Camera not active</p>
                  </div>
                )}
                {isLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center h-full bg-black/20 backdrop-blur-sm">
                    <Loader2 className="h-12 w-12 animate-spin text-purple-400 mb-4" />
                    <p className="text-white/70 text-lg">Starting camera...</p>
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
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center">
                      <div className="flex flex-col items-center">
                        <Loader2 className="h-12 w-12 animate-spin text-purple-400 mb-4" />
                        <p className="text-white text-lg font-medium">Analyzing with deep learning...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {detectionResult && (
                <div className="w-full p-6 bg-black/30 rounded-xl flex items-center space-x-4 border border-white/10 backdrop-blur-sm">
                  <div className="bg-purple-500/20 p-3 rounded-full">
                    <User className="h-6 w-6 text-purple-400" />
                  </div>
                  <div>
                    <p className="font-semibold text-lg text-white">Detected: {detectionResult}</p>
                    <p className="text-sm text-white/50">Using deep learning methods</p>
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between p-6">
              {!stream ? (
                <Button 
                  onClick={startWebcam} 
                  disabled={isLoading} 
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-[0_0_15px_rgba(192,132,252,0.3)]"
                >
                  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Start Camera
                </Button>
              ) : (
                <div className="flex space-x-4 w-full">
                  <Button 
                    variant="outline" 
                    onClick={stopWebcam} 
                    className="flex-1 border-purple-400 text-purple-400 hover:bg-purple-400/10"
                  >
                    Stop Camera
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

