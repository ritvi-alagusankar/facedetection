"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, Loader2, User } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function FacialRecognition() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionResult, setDetectionResult] = useState<string | null>(null)
  const [activeModel, setActiveModel] = useState("deep-learning")
  const [deepLearningModel, setDeepLearningModel] = useState("facenet")
  const [traditionalDetectionModel, setTraditionalDetectionModel] = useState("viola-jones")
  const [traditionalRecognitionModel, setTraditionalRecognitionModel] = useState("eigenfaces")
  const intervalRef = useRef<number | null>(null)
  const processingRef = useRef(false)

  // Start webcam
  const startWebcam = async () => {
    try {
      setIsLoading(true)
      
      // Clear any existing interval first
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }

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

  const postFrameToAPI = async (frame: Blob, modelType: string) => {
    if (processingRef.current) return; // Skip if already processing
    
    processingRef.current = true;
    
    const formData = new FormData();
    formData.append("file", frame);
    formData.append("model_type", modelType);
    formData.append("deep_learning_model", deepLearningModel);
    formData.append("traditional_detection_model", traditionalDetectionModel);
    formData.append("traditional_recognition_model", traditionalRecognitionModel);
    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        console.log("Image processed successfully", data);
        drawBoundingBoxes(data.bounding, data.names);
      } else {
        console.error("Error uploading image", response.statusText);
      }
    } catch (error) {
      console.error("Error posting frame to API", error);
    } finally {
      processingRef.current = false;
    }
  };
  
  const captureFrame = () => {
    if (!videoRef.current || processingRef.current) return;
    
    const canvas = new OffscreenCanvas(
      videoRef.current.videoWidth,
      videoRef.current.videoHeight
    );
    const context = canvas.getContext("2d");
  
    if (!context) return;
  
    // Draw the current video frame onto the canvas
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
  
    // Convert the canvas to a Blob (image file)
    canvas.convertToBlob({ type: "image/jpeg", quality: 0.8 }).then((blob) => {
      postFrameToAPI(blob, activeModel);
    });
  };
  
  const drawBoundingBoxes = (boxes: number[][], names: string[]) => {
    if (!canvasRef.current || !videoRef.current) 
    {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
  
    if (!context) return;

    // Set canvas size to match video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Clear previous drawings
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!boxes || boxes.length === 0)
      {
        return;
      }
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
    
    // Update detection result
    if (names.length > 0) {
      setDetectionResult(names.join(", "));
    }
  };

  // Clean up on unmount or when stream changes
  useEffect(() => {
    // Only start interval if webcam is active
    if (stream && videoRef.current) {
      // Clear any existing interval first
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      
      // Wait for video to be ready
      videoRef.current.onloadedmetadata = () => {
        // Start new interval with faster timing (150ms instead of 300ms)
        intervalRef.current = window.setInterval(() => {
          if (stream && !processingRef.current) { // Only capture if not already processing
            captureFrame();
          }
        }, 150); // Decreased from 300ms to 150ms for more fluid updates
      };
    }

    return () => {
      // Stop interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      
      // Stop webcam stream
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream, activeModel, deepLearningModel, traditionalDetectionModel, traditionalRecognitionModel]);

  // Stop webcam
  const stopWebcam = () => {
    // Stop interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    // Stop webcam stream
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
      
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      
      if (canvasRef.current) {
        const context = canvasRef.current.getContext("2d");
        if (context) {
          context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
      }
    }
    
    setIsDetecting(false);
    setDetectionResult(null);
    processingRef.current = false;
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <Tabs defaultValue="deep-learning" className="w-full" onValueChange={(value) => {
        setActiveModel(value);
        stopWebcam();
      }}>
        <TabsList className="grid w-full grid-cols-2 mb-8 bg-black/20 backdrop-blur-sm p-1 rounded-lg border border-white/10">
          <TabsTrigger 
            value="deep-learning" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500/20 data-[state=active]:to-pink-500/20 data-[state=active]:text-purple-400 data-[state=active]:shadow-[0_0_15px_rgba(192,132,252,0.3)]"
          >
            Deep Learning Methods
          </TabsTrigger>
          <TabsTrigger 
            value="traditional" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:shadow-[0_0_15px_rgba(34,211,238,0.3)]"
          >
            Traditional Methods
          </TabsTrigger>
        </TabsList>

        <TabsContent value="deep-learning" className="mt-0">
          <Card className="border border-white/10 bg-black/20 backdrop-blur-sm shadow-[0_0_30px_rgba(0,0,0,0.3)]">
            <CardHeader className="space-y-2">
              <CardTitle className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Deep Learning Facial Recognition
              </CardTitle>
              <CardDescription className="text-base text-white/70">
                Using advanced neural networks for more accurate recognition.
              </CardDescription>
              <div className="flex flex-col space-y-4 mt-4">
                <div className="flex space-x-4">
                  <Button
                    variant={deepLearningModel === "facenet" ? "default" : "outline"}
                    onClick={() => {
                      setDeepLearningModel("facenet");
                      stopWebcam();
                    }}
                    className={`w-full transition-all duration-300 ${
                      deepLearningModel === "facenet"
                        ? "bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-[0_0_15px_rgba(192,132,252,0.3)]"
                        : "border-purple-400 text-purple-400 hover:bg-purple-400/10 hover:shadow-[0_0_15px_rgba(192,132,252,0.1)]"
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <circle cx="12" cy="12" r="2" fill="currentColor"/>
                        <circle cx="7" cy="7" r="2" fill="currentColor"/>
                        <circle cx="17" cy="7" r="2" fill="currentColor"/>
                        <circle cx="7" cy="17" r="2" fill="currentColor"/>
                        <circle cx="17" cy="17" r="2" fill="currentColor"/>
                      </svg>
                      <span>FaceNet</span>
                    </div>
                  </Button>
                  <Button
                    variant={deepLearningModel === "deepface" ? "default" : "outline"}
                    onClick={() => {
                      setDeepLearningModel("deepface");
                      stopWebcam();
                    }}
                    className={`w-full transition-all duration-300 ${
                      deepLearningModel === "deepface"
                        ? "bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-[0_0_15px_rgba(192,132,252,0.3)]"
                        : "border-purple-400 text-purple-400 hover:bg-purple-400/10 hover:shadow-[0_0_15px_rgba(192,132,252,0.1)]"
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M7 2L7 22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M17 2L17 22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <circle cx="12" cy="12" r="2" fill="currentColor"/>
                        <circle cx="7" cy="7" r="2" fill="currentColor"/>
                        <circle cx="17" cy="7" r="2" fill="currentColor"/>
                        <circle cx="7" cy="17" r="2" fill="currentColor"/>
                        <circle cx="17" cy="17" r="2" fill="currentColor"/>
                      </svg>
                      <span>DeepFace</span>
                    </div>
                  </Button>
                </div>
                <div className="text-sm text-white/50 text-left">
                  {deepLearningModel === "facenet" 
                    ? "FaceNet: A model by Google that maps faces to a compact embedding space using triplet loss for accurate face recognition."
                    : "DeepFace: A model by Facebook that uses 3D alignment and high-dimensional embeddings for face recognition."}
                </div>
              </div>
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
                        <p className="text-white text-lg font-medium">Analyzing with {deepLearningModel}...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              
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

        <TabsContent value="traditional" className="mt-0">
          <Card className="border border-white/10 bg-black/20 backdrop-blur-sm shadow-[0_0_30px_rgba(0,0,0,0.3)]">
            <CardHeader className="space-y-2">
              <CardTitle className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Traditional Facial Recognition
              </CardTitle>
              <CardDescription className="text-base text-white/70">
                Using classical computer vision techniques for face detection and recognition.
              </CardDescription>
              
              <div className="flex flex-col space-y-6 mt-4">
                {/* Face Detection Models */}
                <div>
                  <h3 className="text-white/90 font-medium mb-3">Face Detection Model</h3>
                  <div className="flex space-x-4">
                    <Button
                      variant={traditionalDetectionModel === "viola-jones" ? "default" : "outline"}
                      onClick={() => {
                        setTraditionalDetectionModel("viola-jones");
                        stopWebcam();
                      }}
                      className={`w-full transition-all duration-300 ${
                        traditionalDetectionModel === "viola-jones"
                          ? "bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.3)]"
                          : "border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 hover:shadow-[0_0_15px_rgba(34,211,238,0.1)]"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2" />
                          <rect x="7" y="7" width="4" height="4" fill="currentColor" />
                          <rect x="13" y="7" width="4" height="4" fill="currentColor" />
                          <path d="M8 16h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                        <span>Viola-Jones</span>
                      </div>
                    </Button>
                    <Button
                      variant={traditionalDetectionModel === "hog" ? "default" : "outline"}
                      onClick={() => {
                        setTraditionalDetectionModel("hog");
                        stopWebcam();
                      }}
                      className={`w-full transition-all duration-300 ${
                        traditionalDetectionModel === "hog"
                          ? "bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.3)]"
                          : "border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 hover:shadow-[0_0_15px_rgba(34,211,238,0.1)]"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M2 5h20M2 9h20M2 13h20M2 17h20M2 21h20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                          <path d="M5 2v20M9 2v20M13 2v20M17 2v20M21 2v20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                        <span>HOG</span>
                      </div>
                    </Button>
                  </div>
                  <div className="text-sm text-white/50 text-left mt-2">
                    {traditionalDetectionModel === "viola-jones"
                      ? "Viola-Jones: A classic algorithm for real-time face detection using Haar-like features."
                      : "HOG: Histogram of Oriented Gradients, a feature descriptor used for object detection."}
                </div>
              </div>
                
                
                {/* Face Recognition Models */}
                <div>
                  <h3 className="text-white/90 font-medium mb-3">Face Recognition Model</h3>
                  <div className="flex space-x-4">
                    <Button
                      variant={traditionalRecognitionModel === "eigenfaces" ? "default" : "outline"}
                      onClick={() => {
                        setTraditionalRecognitionModel("eigenfaces");
                        stopWebcam();
                      }}
                      className={`w-full transition-all duration-300 ${
                        traditionalRecognitionModel === "eigenfaces"
                          ? "bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.3)]"
                          : "border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 hover:shadow-[0_0_15px_rgba(34,211,238,0.1)]"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <circle cx="12" cy="12" r="2" fill="currentColor"/>
                          <circle cx="7" cy="7" r="2" fill="currentColor"/>
                          <circle cx="17" cy="7" r="2" fill="currentColor"/>
                          <circle cx="7" cy="17" r="2" fill="currentColor"/>
                          <circle cx="17" cy="17" r="2" fill="currentColor"/>
                        </svg>
                        <span
                          >Eigenfaces</span>
                      </div>
                    </Button>
                    <Button
                      variant={traditionalRecognitionModel === "fisherfaces" ? "default" : "outline"}
                      onClick={() => {
                        setTraditionalRecognitionModel("fisherfaces");
                        stopWebcam();
                      }}
                      className={`w-full transition-all duration-300 ${
                        traditionalRecognitionModel === "fisherfaces"
                          ? "bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.3)]"
                          : "border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 hover:shadow-[0_0_15px_rgba(34,211,238,0.1)]"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <circle cx="12" cy="12" r="2" fill="currentColor"/>
                          <circle cx="7" cy="7" r="2" fill="currentColor"/>
                          <circle cx="17" cy="7" r="2" fill="currentColor"/>
                          <circle cx="7" cy="17" r="2" fill="currentColor"/>
                          <circle cx="17" cy="17" r="2" fill="currentColor"/>
                        </svg>
                        <span>Fisherfaces</span>
                      </div>
                    </Button>
                    <Button
                      variant={traditionalRecognitionModel === "lbph" ? "default" : "outline"}
                      onClick={() => {
                        setTraditionalRecognitionModel("lbph");
                        stopWebcam();
                      }}
                      className={`w-full transition-all duration-300 ${
                        traditionalRecognitionModel === "lbph"
                          ? "bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.3)]"
                          : "border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 hover:shadow-[0_0_15px_rgba(34,211,238,0.1)]"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <circle cx="12" cy="12" r="2" fill="currentColor"/>
                          <circle cx="7" cy="7" r="2" fill="currentColor"/>
                          <circle cx="17" cy="7" r="2" fill="currentColor"/>
                          <circle cx="7" cy="17" r="2" fill="currentColor"/>
                          <circle cx="17" cy="17" r="2" fill="currentColor"/>
                        </svg>
                        <span>LBPH</span>
                      </div>
                    </Button>
                  </div>
                  <div className="text-sm text-white/50 text-left mt-2">
                    {traditionalRecognitionModel === "eigenfaces"
                      ? "Eigenfaces: A method that uses PCA to reduce dimensionality and recognize faces based on eigenvectors."
                      : traditionalRecognitionModel === "fisherfaces"
                      ? "Fisherfaces: An extension of eigenfaces that uses LDA for better discrimination between classes."
                      : "LBPH: Local Binary Patterns Histograms, a simple yet effective method for face recognition."}
                  </div>
                </div>
              </div>
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
                        <p className="text-white text-lg font-medium">Analyzing with {traditionalRecognitionModel}...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
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
      </Tabs>
    </div>
  ) 
}
                     