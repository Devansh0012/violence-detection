import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';

interface ViolenceData {
    violence: boolean;
    score: number;
    timestamp: string;
    annotated_frame: string;
}

interface LiveFeedProps {
    onAlert: (message: string) => void;
}

const LiveFeed: React.FC<LiveFeedProps> = ({ onAlert }) => {
    const webcamRef = useRef<Webcam | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const frameIntervalRef = useRef<number | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [violence, setViolence] = useState<ViolenceData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null);
    const [connectionAttempts, setConnectionAttempts] = useState(0);
    const [rtspUrl, setRtspUrl] = useState<string>('');
    const [useRtsp, setUseRtsp] = useState(false);
    const [useSystemCamera, setUseSystemCamera] = useState(false);

    const processMessage = useCallback((event: MessageEvent) => {
        try {
            const data = JSON.parse(event.data);
            console.log("Received data:", data);
            
            // Check for error messages
            if (data.error) {
                setError(`Server error: ${data.error}`);
                return;
            }
            
            if (data.message) {
                console.log("Server message:", data.message);
            }
            
            if (data.annotated_frame) {
                setAnnotatedFrame(`data:image/jpeg;base64,${data.annotated_frame}`);
            }
            
            if (data.violence !== undefined) {
                setViolence({
                    violence: data.violence,
                    score: data.score,
                    timestamp: data.timestamp,
                    annotated_frame: data.annotated_frame
                });
                
                if (data.violence) {
                    onAlert(`Violence detected at ${new Date(data.timestamp).toLocaleTimeString()} with ${(data.score * 100).toFixed(1)}% confidence`);
                }
            }
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    }, [onAlert]);

    const connectWebSocket = useCallback(() => {
        if (connectionAttempts > 3) {
            setError("Failed to connect after multiple attempts. Please check your server connection.");
            return;
        }

        try {
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                setIsConnected(true);
                setError(null);
                setConnectionAttempts(0);
                console.log('WebSocket connected');
            };

            ws.onclose = () => {
                setIsConnected(false);
                setIsAnalyzing(false);
                console.log('WebSocket disconnected');
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setError('WebSocket connection failed. The server might be offline.');
                setIsConnected(false);
                setIsAnalyzing(false);
                setConnectionAttempts(prev => prev + 1);
            };

            ws.onmessage = processMessage;

            wsRef.current = ws;
        } catch (error) {
            console.error("Failed to create WebSocket:", error);
            setError("Failed to create WebSocket connection");
        }
    }, [processMessage, connectionAttempts]);

    const captureFrame = useCallback(() => {
        if (!isAnalyzing || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || useRtsp || useSystemCamera) return;

        const imageSrc = webcamRef.current?.getScreenshot();
        if (imageSrc) {
            try {
                const base64Data = imageSrc.split(',')[1];
                const byteCharacters = atob(base64Data);
                const byteArray = new Uint8Array(byteCharacters.length);
                
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteArray[i] = byteCharacters.charCodeAt(i);
                }
                
                const blob = new Blob([byteArray], { type: 'image/jpeg' });
                wsRef.current.send(blob);
            } catch (err) {
                console.error('Error sending frame:', err);
            }
        }
    }, [isAnalyzing, useRtsp, useSystemCamera]);

    const startAnalysis = () => {
        if (!isConnected) {
            connectWebSocket();
            
            // Small delay to allow connection to establish
            setTimeout(() => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    setIsAnalyzing(true);
                    
                    // Send configuration based on camera choice
                    if (useRtsp && rtspUrl) {
                        console.log("Sending RTSP URL:", rtspUrl);
                        wsRef.current.send(JSON.stringify({ rtsp_url: rtspUrl }));
                    } else if (useSystemCamera) {
                        console.log("Using system camera");
                        wsRef.current.send(JSON.stringify({ use_system_camera: true }));
                    } else {
                        console.log("Using browser webcam");
                        wsRef.current.send(JSON.stringify({ use_rtsp: false }));
                        
                        // For browser webcam streaming, start sending frames at 10 fps
                        frameIntervalRef.current = window.setInterval(captureFrame, 100);
                    }
                }
            }, 1000);
        } else {
            setIsAnalyzing(true);
            
            // Send configuration based on camera choice
            if (useRtsp && rtspUrl && wsRef.current) {
                wsRef.current.send(JSON.stringify({ rtsp_url: rtspUrl }));
            } else if (useSystemCamera && wsRef.current) {
                wsRef.current.send(JSON.stringify({ use_system_camera: true }));
            } else {
                // For browser webcam streaming, start sending frames at 10 fps
                frameIntervalRef.current = window.setInterval(captureFrame, 100);
            }
        }
    };

    const stopAnalysis = () => {
        setIsAnalyzing(false);
        setAnnotatedFrame(null);
        
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }
    };

    useEffect(() => {
        return () => {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
            }
            
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    // When camera choice changes, reset the analysis
    useEffect(() => {
        if (isAnalyzing) {
            stopAnalysis();
        }
    }, [useRtsp, useSystemCamera]);

    return (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Live Violence Detection</h2>
            
            <div className="mb-4">
                <div className="flex flex-col space-y-2">
                    <div className="flex items-center">
                        <input
                            type="radio"
                            id="useBrowserCamera"
                            checked={!useRtsp && !useSystemCamera}
                            onChange={() => {
                                setUseRtsp(false);
                                setUseSystemCamera(false);
                            }}
                            className="mr-2"
                        />
                        <label htmlFor="useBrowserCamera" className="text-sm">Use Browser Camera</label>
                    </div>
                    
                    <div className="flex items-center">
                        <input
                            type="radio"
                            id="useSystemCamera"
                            checked={!useRtsp && useSystemCamera}
                            onChange={() => {
                                setUseRtsp(false);
                                setUseSystemCamera(true);
                            }}
                            className="mr-2"
                        />
                        <label htmlFor="useSystemCamera" className="text-sm">Use System Camera</label>
                    </div>
                    
                    <div className="flex items-center">
                        <input
                            type="radio"
                            id="useRtsp"
                            checked={useRtsp}
                            onChange={() => {
                                setUseRtsp(true);
                                setUseSystemCamera(false);
                            }}
                            className="mr-2"
                        />
                        <label htmlFor="useRtsp" className="text-sm">Use RTSP Camera</label>
                    </div>
                    
                    {useRtsp && (
                        <div className="mt-2">
                            <input
                                type="text"
                                placeholder="RTSP URL (rtsp://...)"
                                value={rtspUrl}
                                onChange={(e) => setRtspUrl(e.target.value)}
                                className="w-full p-2 rounded bg-gray-700 text-white"
                            />
                        </div>
                    )}
                </div>
            </div>
            
            <div className="relative">
                <div className="rounded-lg overflow-hidden">
                    {annotatedFrame && isAnalyzing ? (
                        <img 
                            src={annotatedFrame} 
                            className="w-full rounded-lg"
                            alt="Processed frame with annotations"
                        />
                    ) : (
                        !useRtsp && !useSystemCamera ? (
                            <Webcam
                                ref={webcamRef}
                                screenshotFormat="image/jpeg"
                                className="w-full rounded-lg"
                                videoConstraints={{
                                    width: 640,
                                    height: 480,
                                    facingMode: "user"
                                }}
                            />
                        ) : (
                            <div className="bg-gray-700 w-full h-80 flex items-center justify-center rounded-lg">
                                <p>{useRtsp ? "RTSP stream" : "System camera"} will appear here when analysis starts</p>
                            </div>
                        )
                    )}
                    
                    {violence && violence.violence && (
                        <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-lg text-sm font-bold">
                            Violence Detected: {(violence.score * 100).toFixed(1)}%
                        </div>
                    )}
                </div>
                
                <div className="mt-4 flex justify-center space-x-4">
                    {!isAnalyzing ? (
                        <button
                            onClick={startAnalysis}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200"
                            disabled={useRtsp && !rtspUrl}
                        >
                            Start Analysis
                        </button>
                    ) : (
                        <button
                            onClick={stopAnalysis}
                            className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200"
                        >
                            Stop Analysis
                        </button>
                    )}
                </div>
            </div>

            {error && (
                <div className="mt-4 p-3 bg-red-500/20 border border-red-500/50 text-red-400 rounded-lg">
                    {error}
                </div>
            )}

            {violence && (
                <div className={`mt-4 p-4 rounded-lg ${
                    violence.violence ? 'bg-red-500/20 border border-red-500/50' : 'bg-green-500/20 border border-green-500/50'
                } ${violence.violence ? 'text-red-400' : 'text-green-400'}`}>
                    <p className="font-bold mb-2">
                        {violence.violence ? '⚠️ Violence Detected!' : '✅ No Violence Detected'}
                    </p>
                    <p>Confidence: {(violence.score * 100).toFixed(2)}%</p>
                    <p className="text-sm opacity-75">
                        {new Date(violence.timestamp).toLocaleString()}
                    </p>
                </div>
            )}
        </div>
    );
};

export default LiveFeed;