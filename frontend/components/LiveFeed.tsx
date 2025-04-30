import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import Image from 'next/image';

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
    // Default to your RTSP URL
    const [rtspUrl, setRtspUrl] = useState<string>('rtsp://test123:test123@192.168.1.23:554/stream1');
    // Default to using RTSP since that's what you want to use
    const [useRtsp, setUseRtsp] = useState(true);
    const [useSystemCamera, setUseSystemCamera] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState<string | null>(null);

    const processMessage = useCallback((event: MessageEvent) => {
        try {
            // Handle binary data (video frames)
            if (event.data instanceof Blob) {
                // Revoke the old URL to prevent memory leaks
                if (annotatedFrame && annotatedFrame.startsWith('blob:')) {
                    URL.revokeObjectURL(annotatedFrame);
                }
                
                const url = URL.createObjectURL(event.data);
                setAnnotatedFrame(url);
                return;
            }
            
            // Handle JSON data
            const data = JSON.parse(event.data);
            console.log("Received WebSocket data:", data);
            
            // Check for error messages
            if (data.error) {
                setError(`Server error: ${data.error}`);
                return;
            }
            
            if (data.message) {
                console.log("Server message:", data.message);
                setConnectionStatus(data.message);
                // Clear the status message after 5 seconds
                setTimeout(() => setConnectionStatus(null), 5000);
            }
            
            // Process violence detection results
            if (data.violence_score !== undefined) {
                const isViolent = data.is_violent;
                setViolence({
                    violence: isViolent,
                    score: data.violence_score,
                    timestamp: data.timestamp,
                    annotated_frame: ""  // We get the frame separately as binary data
                });
                
                if (isViolent) {
                    onAlert(`Violence detected at ${new Date(data.timestamp).toLocaleTimeString()} with ${(data.violence_score * 100).toFixed(1)}% confidence`);
                }
            }
        } catch (err) {
            console.error('Error processing WebSocket message:', err);
        }
    }, [annotatedFrame, onAlert]);

    const connectWebSocket = useCallback(() => {
        if (connectionAttempts > 5) {
            setError("Failed to connect after multiple attempts. Please check your server connection.");
            return;
        }
    
        try {
            // Close existing connection if any
            if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
                console.log("Closing existing WebSocket connection");
                wsRef.current.close();
            }
            
            console.log("Attempting to connect to WebSocket...");
            const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
            const ws = new WebSocket(`${wsUrl}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected successfully');
                setIsConnected(true);
                setError(null);
                setConnectionAttempts(0);
                
                // Add a small delay before sending the initial configuration
                setTimeout(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        console.log("Sending initial configuration...");
                        // If we're analyzing, send the configuration immediately
                        if (isAnalyzing) {
                            sendConfiguration(ws);
                        }
                    } else {
                        console.log("WebSocket is not open, cannot send initial configuration");
                    }
                }, 500); 
            };
    
            ws.onclose = (event) => {
                setIsConnected(false);
                console.log(`WebSocket disconnected: ${event.code} ${event.reason}`);
                
                // Auto-reconnect if we're supposed to be analyzing
                if (isAnalyzing) {
                    console.log("Attempting to reconnect...");
                    setTimeout(() => {
                        connectWebSocket();
                    }, 2000);  // Reconnect after 2 seconds
                }
            };
    
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setError('WebSocket connection failed. The server might be offline or unreachable.');
                setConnectionAttempts(prev => prev + 1);
            };
    
            ws.onmessage = processMessage;
            wsRef.current = ws;
        } catch (error) {
            console.error("Failed to create WebSocket:", error);
            setError("Failed to create WebSocket connection");
        }
    }, [connectionAttempts, isAnalyzing, processMessage]);
    
    const sendConfiguration = (ws: WebSocket | null = wsRef.current) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.error("WebSocket is not open, cannot send configuration");
            return;
        }
        
        if (useRtsp && rtspUrl) {
            console.log("Sending RTSP URL:", rtspUrl);
            ws.send(JSON.stringify({ rtsp_url: rtspUrl }));
        } else if (useSystemCamera) {
            console.log("Using system camera");
            ws.send(JSON.stringify({ use_system_camera: true }));
        } else {
            console.log("Using browser webcam");
            ws.send(JSON.stringify({ use_rtsp: false }));
        }
    };

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
        setError(null);
        setIsAnalyzing(true);
        
        if (!isConnected) {
            // Connect first, configuration will be sent in the onopen handler
            connectWebSocket();
        } else if (wsRef.current?.readyState === WebSocket.OPEN) {
            // Already connected, just send the configuration
            sendConfiguration();
            
            // For browser webcam streaming, start sending frames at 10 fps
            if (!useRtsp && !useSystemCamera) {
                if (frameIntervalRef.current) {
                    clearInterval(frameIntervalRef.current);
                }
                frameIntervalRef.current = window.setInterval(captureFrame, 100);
            }
        } else {
            setError("WebSocket connection is not in OPEN state. Please try again.");
            setIsAnalyzing(false);
        }
    };

    const stopAnalysis = () => {
        setIsAnalyzing(false);
        
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }
        
        // Send a message to stop RTSP streaming on the server
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ use_rtsp: false }));
        }
    };

    useEffect(() => {
        // Clean up resources on unmount
        return () => {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
            }
            
            if (wsRef.current) {
                wsRef.current.close();
            }
            
            // Clean up any blob URLs
            if (annotatedFrame && annotatedFrame.startsWith('blob:')) {
                URL.revokeObjectURL(annotatedFrame);
            }
        };
    }, [annotatedFrame]);

    // When camera choice changes, reset the analysis
    useEffect(() => {
        if (isAnalyzing) {
            stopAnalysis();
        }
    }, [useRtsp, useSystemCamera]);

    // If analysis is active and connection is established, 
    // start webcam frame sending if using browser webcam
    useEffect(() => {
        if (isAnalyzing && isConnected && !useRtsp && !useSystemCamera) {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
            }
            frameIntervalRef.current = window.setInterval(captureFrame, 100);
        }
        
        return () => {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
            }
        };
    }, [isAnalyzing, isConnected, useRtsp, useSystemCamera, captureFrame]);

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
                            <p className="text-xs text-gray-400 mt-1">
                                Example: rtsp://username:password@ip-address:port/stream
                            </p>
                        </div>
                    )}
                </div>
            </div>
            
            <div className="relative">
                <div className="rounded-lg overflow-hidden">
                    {annotatedFrame && isAnalyzing ? (
                        <div className="relative">
                            <Image 
                                src={annotatedFrame} 
                                width={640}
                                height={480}
                                className="w-full rounded-lg"
                                alt="Processed frame with annotations"
                            />
                            {connectionStatus && (
                                <div className="absolute top-2 left-2 bg-blue-600/70 text-white px-2 py-1 rounded text-sm">
                                    {connectionStatus}
                                </div>
                            )}
                        </div>
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
                        <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-1 rounded-lg text-sm font-bold">
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
            
            {isConnected ? (
                <div className="mt-4 p-2 bg-green-500/20 border border-green-500/50 text-green-400 rounded text-xs text-center">
                    Connected to server
                </div>
            ) : isAnalyzing ? (
                <div className="mt-4 p-2 bg-yellow-500/20 border border-yellow-500/50 text-yellow-400 rounded text-xs text-center">
                    Connecting to server...
                </div>
            ) : null}
        </div>
    );
};

export default LiveFeed;