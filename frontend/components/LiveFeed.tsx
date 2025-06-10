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
    const [lastFrameTime, setLastFrameTime] = useState<number | null>(null);
    
    // Default to your RTSP URL
    const [rtspUrl, setRtspUrl] = useState<string>('rtsp://test123:test123@192.168.1.23:554/stream1');
    const [useRtsp, setUseRtsp] = useState(true);
    const [connectionStatus, setConnectionStatus] = useState<string | null>(null);
    const [lastActivity, setLastActivity] = useState<number>(Date.now());
    const [streamStats, setStreamStats] = useState<{fps: number}>({fps: 0});
    const [webcamActive, setWebcamActive] = useState(false);

    const processMessage = useCallback((event: MessageEvent) => {
        try {
            // Reset activity tracker
            setLastActivity(Date.now());
            
            // Handle binary data (video frames)
            if (event.data instanceof Blob) {
                // Add size check to filter out empty frames
                if (event.data.size < 100) {
                    console.warn("Received very small frame, possibly invalid");
                    return;
                }
                
                const now = Date.now();
                if (lastFrameTime) {
                    // Calculate and update FPS (smooth over multiple frames)
                    const frameDelay = now - lastFrameTime;
                    const instantFps = 1000 / frameDelay;
                    setStreamStats(prev => ({
                        fps: Math.round((prev.fps * 0.8 + instantFps * 0.2) * 10) / 10
                    }));
                }
                setLastFrameTime(now);
                
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
    }, [annotatedFrame, lastFrameTime, onAlert]);

    // Health check to detect stalled streams
    useEffect(() => {
        if (!isAnalyzing) return;
        
        const interval = setInterval(() => {
            const now = Date.now();
            if (now - lastActivity > 5000) { // No activity for 5 seconds
                console.warn("No WebSocket activity for 5 seconds");
                setConnectionStatus("Stream may be stalled. Trying to reconnect...");
                
                // Try to reconnect
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    // Re-send configuration
                    sendConfiguration();
                } else {
                    // If socket is closed, reconnect
                    connectWebSocket();
                }
            }
        }, 2000);
        
        return () => clearInterval(interval);
    }, [isAnalyzing, lastActivity]);

    const sendConfiguration = useCallback((ws: WebSocket | null = wsRef.current) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.error("WebSocket is not open, cannot send configuration");
            return;
        }
        
        setLastActivity(Date.now());
        
        if (useRtsp && rtspUrl) {
            console.log("Sending RTSP URL:", rtspUrl);
            ws.send(JSON.stringify({ rtsp_url: rtspUrl }));
        } else {
            console.log("Using browser webcam");
            ws.send(JSON.stringify({ use_rtsp: false }));
        }
    }, [useRtsp, rtspUrl]);

    const connectWebSocket = useCallback(() => {
        if (connectionAttempts > 5) {
            setError("Failed to connect after multiple attempts. Please check your server connection.");
            return;
        }
    
        try {
            if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
                wsRef.current.close();
            }
            
            console.log("Attempting to connect to WebSocket...");
            const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
            const ws = new WebSocket(`${wsUrl}/ws`);
            
            ws.onmessage = processMessage;
            
            // Add connection timeout
            const connectionTimeout = setTimeout(() => {
                if (ws.readyState !== WebSocket.OPEN) {
                    ws.close();
                    setConnectionAttempts(prev => prev + 1);
                    connectWebSocket();
                }
            }, 5000);
            
            ws.onopen = () => {
                clearTimeout(connectionTimeout);
                console.log('WebSocket connected successfully');
                setIsConnected(true);
                setError(null);
                setConnectionAttempts(0);
                setLastActivity(Date.now());
                
                // Send configuration immediately when connection is open
                sendConfiguration(ws);
            };
    
            ws.onclose = (event) => {
                setIsConnected(false);
                console.log(`WebSocket disconnected: ${event.code} ${event.reason}`);
                
                if (isAnalyzing) {
                    setTimeout(() => {
                        setConnectionAttempts(prev => prev + 1);
                        connectWebSocket();
                    }, 1000);
                }
            };
            
            wsRef.current = ws;
        } catch (error) {
            console.error("Failed to create WebSocket:", error);
            setError("Failed to create WebSocket connection");
        }
    }, [connectionAttempts, isAnalyzing, processMessage, sendConfiguration]);

    const captureFrame = useCallback(() => {
        if (!isAnalyzing || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || useRtsp) return;

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
                setLastActivity(Date.now());
            } catch (err) {
                console.error('Error sending frame:', err);
            }
        }
    }, [isAnalyzing, useRtsp]);

    const startAnalysis = () => {
        setError(null);
        setIsAnalyzing(true);
        setStreamStats({fps: 0});
        setLastActivity(Date.now());
        
        if (!useRtsp) {
            setWebcamActive(true);
        }
        
        if (!isConnected) {
            // Connect first, configuration will be sent in the onopen handler
            connectWebSocket();
        } else if (wsRef.current?.readyState === WebSocket.OPEN) {
            // Already connected, just send the configuration
            sendConfiguration();
            
            // For browser webcam streaming, start sending frames at 10 fps
            if (!useRtsp) {
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
        
        if (!useRtsp) {
            setWebcamActive(false);
        }
        
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
    }, [useRtsp]);

    // If analysis is active and connection is established, 
    // start webcam frame sending if using browser webcam
    useEffect(() => {
        if (isAnalyzing && isConnected && !useRtsp) {
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
    }, [isAnalyzing, isConnected, useRtsp, captureFrame]);

    // Generate a key that changes each time we start analysis
    // This forces the Webcam component to rerender
    const webcamKey = useRtsp ? `rtsp-${rtspUrl}` : `webcam-${Date.now()}`;
    
    return (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Live Violence Detection</h2>
            
            <div className="mb-4">
                <div className="flex flex-col space-y-2">
                    {/* <div className="flex items-center">
                        <input
                            type="radio"
                            id="useBrowserCamera"
                            checked={!useRtsp}
                            onChange={() => {
                                if (isAnalyzing) {
                                    stopAnalysis();
                                }
                                setUseRtsp(false);
                            }}
                            className="mr-2"
                        />
                        <label htmlFor="useBrowserCamera" className="text-sm">Use Browser Camera</label>
                    </div> */}
                    
                    <div className="flex items-center">
                        <input
                            type="radio"
                            id="useRtsp"
                            checked={useRtsp}
                            onChange={() => {
                                if (isAnalyzing) {
                                    stopAnalysis();
                                }
                                setUseRtsp(true);
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
                    {!useRtsp ? (
                        <div className="relative">
                            <Webcam
                                key={webcamKey}
                                ref={webcamRef}
                                screenshotFormat="image/jpeg"
                                className="w-full rounded-lg"
                                videoConstraints={{
                                    width: 640,
                                    height: 480,
                                    facingMode: "user"
                                }}
                            />
                            {isAnalyzing && annotatedFrame && (
                                <div className="absolute top-0 left-0 w-full h-full">
                                    <Image 
                                        src={annotatedFrame} 
                                        width={640}
                                        height={480}
                                        alt="Violence detection feed"
                                        className="w-full rounded-lg"
                                    />
                                </div>
                            )}
                        </div>
                    ) : (
                        isAnalyzing ? (
                            <div className="relative">
                                {annotatedFrame ? (
                                    <Image 
                                        src={annotatedFrame} 
                                        width={640}
                                        height={480}
                                        alt="Violence detection feed"
                                        className="w-full rounded-lg"
                                    />
                                ) : (
                                    <div className="bg-gray-700 w-full h-80 flex items-center justify-center rounded-lg">
                                        <p>Waiting for RTSP stream...</p>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="bg-gray-700 w-full h-80 flex items-center justify-center rounded-lg">
                                <p>RTSP stream will appear here when analysis starts</p>
                            </div>
                        )
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
                // <div className={`mt-4 p-4 rounded-lg ${
                //     violence.violence 
                //         ? 'bg-red-500/20 border border-red-500/50 text-red-400' 
                //         : 'bg-green-500/20 border border-green-500/50 text-green-400'
                // }`}>
                //     <p className="font-bold mb-2">
                //         {violence.violence 
                //             ? '⚠️ Violence Detected!' 
                //             : '✅ No Violence Detected'}
                //     </p>
                //     <p className={`${
                //         violence.score > 0.7 
                //             ? 'text-red-400' 
                //             : violence.score > 0.4 
                //                 ? 'text-yellow-400' 
                //                 : 'text-green-400'
                //     }`}>
                //         Confidence: {(violence.score * 100).toFixed(2)}%
                //     </p>
                //     <p className="text-sm opacity-75">
                //         {new Date(violence.timestamp).toLocaleString()}
                //     </p>
                // </div>
                <div></div>
            )}
            
            <div className="mt-4 grid grid-cols-2 gap-2">
                <div className={`p-2 rounded text-xs text-center ${
                    isConnected 
                        ? 'bg-green-500/20 border border-green-500/50 text-green-400' 
                        : isAnalyzing 
                            ? 'bg-yellow-500/20 border border-yellow-500/50 text-yellow-400'
                            : 'bg-gray-700 text-gray-300'
                }`}>
                    {isConnected ? "Connected to server" : isAnalyzing ? "Connecting to server..." : "Server connection"}
                </div>
                
                <div className={`p-2 rounded text-xs text-center ${
                    isAnalyzing ? (annotatedFrame 
                        ? 'bg-green-500/20 border border-green-500/50 text-green-400'
                        : 'bg-yellow-500/20 border border-yellow-500/50 text-yellow-400') 
                    : 'bg-gray-700 text-gray-300'
                }`}>
                    {isAnalyzing 
                        ? (annotatedFrame ? "Stream active" : "Waiting for stream...") 
                        : "Stream inactive"}
                </div>
            </div>
        </div>
    );
};

export default LiveFeed;