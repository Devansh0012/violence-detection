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
    const [isConnected, setIsConnected] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [violence, setViolence] = useState<ViolenceData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null);
    const [connectionAttempts, setConnectionAttempts] = useState(0);

    const processMessage = useCallback((event: MessageEvent) => {
        try {
            const data: ViolenceData = JSON.parse(event.data);
            console.log("Received data:", data);

            if (data.annotated_frame) {
                setAnnotatedFrame(`data:image/jpeg;base64,${data.annotated_frame}`);
            }

            setViolence(data);

            if (data.violence) {
                onAlert(`Violence detected at ${new Date(data.timestamp).toLocaleTimeString()} with ${(data.score * 100).toFixed(1)}% confidence`);
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
        if (!isAnalyzing || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

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

        // Schedule next frame capture
        requestAnimationFrame(captureFrame);
    }, [isAnalyzing]);

    const startAnalysis = () => {
        if (!isConnected) {
            connectWebSocket();
            // Small delay to allow connection to establish
            setTimeout(() => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    setIsAnalyzing(true);
                    requestAnimationFrame(captureFrame);
                }
            }, 1000);
        } else {
            setIsAnalyzing(true);
            requestAnimationFrame(captureFrame);
        }
    };

    const stopAnalysis = () => {
        setIsAnalyzing(false);
    };

    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    return (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Live Violence Detection</h2>

            <div className="relative">
                <div className="rounded-lg overflow-hidden">
                    {annotatedFrame && isAnalyzing ? (
                        <img
                            src={annotatedFrame}
                            className="w-full rounded-lg"
                            alt="Processed frame with annotations"
                        />
                    ) : (
                        <img
                            src={annotatedFrame || '/placeholder.jpg'}
                            className="w-full rounded-lg"
                            alt="Processed frame"
                        />
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
                <div className={`mt-4 p-4 rounded-lg ${violence.violence ? 'bg-red-500/20 border border-red-500/50' : 'bg-green-500/20 border border-green-500/50'
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