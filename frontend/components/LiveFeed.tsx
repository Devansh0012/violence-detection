import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';

interface ViolenceData {
    violence: boolean;
    score: number;
    timestamp: string;
    annotated_frame?: string;
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

    const processMessage = useCallback(async (event: MessageEvent) => {
        // Expecting a JSON message containing violence info and a base64-encoded image.
        try {
            const data: ViolenceData = JSON.parse(typeof event.data === 'string' ? event.data : '');
            if (data.annotated_frame) {
                setAnnotatedFrame(`data:image/jpeg;base64,${data.annotated_frame}`);
            }
            setViolence(data);
            if (data.violence) {
                onAlert(`Violence detected: ${(data.score * 100).toFixed(1)}% confidence`);
            }
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    }, [onAlert]);

    const connectWebSocket = useCallback(() => {
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onopen = () => {
            setIsConnected(true);
            setError(null);
            console.log('WebSocket connected');
        };

        ws.onclose = () => {
            setIsConnected(false);
            setIsAnalyzing(false);
            console.log('WebSocket disconnected');
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setError('WebSocket connection failed');
            setIsConnected(false);
            setIsAnalyzing(false);
        };

        ws.onmessage = processMessage;

        wsRef.current = ws;
    }, [processMessage]);

    const captureFrame = useCallback(() => {
        if (!isAnalyzing || !wsRef.current) return;

        const imageSrc = webcamRef.current?.getScreenshot();
        if (imageSrc && wsRef.current?.readyState === WebSocket.OPEN) {
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
                setError('Error sending frame');
            }
        }
        // Use setTimeout for a controlled frame rate (e.g. 5 fps)
        setTimeout(captureFrame, 200);
    }, [isAnalyzing]);

    const startAnalysis = () => {
        if (!isConnected) {
            connectWebSocket();
        }
        setIsAnalyzing(true);
        setTimeout(captureFrame, 100);
    };

    const stopAnalysis = () => {
        setIsAnalyzing(false);
        if (wsRef.current) {
            wsRef.current.close();
        }
    };

    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    return (
        <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-4">Live Violence Detection</h2>
            
            <div className="relative">
                {annotatedFrame ? (
                    <img 
                        src={annotatedFrame} 
                        className="w-full rounded-lg"
                        alt="Processed frame"
                    />
                ) : (
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
                )}
                
                <div className="mt-4 space-x-4">
                    {!isAnalyzing ? (
                        <button
                            onClick={startAnalysis}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                        >
                            Start Analysis
                        </button>
                    ) : (
                        <button
                            onClick={stopAnalysis}
                            className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                        >
                            Stop Analysis
                        </button>
                    )}
                </div>
            </div>

            {error && (
                <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
                    {error}
                </div>
            )}

            {violence && (
                <div className={`mt-4 p-4 rounded-lg ${
                    violence.violence ? 'bg-red-100' : 'bg-green-100'
                }`}>
                    <p className="font-bold mb-2">
                        {violence.violence ? 'Violence Detected!' : 'No Violence Detected'}
                    </p>
                    <p>Confidence: {(violence.score * 100).toFixed(2)}%</p>
                    <p className="text-sm text-gray-600">
                        {new Date(violence.timestamp).toLocaleString()}
                    </p>
                </div>
            )}
        </div>
    );
};

export default LiveFeed;