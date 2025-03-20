import React, { useState } from 'react';
import { analyzeViolence } from '../lib/api';

interface AnalysisResult {
    violence_detected: boolean;
    confidence_score: number;
    timestamp: string;
}

const ViolenceDetection: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) {
            setError('Please select a file');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const analysisResult = await analyzeViolence(file);
            setResult(analysisResult);
        } catch (err) {
            setError('Error analyzing file. Please try again.');
            console.error('Analysis error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-4">Violence Detection</h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4">
                    <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700">
                        Upload file
                    </label>
                    <input
                        id="file-upload"
                        type="file"
                        onChange={handleFileChange}
                        accept="image/*,video/*"
                        className="w-full"
                        title="Choose a file to upload"
                    />
                </div>

                <button
                    type="submit"
                    disabled={!file || loading}
                    className={`w-full py-2 px-4 rounded-lg text-white font-medium
                        ${loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}
                        transition duration-200`}
                >
                    {loading ? 'Analyzing...' : 'Analyze'}
                </button>
            </form>

            {error && (
                <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
                    {error}
                </div>
            )}

            {result && (
                <div className={`mt-6 p-4 rounded-lg ${
                    result.violence_detected ? 'bg-red-100' : 'bg-green-100'
                }`}>
                    <h3 className="font-bold mb-2">Analysis Result:</h3>
                    <p className="mb-1">
                        Status: {result.violence_detected ? 'Violence Detected!' : 'No Violence Detected'}
                    </p>
                    <p className="mb-1">
                        Confidence: {(result.confidence_score * 100).toFixed(2)}%
                    </p>
                    <p className="text-sm text-gray-600">
                        Timestamp: {new Date(result.timestamp).toLocaleString()}
                    </p>
                </div>
            )}
        </div>
    );
};

export default ViolenceDetection;