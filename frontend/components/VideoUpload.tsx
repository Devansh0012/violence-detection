'use client';

import React, { useState } from "react";

const VideoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [progress, setProgress] = useState<number>(0);
  //const [analysisData, setAnalysisData] = useState<null | Record<string, unknown>>(null);
  

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setSelectedFile(e.target.files[0]);
      setMessage("");
      setStatus("idle");
    }
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;
    
    setStatus("uploading");
    setMessage("Uploading and analyzing video...");
    setProgress(10); // Start progress

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + Math.random() * 10;
          return newProgress >= 90 ? 90 : newProgress; // Cap at 90% until response
        });
      }, 1000);

      const response = await fetch("http://localhost:8000/analyze_video", {
        method: "POST",
        body: formData,
      });
      
      clearInterval(progressInterval);
      setProgress(100);
      
      const data = await response.json();
      
      if (response.ok) {
        setStatus("success");
        setMessage(data.message || "Video analyzed successfully");
        // setAnalysisData(data);
        
        // Store the analysis data in localStorage for use on the analysis page
        localStorage.setItem('currentAnalysis', JSON.stringify(data));
        
        // Navigate to the analysis page
        const analysisId = data.analysis_id || Date.now().toString();
        const videoUrl = encodeURIComponent(data.video_url || '');
        
        // Use window.location for navigation after upload completes
        window.location.href = `/analysis/${analysisId}?video=${videoUrl}`;
      } else {
        setStatus("error");
        setMessage("Error: " + (data.detail || "Failed to analyze video"));
      }
    } catch (err: unknown) {
      setStatus("error");
      console.error(err);
      setMessage("An error occurred while uploading video.");
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
      <h2 className="text-xl font-semibold mb-4">Video Analysis</h2>
      
      <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center">
        <input 
          id="video-upload" 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange}
          className="hidden" 
        />
        
        {!selectedFile ? (
          <label 
            htmlFor="video-upload" 
            className="cursor-pointer flex flex-col items-center justify-center"
          >
            <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
            <span className="text-gray-300">Click to upload video</span>
            <span className="text-gray-500 text-sm mt-1">MP4, AVI, MOV files</span>
          </label>
        ) : (
          <div className="space-y-4">
            <div className="text-gray-300">{selectedFile.name}</div>
            
            {status === "uploading" ? (
              <div className="w-full">
                <div className="bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div 
                    className={`bg-blue-500 h-full rounded-full w-[${Math.round(progress)}%]`}
                  ></div>
                </div>
                <div className="text-xs text-gray-400 mt-1 text-right">{Math.round(progress)}%</div>
              </div>
            ) : (
              <button
                onClick={uploadVideo}
                disabled={false}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-md text-white transition"
              >
                Analyze Video
              </button>
            )}
            
            <button 
              onClick={() => setSelectedFile(null)}
              className="text-gray-400 hover:text-gray-300 text-sm underline"
              disabled={status === "uploading"}
            >
              Cancel
            </button>
          </div>
        )}
      </div>
      
      {message && (
        <div className={`mt-4 p-3 rounded-md ${
          status === "error" 
            ? "bg-red-500/20 text-red-400 border border-red-500/50" 
            : "bg-green-500/20 text-green-400 border border-green-500/50"
        }`}>
          {message}
        </div>
      )}
    </div>
  );
};

export default VideoUpload;