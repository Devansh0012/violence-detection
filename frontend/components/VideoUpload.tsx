import React, { useState } from "react";

const VideoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [message, setMessage] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [progress, setProgress] = useState<number>(0);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setSelectedFile(e.target.files[0]);
      setMessage("");
      setStatus("idle");
      setAnnotatedVideoUrl(null);
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
          const newProgress = prev + Math.random() * 15;
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
        setMessage(data.message);
        setAnnotatedVideoUrl(`http://localhost:8000/${data.annotated_video}`);
      } else {
        setStatus("error");
        setMessage("Error: " + data.detail);
      }
    } catch (err: any) {
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span className="text-gray-300 mb-1">Upload a video file</span>
            <span className="text-gray-500 text-sm">MP4, MOV, or AVI</span>
          </label>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center">
              <svg className="w-8 h-8 text-blue-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span className="text-gray-300 truncate max-w-xs">{selectedFile.name}</span>
              <button 
                onClick={() => setSelectedFile(null)}
                className="ml-2 text-gray-400 hover:text-gray-300"
              >
                ✕
              </button>
            </div>
            
            {status !== "uploading" && (
              <button
                onClick={uploadVideo}
                disabled={false}
                className="w-full py-2 px-4 rounded-lg text-white font-medium transition-colors bg-blue-600 hover:bg-blue-700"
              >
                Analyze Video
              </button>
            )}
          </div>
        )}
      </div>
      
      {status === "uploading" && (
        <div className="mt-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">Analyzing video...</span>
            <span className="text-gray-400">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-600 rounded-full h-2">
            <div 
              className={`bg-blue-500 h-2 rounded-full transition-all duration-300 ease-in-out w-[${Math.round(progress)}%]`}
            ></div>
          </div>
        </div>
      )}

      {message && status !== "uploading" && (
        <div className={`mt-4 p-3 rounded-lg text-sm ${
          status === "error" 
            ? "bg-red-500/20 text-red-400 border border-red-500/50" 
            : "bg-green-500/20 text-green-400 border border-green-500/50"
        }`}>
          {message}
        </div>
      )}

      {annotatedVideoUrl && (
        <div className="mt-6">
          <h3 className="text-lg font-medium mb-3 text-gray-300">Analyzed Video Result</h3>
          <div className="rounded-lg overflow-hidden bg-gray-900">
            <video src={annotatedVideoUrl} controls className="w-full h-auto" />
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;