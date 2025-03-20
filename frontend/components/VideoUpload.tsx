import React, { useState } from "react";

const VideoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [message, setMessage] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;
    setMessage("Uploading and analyzing video...");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/analyze_video", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setMessage(data.message);
        // Assuming the backend serves static files, adjust URL accordingly.
        setAnnotatedVideoUrl(`http://localhost:8000/${data.annotated_video}`);
      } else {
        setMessage("Error: " + data.detail);
      }
    } catch (err: any) {
      console.error(err);
      setMessage("An error occurred while uploading video.");
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Custom Video Analysis</h2>
      <label htmlFor="video-upload" className="block text-sm font-medium text-gray-700">
        Upload Video
      </label>
      <input id="video-upload" type="file" accept="video/*" onChange={handleFileChange} />
      <button
        onClick={uploadVideo}
        className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
      >
        Upload and Analyze Video
      </button>
      {message && (
        <p className="mt-4 p-3 bg-gray-200 rounded">{message}</p>
      )}
      {annotatedVideoUrl && (
        <div className="mt-4">
          <h3 className="text-xl font-semibold mb-2">Annotated Video</h3>
          <video src={annotatedVideoUrl} controls className="w-full rounded-lg" />
        </div>
      )}
    </div>
  );
};

export default VideoUpload;