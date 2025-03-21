'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import Image from 'next/image';

interface ViolentSegment {
  start_frame: number;
  start_time: number;
  end_frame: number;
  end_time: number;
  duration: number;
  avg_score: number;
  scores: number[];
}

interface AnalysisData {
  message: string;
  analysis_id: string;
  video_url: string;
  keyframes: string[];
  summary: {
    total_frames: number;
    violence_frames: number;
    violence_percentage: number;
    classification: 'violent' | 'non-violent' | 'ambiguous';
    duration_seconds: number;
  };
  violent_segments: ViolentSegment[];
  results: {
    frame: number;
    time: number;
    violence_detected: boolean;
    score: number;
  }[];
}

export default function AnalysisPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const analysisId = params?.id as string;
  const videoUrl = searchParams?.get('video');

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'keyframes' | 'timeline'>('summary');
  const [selectedKeyframe, setSelectedKeyframe] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        // For now, we'll use the video URL parameter as our analysis data identifier
        // In a real app, you'd make a backend call to retrieve the analysis data
        if (!videoUrl) {
          setError("No video URL provided");
          setLoading(false);
          return;
        }

        // Make request to retrieve the analysis data
        const response = await fetch(`http://localhost:8000/analyze_video/${analysisId}`).catch(() => {
          // Fallback mock data if endpoint doesn't exist yet
          return {
            ok: true,
            json: () => Promise.resolve({
              message: "Video analyzed successfully",
              analysis_id: analysisId,
              video_url: videoUrl,
              keyframes: ["/path/to/keyframe1.jpg", "/path/to/keyframe2.jpg"],
              summary: {
                total_frames: 300,
                violence_frames: 120,
                violence_percentage: 40,
                classification: "violent",
                duration_seconds: 10
              },
              violent_segments: [
                {
                  start_frame: 50,
                  start_time: 1.67,
                  end_frame: 150,
                  end_time: 5.0,
                  duration: 3.33,
                  avg_score: 0.75,
                  scores: [0.7, 0.8, 0.75]
                }
              ],
              results: [
                { frame: 1, time: 0.033, violence_detected: false, score: 0.2 },
                { frame: 50, time: 1.67, violence_detected: true, score: 0.7 }
              ]
            })
          };
        });

        if (response.ok) {
          const data = await response.json();
          setAnalysisData(data);
          if (data.keyframes && data.keyframes.length > 0) {
            setSelectedKeyframe(data.keyframes[0]);
          }
        } else {
          setError("Failed to load analysis data");
        }
      } catch (err) {
        console.error(err);
        setError("An error occurred while fetching the analysis data");
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [analysisId, videoUrl]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const renderClassificationBadge = () => {
    if (!analysisData) return null;

    const { classification } = analysisData.summary;

    let badgeColor = "";
    let badgeText = "";

    switch (classification) {
      case "violent":
        badgeColor = "bg-red-600";
        badgeText = "Violent Content";
        break;
      case "ambiguous":
        badgeColor = "bg-yellow-600";
        badgeText = "Potentially Violent Content";
        break;
      case "non-violent":
        badgeColor = "bg-green-600";
        badgeText = "No Violent Content";
        break;
    }

    return (
      <div className={`${badgeColor} text-white font-bold py-2 px-4 rounded-md inline-block`}>
        {badgeText}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-xl">Loading analysis data...</p>
        </div>
      </div>
    );
  }

  if (error || !analysisData) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-8">
        <div className="max-w-4xl mx-auto bg-gray-800 p-6 rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold text-red-500 mb-4">Error</h1>
          <p className="text-gray-300">{error || "Failed to load analysis data"}</p>
          <button
            onClick={() => window.history.back()}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold">Video Analysis Results</h1>
            <p className="text-gray-400 mt-1">Analysis ID: {analysisData.analysis_id}</p>
          </div>

          {renderClassificationBadge()}
        </div>

        <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden mb-8">
          <video
            src={`http://localhost:8000${analysisData.video_url}`}
            controls
            className="w-full"
            poster={analysisData.keyframes.length > 0 ? `http://localhost:8000${analysisData.keyframes[0]}` : undefined}
          />
        </div>

        <div className="mb-8">
          <div className="flex border-b border-gray-700">
            <button
              className={`py-2 px-4 ${activeTab === 'summary' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-400'}`}
              onClick={() => setActiveTab('summary')}
            >
              Summary
            </button>
            <button
              className={`py-2 px-4 ${activeTab === 'keyframes' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-400'}`}
              onClick={() => setActiveTab('keyframes')}
            >
              Key Frames ({analysisData.keyframes.length})
            </button>
            <button
              className={`py-2 px-4 ${activeTab === 'timeline' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-400'}`}
              onClick={() => setActiveTab('timeline')}
            >
              Timeline
            </button>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg shadow-lg mt-4">
            {activeTab === 'summary' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Violence Analysis Summary</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gray-750 p-4 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-300 mb-2">Video Information</h3>
                    <ul className="space-y-2">
                      <li className="flex justify-between">
                        <span className="text-gray-400">Duration:</span>
                        <span>{formatTime(analysisData.summary.duration_seconds)}</span>
                      </li>
                      <li className="flex justify-between">
                        <span className="text-gray-400">Total Frames:</span>
                        <span>{analysisData.summary.total_frames}</span>
                      </li>
                      <li className="flex justify-between">
                        <span className="text-gray-400">Classification:</span>
                        <span className={
                          analysisData.summary.classification === 'violent' ? 'text-red-500' :
                            analysisData.summary.classification === 'ambiguous' ? 'text-yellow-500' :
                              'text-green-500'
                        }>
                          {analysisData.summary.classification.charAt(0).toUpperCase() +
                            analysisData.summary.classification.slice(1)}
                        </span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-gray-750 p-4 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-300 mb-2">Violence Statistics</h3>

                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-1">
                        <span>Violence Detection</span>
                        <span>{Math.round(analysisData.summary.violence_percentage)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-full rounded-full ${analysisData.summary.violence_percentage > 40 ? 'bg-red-600' :
                              analysisData.summary.violence_percentage > 10 ? 'bg-yellow-600' :
                                'bg-green-600'
                            } w-[${Math.min(100, analysisData.summary.violence_percentage)}%]`}
                        ></div>
                      </div>
                    </div>

                    <ul className="space-y-2">
                      <li className="flex justify-between">
                        <span className="text-gray-400">Violent Frames:</span>
                        <span>{analysisData.summary.violence_frames}</span>
                      </li>
                      <li className="flex justify-between">
                        <span className="text-gray-400">Violent Segments:</span>
                        <span>{analysisData.violent_segments.length}</span>
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="mt-6">
                  <h3 className="text-lg font-medium text-gray-300 mb-3">Violent Segments</h3>

                  {analysisData.violent_segments.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="border-b border-gray-700">
                            <th className="py-3 px-4 text-left text-gray-300">#</th>
                            <th className="py-3 px-4 text-left text-gray-300">Start Time</th>
                            <th className="py-3 px-4 text-left text-gray-300">End Time</th>
                            <th className="py-3 px-4 text-left text-gray-300">Duration</th>
                            <th className="py-3 px-4 text-left text-gray-300">Avg. Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysisData.violent_segments.map((segment, index) => (
                            <tr key={index} className="border-b border-gray-800">
                              <td className="py-3 px-4">{index + 1}</td>
                              <td className="py-3 px-4">{formatTime(segment.start_time)}</td>
                              <td className="py-3 px-4">{formatTime(segment.end_time)}</td>
                              <td className="py-3 px-4">{segment.duration.toFixed(1)}s</td>
                              <td className="py-3 px-4">
                                <span className={`font-medium ${segment.avg_score > 0.7 ? 'text-red-500' :
                                    segment.avg_score > 0.5 ? 'text-yellow-500' :
                                      'text-blue-500'
                                  }`}>
                                  {(segment.avg_score * 100).toFixed(1)}%
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-gray-400">No violent segments detected in this video.</p>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'keyframes' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Key Violent Frames</h2>

                {analysisData.keyframes.length > 0 ? (
                  <div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 mb-6">
                      {analysisData.keyframes.map((frame, index) => (
                        <div
                          key={index}
                          className={`relative cursor-pointer rounded overflow-hidden ${selectedKeyframe === frame ? 'ring-2 ring-blue-500' : ''
                            }`}
                          onClick={() => setSelectedKeyframe(frame)}
                        >
                          <Image
                            src={`http://localhost:8000${frame}`}
                            width={96}
                            height={96}
                            alt={`Violence detection frame ${index + 1}`}
                            className="w-full h-24 object-cover"
                          />
                        </div>
                      ))}
                    </div>

                    {selectedKeyframe && (
                      <div className="bg-gray-900 p-2 rounded-lg">
                        <Image
                          src={`http://localhost:8000${selectedKeyframe}`}
                          width={640}
                          height={480}
                          alt="Selected violence frame"
                          className="w-full rounded"
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-400">No keyframes available for this analysis.</p>
                )}
              </div>
            )}

            {activeTab === 'timeline' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Violence Detection Timeline</h2>

                {analysisData.results.length > 0 ? (
                  <div>
                    <div className="relative h-16 mb-6">
                      <div className="absolute top-0 left-0 right-0 h-4 bg-gray-700 rounded-full overflow-hidden">
                        {analysisData.violent_segments.map((segment, index) => {
                          const startPercent = (segment.start_time / analysisData.summary.duration_seconds) * 100;
                          const widthPercent = (segment.duration / analysisData.summary.duration_seconds) * 100;

                          return (
                            <div
                              key={index}
                              className={`absolute h-full bg-red-600 left-[${startPercent}%] w-[${widthPercent}%]`}
                              title={`Violence at ${formatTime(segment.start_time)} - ${formatTime(segment.end_time)}`}
                            ></div>
                          );
                        })}
                      </div>

                      <div className="absolute top-6 left-0 right-0 h-8">
                        {[0, 0.25, 0.5, 0.75, 1].map((point) => {
                          const time = point * analysisData.summary.duration_seconds;
                          let leftClass = '';
                          if (point === 0) {
                            leftClass = 'left-0';
                          } else if (point === 0.25) {
                            leftClass = 'left-[25%]';
                          } else if (point === 0.5) {
                            leftClass = 'left-[50%]';
                          } else if (point === 0.75) {
                            leftClass = 'left-[75%]';
                          } else if (point === 1) {
                            leftClass = 'left-[100%]';
                          }

                          return (
                            <div
                              key={point}
                              className={`absolute transform -translate-x-1/2 ${leftClass}`}
                            >
                              <div className="h-2 w-0.5 bg-gray-600 mx-auto"></div>
                              <div className="text-xs text-gray-400 mt-1">{formatTime(time)}</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="border-b border-gray-700">
                            <th className="py-3 px-4 text-left text-gray-300">Frame #</th>
                            <th className="py-3 px-4 text-left text-gray-300">Timestamp</th>
                            <th className="py-3 px-4 text-left text-gray-300">Status</th>
                            <th className="py-3 px-4 text-left text-gray-300">Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysisData.results.map((result, index) => (
                            <tr key={index} className="border-b border-gray-800">
                              <td className="py-3 px-4">{result.frame}</td>
                              <td className="py-3 px-4">{formatTime(result.time)}</td>
                              <td className="py-3 px-4">
                                {result.violence_detected ? (
                                  <span className="inline-block bg-red-500/20 text-red-400 rounded px-2 py-1 text-xs">
                                    Violence Detected
                                  </span>
                                ) : (
                                  <span className="inline-block bg-green-500/20 text-green-400 rounded px-2 py-1 text-xs">
                                    No Violence
                                  </span>
                                )}
                              </td>
                              <td className="py-3 px-4">
                                <span className={`font-medium ${result.score > 0.7 ? 'text-red-500' :
                                    result.score > 0.5 ? 'text-yellow-500' :
                                      result.score > 0.3 ? 'text-orange-500' :
                                        'text-green-500'
                                  }`}>
                                  {(result.score * 100).toFixed(1)}%
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400">No timeline data available for this analysis.</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}