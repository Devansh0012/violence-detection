'use client'
import { useState, useEffect } from 'react'
import { fetchAmbiguousSamples, labelSample } from '../lib/api'
import Image from 'next/image'

interface Sample {
  id: number;
  image: string;
  confidence: number;
}

export default function ActiveLearningPanel() {
  const [samples, setSamples] = useState<Sample[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadSamples()
  }, [])

  const loadSamples = async () => {
    try {
      setLoading(true)
      const data = await fetchAmbiguousSamples()
      setSamples(data)
      setError(null)
    } catch (err) {
      setError('Failed to load ambiguous samples')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleLabel = async (id: number, isViolent: boolean) => {
    try {
      setSamples(prev => prev.filter(sample => sample.id !== id))
      await labelSample(id, isViolent)
    } catch (err) {
      console.error('Error labeling sample:', err)
      // Optionally show error to user
    }
  }

  if (loading) {
    return (
      <div className="bg-gray-800 p-6 rounded-xl">
        <div className="text-center py-4 text-gray-400">
          Loading samples...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-6 rounded-xl">
        <div className="text-center py-4 text-red-400">
          {error}
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 p-6 rounded-xl">
      <h2 className="text-xl font-semibold mb-4">Active Learning Interface</h2>

      <div className="space-y-4">
        {Array.isArray(samples) && samples.length > 0 ? (
          samples.map(sample => (
            <div key={sample.id} className="bg-gray-700 p-4 rounded-lg">
              <div className="flex items-center gap-4">
                <Image
                  src={sample.image}
                  width={96}
                  height={96}
                  className="w-24 h-24 object-cover rounded"
                  alt="Ambiguous sample"
                />
                <div className="flex-1">
                  <div className="text-sm mb-2">
                    Confidence: {Math.round((sample.confidence || 0) * 100)}%
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleLabel(sample.id, true)}
                      className="bg-red-500 hover:bg-red-600 px-3 py-1 rounded text-sm transition-colors"
                    >
                      Mark as Violent
                    </button>
                    <button
                      onClick={() => handleLabel(sample.id, false)}
                      className="bg-green-500 hover:bg-green-600 px-3 py-1 rounded text-sm transition-colors"
                    >
                      Mark as Safe
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-4 text-gray-400">
            No ambiguous samples to review
          </div>
        )}
      </div>
    </div>
  )
}