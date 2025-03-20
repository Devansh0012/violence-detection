'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import LiveFeed from '@/components/LiveFeed'
import VideoUpload from '@/components/VideoUpload'
import AlertPanel from '@/components/AlertPanel'
import ActiveLearningPanel from '@/components/ActiveLearningPanel'
import { clearAuthToken } from '@/lib/api'

export default function DashboardPage() {
  const [alerts, setAlerts] = useState<string[]>([])
  const [role, setRole] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('token')
      const userRole = localStorage.getItem('role')
      
      if (!token || !userRole) {
        clearAuthToken()
        router.push('/login')
        return
      }

      setRole(userRole)
      setIsLoading(false)
    }

    checkAuth()
  }, [router])

  const handleLogout = () => {
    clearAuthToken()
    router.push('/login')
  }

  const handleAlert = (message: string) => {
    setAlerts(prev => [message, ...prev].slice(0, 20)) // Keep only most recent 20 alerts
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="py-6 px-4 sm:px-6 lg:px-8">
      <header className="bg-gray-900 p-4 rounded-xl mb-6 shadow-lg">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-white">SafeVision</h1>
            <span className="ml-2 text-gray-400">AI-Powered Violence Detection System</span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-gray-300">{role === 'admin' ? 'Admin Dashboard' : 'User Dashboard'}</span>
            <button
              onClick={handleLogout}
              className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-white transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <LiveFeed onAlert={handleAlert} />
          <VideoUpload />
        </div>

        <div className="space-y-6">
          <AlertPanel alerts={alerts} />
          {role === 'admin' && <ActiveLearningPanel />}
        </div>
      </div>
    </div>
  )
}