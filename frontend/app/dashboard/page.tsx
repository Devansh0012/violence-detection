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

  if (isLoading) {
    return <div>Loading...</div>
  }

  return (
    <div className="space-y-8 py-8">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">CCTV Monitoring Dashboard</h1>
        <button
          onClick={handleLogout}
          className="bg-red-600 px-4 py-2 rounded-lg hover:bg-red-700"
        >
          Logout
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          <LiveFeed onAlert={(message: string) => setAlerts((prevAlerts) => [...prevAlerts, message])} />
          <VideoUpload />
        </div>

        <div className="lg:col-span-1 space-y-8">
          <AlertPanel alerts={alerts} />
          {role === 'admin' && <ActiveLearningPanel />}
        </div>
      </div>
    </div>
  )
}