'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import LiveFeed from '@/components/LiveFeed'
import VideoUpload from '@/components/VideoUpload'
import AlertPanel from '@/components/AlertPanel'
import ActiveLearningPanel from '@/components/ActiveLearningPanel'
import { clearAuthToken } from '@/lib/api'
import { 
  Eye, 
  Shield, 
  LogOut, 
  Activity, 
  Camera, 
  Upload, 
  AlertTriangle, 
  Settings,
  Users,
  BarChart3,
  Zap,
  Clock
} from 'lucide-react'

export default function DashboardPage() {
  const [alerts, setAlerts] = useState<string[]>([])
  const [role, setRole] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('live')
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
    setAlerts(prev => [message, ...prev].slice(0, 20))
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading Dashboard...</p>
        </div>
      </div>
    )
  }

  const stats = [
    {
      icon: Activity,
      label: 'System Status',
      value: 'Active',
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
      borderColor: 'border-green-500/50'
    },
    {
      icon: AlertTriangle,
      label: 'Active Alerts',
      value: alerts.length.toString(),
      color: 'text-red-400',
      bgColor: 'bg-red-500/20',
      borderColor: 'border-red-500/50'
    },
    {
      icon: Users,
      label: 'Monitored Areas',
      value: '3',
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
      borderColor: 'border-blue-500/50'
    },
    {
      icon: Clock,
      label: 'Uptime',
      value: '24/7',
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
      borderColor: 'border-purple-500/50'
    }
  ]

  const navigationTabs = [
    { id: 'live', label: 'Live Feed', icon: Camera },
    { id: 'upload', label: 'Video Analysis', icon: Upload },
    { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
    ...(role === 'admin' ? [{ id: 'learning', label: 'AI Training', icon: Settings }] : [])
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-hidden relative">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-blob animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/4 w-60 h-60 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-blob animation-delay-4000"></div>
      </div>

      {/* Floating Grid Pattern */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen">
        {/* Header */}
        <header className="backdrop-blur-xl bg-black bg-opacity-20 border-b border-white border-opacity-10 sticky top-0 z-20">
          <div className="px-6 py-4">
            <div className="flex justify-between items-center">
              {/* Logo and Brand */}
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                  <Eye className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">Third Eye</h1>
                  <p className="text-blue-300 text-sm">Security Dashboard</p>
                </div>
              </div>

              {/* User Info and Actions */}
              <div className="flex items-center space-x-6">
                {/* Status Indicator */}
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-gray-300">System Online</span>
                </div>

                {/* User Badge */}
                <div className="bg-white bg-opacity-10 backdrop-blur-sm px-4 py-2 rounded-lg border border-white border-opacity-20">
                  <div className="flex items-center space-x-2">
                    <Shield className="w-4 h-4 text-blue-400" />
                    <span className="text-white text-sm font-medium">
                      {role === 'admin' ? 'Administrator' : 'Security Officer'}
                    </span>
                  </div>
                </div>

                {/* Logout Button */}
                <button
                  onClick={handleLogout}
                  className="group bg-red-500 bg-opacity-20 hover:bg-opacity-30 border border-red-500 border-opacity-50 px-4 py-2 rounded-lg text-red-400 hover:text-red-300 transition-all duration-200 flex items-center space-x-2"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="text-sm font-medium">Logout</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Stats Overview */}
        <div className="px-6 py-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {stats.map((stat, index) => (
              <div
                key={index}
                className={`${stat.bgColor} backdrop-blur-sm rounded-xl p-4 border ${stat.borderColor} hover:scale-105 transition-transform duration-200`}
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-10 h-10 ${stat.bgColor} rounded-lg flex items-center justify-center`}>
                    <stat.icon className={`w-5 h-5 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-gray-300 text-xs font-medium">{stat.label}</p>
                    <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Navigation Tabs */}
          <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 mb-6">
            <div className="flex overflow-x-auto scrollbar-hide">
              {navigationTabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-6 py-4 font-medium transition-all duration-200 whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'text-white bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl m-2'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <tab.icon className="w-5 h-5" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Main Content Area */}
          <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
            {/* Primary Content */}
            <div className="xl:col-span-3 space-y-6">
              {activeTab === 'live' && (
                <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 overflow-hidden">
                  <div className="p-6 border-b border-white border-opacity-10">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                        <Camera className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold text-white">Live Security Feed</h2>
                        <p className="text-gray-300 text-sm">Real-time violence detection and monitoring</p>
                      </div>
                    </div>
                  </div>
                  <div className="p-6">
                    <LiveFeed onAlert={handleAlert} />
                  </div>
                </div>
              )}

              {activeTab === 'upload' && (
                <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 overflow-hidden">
                  <div className="p-6 border-b border-white border-opacity-10">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                        <Upload className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold text-white">Video Analysis</h2>
                        <p className="text-gray-300 text-sm">Upload and analyze recorded videos for violence detection</p>
                      </div>
                    </div>
                  </div>
                  <div className="p-6">
                    <VideoUpload />
                  </div>
                </div>
              )}

              {activeTab === 'alerts' && (
                <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 overflow-hidden">
                  <div className="p-6 border-b border-white border-opacity-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gradient-to-r from-red-500 to-orange-600 rounded-lg flex items-center justify-center">
                          <AlertTriangle className="w-4 h-4 text-white" />
                        </div>
                        <div>
                          <h2 className="text-xl font-bold text-white">Security Alerts</h2>
                          <p className="text-gray-300 text-sm">Monitoring threats and security incidents</p>
                        </div>
                      </div>
                      {alerts.length > 0 && (
                        <span className="bg-red-500 text-white text-xs font-bold px-3 py-1 rounded-full">
                          {alerts.length} Active
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="p-6">
                    <AlertPanel alerts={alerts} />
                  </div>
                </div>
              )}

              {activeTab === 'learning' && role === 'admin' && (
                <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 overflow-hidden">
                  <div className="p-6 border-b border-white border-opacity-10">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
                        <Settings className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold text-white">AI Training & Learning</h2>
                        <p className="text-gray-300 text-sm">Improve model accuracy through active learning</p>
                      </div>
                    </div>
                  </div>
                  <div className="p-6">
                    <ActiveLearningPanel />
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Quick Stats */}
              <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 p-6">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  <span>Quick Stats</span>
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">Detection Accuracy</span>
                    <span className="text-green-400 font-semibold">99.2%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div className="bg-green-400 h-2 rounded-full w-[99%]"></div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">Response Time</span>
                    <span className="text-blue-400 font-semibold">&lt;50ms</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div className="bg-blue-400 h-2 rounded-full w-[95%]"></div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">System Health</span>
                    <span className="text-purple-400 font-semibold">Excellent</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div className="bg-purple-400 h-2 rounded-full w-[98%]"></div>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 p-6">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  <span>Recent Activity</span>
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-gray-300 text-sm">System started monitoring</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span className="text-gray-300 text-sm">Camera feed connected</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    <span className="text-gray-300 text-sm">AI model initialized</span>
                  </div>
                  {alerts.length > 0 && (
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
                      <span className="text-red-400 text-sm font-medium">
                        {alerts.length} security alert{alerts.length > 1 ? 's' : ''} detected
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* System Information */}
              <div className="bg-black bg-opacity-20 backdrop-blur-xl rounded-2xl border border-white border-opacity-10 p-6">
                <h3 className="text-lg font-bold text-white mb-4">System Info</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Version</span>
                    <span className="text-white">v2.1.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">AI Model</span>
                    <span className="text-white">Enhanced CNN</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Last Update</span>
                    <span className="text-white">2 hrs ago</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Custom Styles */}
      <style jsx>{`
        @keyframes blob {
          0%, 100% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .bg-grid-pattern {
          background-image: 
            linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
          background-size: 20px 20px;
        }
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  )
}