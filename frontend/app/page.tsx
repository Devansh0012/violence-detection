'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { Shield, Eye, Zap, ChevronRight, Play } from 'lucide-react'

export default function Home() {
  const [isVisible, setIsVisible] = useState(false)
  const [currentSlide, setCurrentSlide] = useState(0)

  useEffect(() => {
    setIsVisible(true)
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % 3)
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  const features = [
    {
      icon: Eye,
      title: "Real-time Detection",
      description: "AI-powered violence detection in live video streams with millisecond response times"
    },
    {
      icon: Shield,
      title: "Advanced Security",
      description: "Multi-layered protection with pose analysis, crowd behavior monitoring, and threat assessment"
    },
    {
      icon: Zap,
      title: "Instant Alerts",
      description: "Immediate notifications and automated response systems for critical situations"
    }
  ]

  const slides = [
    {
      title: "Next-Gen Security",
      subtitle: "AI-Powered Violence Detection",
      description: "Protect what matters most with cutting-edge artificial intelligence"
    },
    {
      title: "Real-Time Monitoring",
      subtitle: "24/7 Surveillance System",
      description: "Continuous protection with instant threat detection and response"
    },
    {
      title: "Smart Analytics",
      subtitle: "Behavioral Pattern Recognition",
      description: "Advanced algorithms that learn and adapt to emerging security threats"
    }
  ]

  return (
    <div className="h-full bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* Floating Grid Pattern */}
      <div className="relative inset-0 bg-grid-pattern opacity-5"></div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Navigation */}
        <nav className="p-6 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white">Third Eye</span>
          </div>
          <div className="space-x-4">
            <Link href="/login" className="text-gray-300 hover:text-white transition-colors">
              Login
            </Link>
            <Link href="/register" className="bg-blue bg-opacity-10 backdrop-blur-sm px-4 py-2 rounded-lg text-white hover:bg-opacity-20 transition-all">
              Sign Up
            </Link>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div className={`transform transition-all duration-1000 ${isVisible ? 'translate-x-0 opacity-100' : '-translate-x-10 opacity-0'}`}>
              <div className="space-y-6">
                <div className="inline-flex items-center space-x-2 bg-blue-500 bg-opacity-20 backdrop-blur-sm px-4 py-2 rounded-full text-blue-300 text-sm">
                  <Zap className="w-4 h-4" />
                  <span>AI-Powered Security</span>
                </div>
                
                <h1 className="text-5xl lg:text-7xl font-bold text-white leading-tight">
                  <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                    Third Eye
                  </span>
                  <br />
                  <span className="text-gray-200">Security</span>
                </h1>
                
                <p className="text-xl text-gray-300 leading-relaxed max-w-lg">
                  Revolutionary AI-powered violence detection system that provides real-time monitoring, 
                  instant alerts, and comprehensive security analytics.
                </p>

                <div className="flex flex-col sm:flex-row gap-4 pt-4">
                  <Link href="/login" className="group bg-gradient-to-r from-blue-500 to-purple-600 px-8 py-4 rounded-xl text-white font-semibold hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg hover:shadow-xl">
                    <span className="flex items-center space-x-2">
                      <span>Get Started</span>
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </span>
                  </Link>
                  
                  <button className="group bg-black bg-opacity-10 backdrop-blur-sm px-8 py-4 rounded-xl text-white font-semibold hover:bg-opacity-20 transition-all transform hover:scale-105 border border-white border-opacity-20">
                    <span className="flex items-center space-x-2">
                      <Play className="w-4 h-4" />
                      <span>Watch Demo</span>
                    </span>
                  </button>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-6 pt-8">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-white">99.9%</div>
                    <div className="text-sm text-gray-400">Accuracy</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-white">&lt;100ms</div>
                    <div className="text-sm text-gray-400">Response Time</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-white">24/7</div>
                    <div className="text-sm text-gray-400">Monitoring</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Content - Dynamic Slides */}
            <div className={`transform transition-all duration-1000 delay-300 ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-10 opacity-0'}`}>
              <div className="relative">
                {/* Main Card */}
                <div className="bg-black bg-opacity-10 backdrop-blur-lg rounded-2xl p-8 border border-white border-opacity-20 shadow-2xl">
                  <div className="space-y-6">
                    {/* Slide Content */}
                    <div className="h-40 flex flex-col justify-center">
                      <h3 className="text-2xl font-bold text-white mb-2">
                        {slides[currentSlide].title}
                      </h3>
                      <p className="text-blue-300 font-semibold mb-3">
                        {slides[currentSlide].subtitle}
                      </p>
                      <p className="text-gray-300">
                        {slides[currentSlide].description}
                      </p>
                    </div>

                    {/* Slide Indicators */}
                    <div className="flex space-x-2">
                      {slides.map((_, index) => (
                        <div
                          key={index}
                          className={`w-2 h-2 rounded-full transition-all ${
                            index === currentSlide ? 'bg-blue-400 w-8' : 'bg-gray-500'
                          }`}
                        />
                      ))}
                    </div>

                    {/* Mock Interface */}
                    <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-green-400">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                        <span className="text-sm">System Active</span>
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-gray-300">
                          <span>Detection Accuracy</span>
                          <span>99.9%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-1">
                          <div className="bg-green-400 h-1 rounded-full w-[99%]"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Floating Elements */}
                <div className="absolute -top-4 -right-4 w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full opacity-20 animate-pulse"></div>
                <div className="absolute -bottom-4 -left-4 w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full opacity-20 animate-bounce"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="py-16 px-6">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-white mb-4">Why Choose Third Eye?</h2>
              <p className="text-gray-300 max-w-2xl mx-auto">
                Advanced AI technology combined with intuitive design for comprehensive security solutions
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className={`transform transition-all duration-500 delay-${index * 200} ${
                    isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
                  }`}
                >
                  <div className="bg-black bg-opacity-5 backdrop-blur-sm rounded-xl p-6 border border-white border-opacity-10 hover:bg-opacity-10 transition-all group hover:scale-105">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                    <p className="text-gray-300">{feature.description}</p>
                  </div>
                </div>
              ))}
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
      `}</style>
    </div>
  )
}