import { useState } from "react";
import { useRouter } from "next/navigation";
import { login, signup } from "../../lib/api";
import { Eye, Shield, Lock, User, ArrowRight, AlertCircle } from "lucide-react";

export default function Auth() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [auth, setAuth] = useState({ username: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focusedField, setFocusedField] = useState<string | null>(null);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isLogin) {
        await login(auth.username, auth.password);
        router.push("/dashboard");
      } else {
        await signup(auth.username, auth.password);
        setIsLogin(true);
        setError(null);
        // Show success message briefly
        setTimeout(() => setError(null), 3000);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message || "An error occurred");
      } else {
        setError("An unknown error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-hidden relative">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/4 w-60 h-60 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* Floating Grid Pattern */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center px-6">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          
          {/* Left Side - Branding */}
          <div className="hidden lg:block space-y-8">
            <div className="space-y-6">
              {/* Logo and Brand */}
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                  <Eye className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-white">Third Eye</h1>
                  <p className="text-blue-300 text-sm">Security Intelligence</p>
                </div>
              </div>

              {/* Welcome Message */}
              <div className="space-y-4">
                <h2 className="text-4xl font-bold text-white leading-tight">
                  {isLogin ? "Welcome Back to" : "Join the Future of"}
                  <br />
                  <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                    Smart Security
                  </span>
                </h2>
                <p className="text-gray-300 text-lg max-w-md">
                  {isLogin 
                    ? "Access your advanced AI-powered security dashboard and protect what matters most."
                    : "Create your account and experience next-generation violence detection technology."
                  }
                </p>
              </div>

              {/* Feature Highlights */}
              <div className="space-y-4">
                <div className="flex items-center space-x-3 text-gray-300">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span>Real-time AI violence detection</span>
                </div>
                <div className="flex items-center space-x-3 text-gray-300">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse animation-delay-1000"></div>
                  <span>Advanced behavioral analytics</span>
                </div>
                <div className="flex items-center space-x-3 text-gray-300">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse animation-delay-2000"></div>
                  <span>Instant alert notifications</span>
                </div>
              </div>
            </div>

            {/* Decorative Elements */}
            <div className="relative">
              <div className="absolute -top-8 -left-8 w-32 h-32 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full opacity-10 animate-pulse"></div>
              <div className="absolute -bottom-8 -right-8 w-24 h-24 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full opacity-10 animate-bounce"></div>
            </div>
          </div>

          {/* Right Side - Auth Form */}
          <div className="w-full max-w-md mx-auto">
            <div className="bg-black bg-opacity-10 backdrop-blur-xl rounded-2xl p-8 border border-white border-opacity-20 shadow-2xl">
              {/* Form Header */}
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                  {isLogin ? (
                    <Shield className="w-8 h-8 text-white" />
                  ) : (
                    <User className="w-8 h-8 text-white" />
                  )}
                </div>
                <h3 className="text-2xl font-bold text-white mb-2">
                  {isLogin ? "Sign In" : "Create Account"}
                </h3>
                <p className="text-gray-300 text-sm">
                  {isLogin 
                    ? "Enter your credentials to access your dashboard" 
                    : "Join thousands of users protecting their spaces"
                  }
                </p>
              </div>

              {/* Auth Form */}
              <form onSubmit={handleAuth} className="space-y-6">
                {/* Username Field */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300 block">
                    Username
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <User className={`w-5 h-5 transition-colors ${
                        focusedField === 'username' ? 'text-blue-400' : 'text-gray-400'
                      }`} />
                    </div>
                    <input
                      type="text"
                      required
                      placeholder="Enter your username"
                      value={auth.username}
                      onChange={(e) => setAuth({ ...auth, username: e.target.value })}
                      onFocus={() => setFocusedField('username')}
                      onBlur={() => setFocusedField(null)}
                      className="w-full pl-10 pr-4 py-3 bg-black bg-opacity-10 border border-white border-opacity-20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    />
                  </div>
                </div>

                {/* Password Field */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300 block">
                    Password
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Lock className={`w-5 h-5 transition-colors ${
                        focusedField === 'password' ? 'text-blue-400' : 'text-gray-400'
                      }`} />
                    </div>
                    <input
                      type="password"
                      required
                      placeholder="Enter your password"
                      value={auth.password}
                      onChange={(e) => setAuth({ ...auth, password: e.target.value })}
                      onFocus={() => setFocusedField('password')}
                      onBlur={() => setFocusedField(null)}
                      className="w-full pl-10 pr-4 py-3 bg-black bg-opacity-10 border border-white border-opacity-20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    />
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <div className="flex items-center space-x-2 p-3 bg-red-500 bg-opacity-20 border border-red-500 border-opacity-30 rounded-xl">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                    <span className="text-red-300 text-sm">{error}</span>
                  </div>
                )}

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={loading}
                  className={`w-full group relative overflow-hidden rounded-xl py-3 px-6 font-semibold text-white transition-all duration-300 ${
                    loading 
                      ? "bg-gray-600 cursor-not-allowed" 
                      : "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transform hover:scale-105 shadow-lg hover:shadow-xl"
                  }`}
                >
                  <span className="relative flex items-center justify-center space-x-2">
                    {loading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <span>{isLogin ? "Sign In" : "Create Account"}</span>
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                      </>
                    )}
                  </span>
                </button>

                {/* Toggle Auth Mode */}
                <div className="text-center pt-4 border-t border-white border-opacity-10">
                  <span className="text-gray-300 text-sm">
                    {isLogin ? "Don't have an account?" : "Already have an account?"}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setIsLogin(!isLogin);
                      setError(null);
                      setAuth({ username: "", password: "" });
                    }}
                    className="ml-2 text-blue-400 hover:text-blue-300 font-medium text-sm transition-colors"
                  >
                    {isLogin ? "Sign up" : "Sign in"}
                  </button>
                </div>
              </form>

              {/* Additional Info */}
              <div className="mt-6 pt-6 border-t border-white border-opacity-10">
                <p className="text-xs text-gray-400 text-center">
                  By continuing, you agree to our Terms of Service and Privacy Policy
                </p>
              </div>
            </div>

            {/* Mobile Logo */}
            <div className="lg:hidden text-center mt-8">
              <div className="flex items-center justify-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <Eye className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-white">Third Eye Security</span>
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
        .animation-delay-1000 {
          animation-delay: 1s;
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
  );
}