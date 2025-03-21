import { useState } from "react";
import { useRouter } from "next/navigation";
import { login, signup } from "../../lib/api";

export default function Auth() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [auth, setAuth] = useState({ username: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 animate-gradient">
      <div className="max-w-md w-full p-8 bg-white rounded-xl shadow-2xl transform transition duration-500 hover:scale-105">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 animate-bounce">
            {isLogin ? "Welcome Back" : "Join Us"}
          </h1>
          <p className="text-gray-600 mt-2">
            {isLogin ? "Sign in to access your dashboard" : "Create a new account"}
          </p>
        </div>
        <form onSubmit={handleAuth} className="space-y-6">
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <input
                type="text"
                required
                placeholder="Username"
                value={auth.username}
                onChange={(e) => setAuth({ ...auth, username: e.target.value })}
                className="appearance-none rounded-t-md block w-full px-4 py-3 border border-gray-300 placeholder-gray-500 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 transition duration-300"
              />
            </div>
            <div>
              <input
                type="password"
                required
                placeholder="Password"
                value={auth.password}
                onChange={(e) => setAuth({ ...auth, password: e.target.value })}
                className="appearance-none rounded-b-md block w-full px-4 py-3 border border-gray-300 placeholder-gray-500 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 transition duration-300"
              />
            </div>
          </div>

          {error && (
            <div className="text-center text-red-500 text-sm">
              {error}
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 px-4 rounded-md font-medium text-white transition duration-300 ${
                loading ? "bg-gray-400" : "bg-indigo-600 hover:bg-indigo-700"
              }`}
            >
              {loading ? "Processing..." : isLogin ? "Sign In" : "Sign Up"}
            </button>
          </div>
        </form>

        <div className="text-center mt-6">
          <button
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="text-sm text-indigo-600 hover:text-indigo-500 transition duration-300"
          >
            {isLogin ? "Need an account? Sign up" : "Already have an account? Sign in"}
          </button>
        </div>
      </div>
    </div>
  );
}