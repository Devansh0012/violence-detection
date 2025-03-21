import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    // Set NEXT_PUBLIC_API_URL to your Render URL in deployment.
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
