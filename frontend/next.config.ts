import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    // Set NEXT_PUBLIC_API_URL to your Render URL in deployment.
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    NEXT_PUbLIC_WS_URL:
      process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000",
  },
  // Add images configuration using remotePatterns
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'violence-detection-dev.s3.ap-south-1.amazonaws.com',
        pathname: '/annotated_videos/**',
      },
      {
        protocol: 'http',
        hostname: 'localhost',
      },
      {
        protocol: 'http',
        hostname: '127.0.0.1',
      }
    ],
    formats: ['image/webp'],
  },
};

export default nextConfig;