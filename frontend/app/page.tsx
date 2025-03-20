import Link from 'next/link'
import Image from 'next/image'

export default function Home() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-cover bg-center bg-third-eye">
      <div className="text-center space-y-6 bg-gray-500 bg-opacity-75 p-10 rounded-lg">
        <h1 className="text-4xl font-bold">Third Eye: Streaming Security in Every Frame</h1>
        <div className="space-x-4">
          <Link href="/login" className="bg-blue-600 px-6 py-3 rounded-lg hover:bg-blue-700">
            Get Started
          </Link>
        </div>
      </div>
    </div>
  )
}