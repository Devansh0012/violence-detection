import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Replace this with your actual API call to the backend
    const response = await fetch('http://localhost:8000/api/ambiguous-samples')
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching ambiguous samples:', error)
    return NextResponse.json({ error: 'Failed to fetch samples' }, { status: 500 })
  }
}