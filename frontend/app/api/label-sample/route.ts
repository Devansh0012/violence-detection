import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const body = await request.json()
    
    // Replace this with your actual API call to the backend
    const response = await fetch('http://localhost:8000/api/label-sample', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error labeling sample:', error)
    return NextResponse.json({ error: 'Failed to label sample' }, { status: 500 })
  }
}