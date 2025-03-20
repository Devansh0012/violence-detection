import axios from 'axios';

const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    headers: {
        'Content-Type': 'application/json',
    }
});

export const setAuthToken = (token: string) => {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
};

export const clearAuthToken = () => {
    delete api.defaults.headers.common['Authorization'];
    localStorage.removeItem('token');
    localStorage.removeItem('role');
};

export const login = async (username: string, password: string) => {
    try {
        const response = await api.post('/auth/login', { username, password });
        const { access_token, role } = response.data;
        
        // Store both token and role
        localStorage.setItem('token', access_token);
        localStorage.setItem('role', role || 'user'); // default to 'user' if no role provided
        
        setAuthToken(access_token);
        return response.data;
    } catch (error) {
        console.error('Login error:', error);
        throw error;
    }
};

export const signup = async (username: string, password: string) => {
    try {
        const response = await api.post('/auth/signup', { username, password });
        return response.data;
    } catch (error) {
        console.error('Signup error:', error);
        throw error;
    }
};

export const analyzeViolence = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await api.post('/analyze/violence', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error analyzing violence:', error);
        throw error;
    }
};

// Initialize token from localStorage if exists
if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token');
    if (token) {
        setAuthToken(token);
    }
}

export default api;

interface Sample {
  id: number;
  image: string;
  confidence: number;
}

export async function fetchAmbiguousSamples(): Promise<Sample[]> {
  try {
    const response = await fetch('/api/ambiguous-samples');
    if (!response.ok) {
      throw new Error('Failed to fetch ambiguous samples');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching ambiguous samples:', error);
    return [];
  }
}

export async function labelSample(id: number, isViolent: boolean): Promise<void> {
  try {
    const response = await fetch('/api/label-sample', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        id,
        isViolent,
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to label sample');
    }
  } catch (error) {
    console.error('Error labeling sample:', error);
    throw error;
  }
}
