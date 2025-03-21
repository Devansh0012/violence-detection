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

// Update the fetchAmbiguousSamples function

export async function fetchAmbiguousSamples() {
  try {
    const response = await fetch('http://localhost:8000/ambiguous_samples');
    
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }
    
    const data = await response.json();
    
    // Ensure we always return an array, even if the API fails
    if (!data || !Array.isArray(data)) {
      console.error('API did not return an array:', data);
      return [];
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching ambiguous samples:', error);
    // Return empty array on error rather than undefined or null
    return [];
  }
}

export async function labelSample(id: number, isViolent: boolean) {
  try {
    const response = await fetch('http://localhost:8000/label_sample', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ id, is_violent: isViolent }),
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error labeling sample:', error);
    throw error;
  }
}
