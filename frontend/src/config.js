// config.js - frontend configuration

// In Vite, environment variables MUST start with the VITE_ prefix to be
// exposed to our browser code. This lets us swap the backend URL out when we deploy.
// If the variable isn't set (like running locally), this correctly falls back to localhost.
export const API_BASE_URL = import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL : "http://localhost:5001";
