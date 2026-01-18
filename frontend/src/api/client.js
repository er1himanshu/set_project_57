import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

// Maximum length for error messages before truncating to generic message
const MAX_ERROR_MESSAGE_LENGTH = 200;

/**
 * Extract a user-friendly error message from various error response formats
 * @param {import('axios').AxiosError} error - The error object from axios
 * @returns {string} A user-friendly error message
 */
const extractErrorMessage = (error) => {
  // If there's no response, it's likely a network error
  if (!error.response) {
    return "Network error. Please check your connection and try again.";
  }

  const { data, status } = error.response;

  // Handle non-JSON responses (e.g., HTML error pages)
  if (typeof data === 'string') {
    // If it's HTML or very long, return a generic message
    if (data.includes('<!DOCTYPE') || data.includes('<html') || data.length > MAX_ERROR_MESSAGE_LENGTH) {
      return `Server error (${status}). Please try again.`;
    }
    return data;
  }

  // Handle JSON responses
  if (data && typeof data === 'object') {
    // FastAPI validation errors: detail is an array of error objects
    if (Array.isArray(data.detail)) {
      // Extract and format validation errors
      const messages = data.detail.map(err => {
        if (err.msg) return err.msg;
        if (err.message) return err.message;
        return 'Invalid input';
      });
      return messages.join('; ');
    }
    
    // Standard error format: detail is a string
    if (data.detail && typeof data.detail === 'string') {
      return data.detail;
    }

    // Alternative error formats
    if (data.message) return data.message;
    if (data.error) return data.error;
  }

  // Fallback to status text or generic message
  return error.response.statusText || "Upload failed. Please try again.";
};

export const uploadImage = async (file, description = "") => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    if (description) {
      formData.append("description", description);
    }
    return await API.post("/upload", formData);
  } catch (error) {
    console.error("Upload error:", error);
    // Enhance error object with extracted message for easier consumption
    error.userMessage = extractErrorMessage(error);
    throw error;
  }
};

export const fetchResults = async () => {
  try {
    return await API.get("/results");
  } catch (error) {
    console.error("Fetch results error:", error);
    error.userMessage = extractErrorMessage(error);
    throw error;
  }
};

export const fetchResultDetail = async (id) => {
  try {
    return await API.get(`/results/${id}`);
  } catch (error) {
    console.error("Fetch result detail error:", error);
    error.userMessage = extractErrorMessage(error);
    throw error;
  }
};