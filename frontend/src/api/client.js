import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

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
    throw error;
  }
};

export const explainImage = async (file, description) => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("description", description);
    return await API.post("/explain", formData);
  } catch (error) {
    console.error("Explain error:", error);
    throw error;
  }
};

export const fetchResults = async () => {
  try {
    return await API.get("/results");
  } catch (error) {
    console.error("Fetch results error:", error);
    throw error;
  }
};

export const fetchResultDetail = async (id) => {
  try {
    return await API.get(`/results/${id}`);
  } catch (error) {
    console.error("Fetch result detail error:", error);
    throw error;
  }
};