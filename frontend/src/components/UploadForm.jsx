import { useState, useRef } from "react";
import { uploadImage } from "../api/client";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file");
      return;
    }
    
    setLoading(true);
    setError("");
    setMessage("");
    
    try {
      const res = await uploadImage(file);
      setMessage(`Uploaded! Result ID: ${res.data.result_id}`);
      setFile(null);
      // Reset the file input element
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded shadow">
      <h2 className="text-xl font-bold mb-4">Upload Image for Analysis</h2>
      <input 
        ref={fileInputRef}
        type="file" 
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])} 
      />
      <button
        className="ml-4 bg-blue-600 text-white px-4 py-2 rounded disabled:bg-gray-400"
        onClick={handleUpload}
        disabled={loading}
      >
        {loading ? "Uploading..." : "Upload"}
      </button>
      {message && <p className="mt-3 text-green-600">{message}</p>}
      {error && <p className="mt-3 text-red-600">{error}</p>}
    </div>
  );
}