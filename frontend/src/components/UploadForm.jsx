import { useState } from "react";
import { uploadImage } from "../api/client";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    const res = await uploadImage(file);
    setMessage(`Uploaded! Result ID: ${res.data.result_id}`);
  };

  return (
    <div className="p-6 bg-white rounded shadow">
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button
        className="ml-4 bg-blue-600 text-white px-4 py-2 rounded"
        onClick={handleUpload}
      >
        Upload
      </button>
      {message && <p className="mt-3 text-green-600">{message}</p>}
    </div>
  );
}