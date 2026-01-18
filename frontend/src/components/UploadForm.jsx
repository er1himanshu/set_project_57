import { useState, useRef, useEffect } from "react";
import { uploadImage, fetchResultDetail } from "../api/client";
import { useNavigate } from "react-router-dom";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [description, setDescription] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resultId, setResultId] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      // Create preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image file");
      return;
    }
    
    setLoading(true);
    setError("");
    setMessage("");
    
    try {
      const res = await uploadImage(file, description);
      setMessage(`Analysis complete! View detailed results below.`);
      setResultId(res.data.result_id);
      // Don't clear form yet so user can see the preview with results
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setDescription("");
    setMessage("");
    setError("");
    setResultId(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleViewAllResults = () => {
    navigate("/results");
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 px-8 py-6">
        <h2 className="text-2xl font-bold text-white mb-2">Upload Product Image</h2>
        <p className="text-indigo-100">Upload your product image and description for comprehensive quality analysis</p>
      </div>
      
      <div className="p-8">
        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Area */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Product Image *
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-indigo-400 transition-colors">
              {!preview ? (
                <div>
                  <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div className="text-gray-600 mb-2">
                    <label htmlFor="file-upload" className="relative cursor-pointer rounded-md font-semibold text-indigo-600 hover:text-indigo-500">
                      <span>Click to upload</span>
                      <input
                        id="file-upload"
                        ref={fileInputRef}
                        type="file"
                        className="sr-only"
                        accept="image/*"
                        onChange={handleFileChange}
                      />
                    </label>
                    <span className="pl-1">or drag and drop</span>
                  </div>
                  <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                </div>
              ) : (
                <div className="relative">
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-lg" />
                  <button
                    onClick={() => {
                      setFile(null);
                      setPreview(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                    className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
            
            {file && (
              <div className="mt-4 p-3 bg-indigo-50 rounded-lg">
                <p className="text-sm text-gray-700">
                  <span className="font-semibold">Selected:</span> {file.name}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Size: {(file.size / 1024).toFixed(2)} KB
                </p>
              </div>
            )}
          </div>

          {/* Description Area */}
          <div>
            <label htmlFor="description" className="block text-sm font-semibold text-gray-700 mb-3">
              Product Description (Optional)
            </label>
            <textarea
              id="description"
              rows="6"
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
              placeholder="E.g., Red leather handbag with gold hardware, 12 inch width..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            ></textarea>
            <p className="mt-2 text-xs text-gray-500">
              Provide a description to check consistency between your image and text
            </p>

            {/* Action Buttons */}
            <div className="mt-6 space-y-3">
              <button
                onClick={handleUpload}
                disabled={loading || !file}
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-xl font-semibold hover:from-indigo-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing...
                  </span>
                ) : (
                  "Analyze Image Quality"
                )}
              </button>
              
              {resultId && (
                <button
                  onClick={handleReset}
                  className="w-full bg-white text-gray-700 px-6 py-3 rounded-xl font-semibold border-2 border-gray-300 hover:border-indigo-500 hover:text-indigo-600 transition-all"
                >
                  Upload Another Image
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Messages */}
        {message && (
          <div className="mt-6 p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <p className="text-green-800 font-medium">{message}</p>
            </div>
            {resultId && (
              <button
                onClick={handleViewAllResults}
                className="mt-3 text-sm text-green-700 hover:text-green-800 font-semibold underline"
              >
                View All Results →
              </button>
            )}
          </div>
        )}
        
        {error && (
          <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-red-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <p className="text-red-800 font-medium">{error}</p>
            </div>
          </div>
        )}

        {/* Show detailed results if analysis is complete */}
        {resultId && <ResultsDisplay resultId={resultId} />}
      </div>
    </div>
  );
}

function ResultsDisplay({ resultId }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (resultId) {
      fetchResultDetail(resultId)
        .then((res) => {
          setResult(res.data);
          setLoading(false);
        })
        .catch((err) => {
          console.error("Failed to fetch result details:", err);
          setLoading(false);
        });
    }
  }, [resultId]);

  if (loading) {
    return (
      <div className="mt-8 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        <p className="mt-2 text-gray-600">Loading results...</p>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const suggestions = result.improvement_suggestions ? result.improvement_suggestions.split(';').map(s => s.trim()).filter(s => s) : [];

  return (
    <div className="mt-8 border-t-2 border-gray-200 pt-8">
      <h3 className="text-2xl font-bold text-gray-800 mb-6">Analysis Results</h3>
      
      {/* Overall Status */}
      <div className={`p-6 rounded-xl mb-6 ${result.passed ? 'bg-green-50 border-2 border-green-200' : 'bg-red-50 border-2 border-red-200'}`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className={`text-xl font-bold ${result.passed ? 'text-green-800' : 'text-red-800'}`}>
              {result.passed ? '✓ PASSED' : '✗ FAILED'}
            </h4>
            <p className={`mt-1 ${result.passed ? 'text-green-700' : 'text-red-700'}`}>
              {result.reason}
            </p>
          </div>
          {result.passed ? (
            <svg className="w-16 h-16 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ) : (
            <svg className="w-16 h-16 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>
      </div>

      {/* Quality Checklist */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white border rounded-xl p-6">
          <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            Quality Metrics
          </h4>
          <div className="space-y-3">
            <MetricRow label="Resolution" value={`${result.width} × ${result.height}`} passed={result.width >= 1000 && result.height >= 1000} />
            <MetricRow label="Aspect Ratio" value={result.aspect_ratio?.toFixed(2)} passed={true} />
            <MetricRow label="Blur Score" value={result.blur_score?.toFixed(2)} passed={result.blur_score >= 100} />
            <MetricRow label="Sharpness" value={result.sharpness_score?.toFixed(2)} passed={result.sharpness_score >= 50} />
            <MetricRow label="Brightness" value={result.brightness_score?.toFixed(2)} passed={result.brightness_score >= 60 && result.brightness_score <= 200} />
            <MetricRow label="Contrast" value={result.contrast_score?.toFixed(2)} passed={true} />
          </div>
        </div>

        <div className="bg-white border rounded-xl p-6">
          <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
            </svg>
            Ecommerce Standards
          </h4>
          <div className="space-y-3">
            <MetricRow label="Background Quality" value={(result.background_score * 100)?.toFixed(0) + '%'} passed={result.background_score >= 0.7} />
            <MetricRow label="Watermark Detection" value={result.has_watermark ? 'Detected' : 'None'} passed={!result.has_watermark} />
            <MetricRow 
              label="Description Match" 
              value={result.description_consistency || 'N/A'} 
              passed={result.description_consistency === 'Consistent' || result.description_consistency === 'No description provided'} 
            />
          </div>
        </div>
      </div>

      {/* Improvement Suggestions */}
      {suggestions.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h4 className="font-semibold text-blue-900 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Improvement Suggestions
          </h4>
          <ul className="space-y-2">
            {suggestions.map((suggestion, idx) => (
              <li key={idx} className="flex items-start text-blue-800">
                <span className="mr-2 text-blue-600">•</span>
                <span>{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value, passed }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <span className="text-sm text-gray-600">{label}</span>
      <div className="flex items-center">
        <span className="text-sm font-semibold text-gray-800 mr-2">{value}</span>
        {passed ? (
          <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        ) : (
          <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        )}
      </div>
    </div>
  );
}