import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import ResultsTable from "../components/ResultsTable";
import { fetchResults } from "../api/client";

export default function Results() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchResults()
      .then((res) => {
        setResults(res.data);
        setLoading(false);
      })
      .catch((err) => {
        setError("Failed to load results");
        setLoading(false);
        console.error(err);
      });
  }, []);

  const passedCount = results.filter(r => r.passed).length;
  const failedCount = results.length - passedCount;

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
      <Navbar />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-10 animate-fade-in">
          <h2 className="text-5xl font-bold mb-3">
            <span className="gradient-text">Analysis Results</span>
          </h2>
          <p className="text-gray-600 text-xl">View all your product image quality assessments</p>
        </div>

        {/* Statistics Cards */}
        {!loading && !error && results.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 animate-slide-in">
            <div className="card p-8 border-l-4 border-primary-500 hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm font-bold uppercase tracking-wide">Total Analyzed</p>
                  <p className="text-4xl font-bold text-gray-900 mt-3">{results.length}</p>
                </div>
                <div className="w-16 h-16 bg-primary-100 rounded-xl flex items-center justify-center shadow-soft">
                  <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="card p-8 border-l-4 border-success-500 hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm font-bold uppercase tracking-wide">Passed</p>
                  <p className="text-4xl font-bold text-success-600 mt-3">{passedCount}</p>
                </div>
                <div className="w-16 h-16 bg-success-100 rounded-xl flex items-center justify-center shadow-soft">
                  <svg className="w-8 h-8 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="card p-8 border-l-4 border-danger-500 hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm font-bold uppercase tracking-wide">Failed</p>
                  <p className="text-4xl font-bold text-danger-600 mt-3">{failedCount}</p>
                </div>
                <div className="w-16 h-16 bg-danger-100 rounded-xl flex items-center justify-center shadow-soft">
                  <svg className="w-8 h-8 text-danger-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="card p-16 text-center animate-fade-in">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-b-4 border-primary-600 mb-6"></div>
            <p className="text-gray-600 font-semibold text-lg">Loading results...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-danger-50 border-2 border-danger-200 rounded-2xl p-10 text-center shadow-soft animate-slide-in">
            <svg className="w-16 h-16 text-danger-500 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-danger-800 font-bold text-xl">{error}</p>
          </div>
        )}

        {/* Results Table */}
        {!loading && !error && <ResultsTable results={results} />}
      </div>
    </div>
  );
}