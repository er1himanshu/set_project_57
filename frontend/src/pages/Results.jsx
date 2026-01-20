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
  
  // Calculate average similarity score, filtering out null values
  const validScores = results.filter(r => r.similarity_score !== null && r.similarity_score !== undefined);
  const averageScore = validScores.length > 0
    ? validScores.reduce((sum, r) => sum + r.similarity_score, 0) / validScores.length
    : 0;

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-10 animate-fade-in">
          <h2 className="text-5xl font-bold mb-3 tracking-tight">
            <span className="gradient-text">Analysis Results</span>
          </h2>
          <p className="text-slate-600 text-lg">Comprehensive quality assessments for all product images</p>
        </div>

        {/* Statistics Cards */}
        {!loading && !error && results.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
            <div className="card p-6 border-l-4 border-primary-600 hover:shadow-xl transition-all duration-300">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-600 text-xs font-bold uppercase tracking-wider mb-1">Total Analyzed</p>
                  <p className="text-4xl font-bold text-slate-900 mt-1">{results.length}</p>
                  <p className="text-sm text-slate-500 mt-2">Images processed</p>
                </div>
                <div className="w-14 h-14 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center shadow-lg">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="card p-6 border-l-4 border-success-600 hover:shadow-xl transition-all duration-300">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-600 text-xs font-bold uppercase tracking-wider mb-1">Passed Quality</p>
                  <p className="text-4xl font-bold text-success-600 mt-1">{passedCount}</p>
                  <p className="text-sm text-slate-500 mt-2">{results.length > 0 ? ((passedCount / results.length) * 100).toFixed(1) : 0}% success rate</p>
                </div>
                <div className="w-14 h-14 bg-gradient-to-br from-success-500 to-success-600 rounded-lg flex items-center justify-center shadow-lg">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="card p-6 border-l-4 border-danger-600 hover:shadow-xl transition-all duration-300">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-600 text-xs font-bold uppercase tracking-wider mb-1">Need Improvement</p>
                  <p className="text-4xl font-bold text-danger-600 mt-1">{failedCount}</p>
                  <p className="text-sm text-slate-500 mt-2">{results.length > 0 ? ((failedCount / results.length) * 100).toFixed(1) : 0}% need attention</p>
                </div>
                <div className="w-14 h-14 bg-gradient-to-br from-danger-500 to-danger-600 rounded-lg flex items-center justify-center shadow-lg">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
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
            <p className="text-slate-600 font-semibold text-lg">Loading results...</p>
            <p className="text-slate-500 text-sm mt-2">Please wait while we fetch your analysis data</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="card p-10 text-center shadow-medium animate-fade-in border-l-4 border-danger-500">
            <svg className="w-16 h-16 text-danger-500 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-danger-800 font-bold text-xl mb-2">{error}</p>
            <p className="text-slate-600">Please try refreshing the page or contact support if the issue persists.</p>
          </div>
        )}

        {/* Results Table */}
        {!loading && !error && (
          <div className="animate-fade-in-up" style={{animationDelay: '0.2s'}}>
            <ResultsTable results={results} />
          </div>
        )}
      </div>
    </div>
  );
}