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

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <div className="p-8">
        <h2 className="text-2xl font-bold mb-4">Image Analysis Results</h2>
        {loading && <p className="text-gray-600">Loading results...</p>}
        {error && <p className="text-red-600">{error}</p>}
        {!loading && !error && <ResultsTable results={results} />}
      </div>
    </div>
  );
}