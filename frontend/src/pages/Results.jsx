import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import ResultsTable from "../components/ResultsTable";
import { fetchResults } from "../api/client";

export default function Results() {
  const [results, setResults] = useState([]);

  useEffect(() => {
    fetchResults().then((res) => setResults(res.data));
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <div className="p-8">
        <ResultsTable results={results} />
      </div>
    </div>
  );
}