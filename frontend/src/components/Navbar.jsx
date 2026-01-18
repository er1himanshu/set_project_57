import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <div className="bg-gray-900 text-white px-6 py-3 flex justify-between">
      <h1 className="text-xl font-bold">AI Image Quality</h1>
      <div className="space-x-4">
        <Link to="/">Upload</Link>
        <Link to="/results">Results</Link>
      </div>
    </div>
  );
}