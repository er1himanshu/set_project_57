import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <nav className="bg-white shadow-medium border-b border-gray-100 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex justify-between items-center">
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="w-12 h-12 bg-gradient-to-br from-primary-600 to-secondary-600 rounded-xl flex items-center justify-center transform group-hover:scale-105 transition-transform duration-200 shadow-lg">
              <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors duration-200">
                Product Quality AI
              </h1>
              <p className="text-xs text-gray-500 font-medium">Ecommerce Image Analyzer</p>
            </div>
          </Link>
          <div className="flex space-x-2">
            <Link 
              to="/" 
              className={`px-6 py-2.5 rounded-xl font-semibold transition-all duration-200 ${
                isActive('/') 
                  ? 'bg-gradient-to-r from-primary-600 to-secondary-600 text-white shadow-lg transform scale-105' 
                  : 'text-gray-700 hover:bg-gray-100 hover:text-primary-600'
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                Upload
              </span>
            </Link>
            <Link 
              to="/results" 
              className={`px-6 py-2.5 rounded-xl font-semibold transition-all duration-200 ${
                isActive('/results') 
                  ? 'bg-gradient-to-r from-primary-600 to-secondary-600 text-white shadow-lg transform scale-105' 
                  : 'text-gray-700 hover:bg-gray-100 hover:text-primary-600'
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Results
              </span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}