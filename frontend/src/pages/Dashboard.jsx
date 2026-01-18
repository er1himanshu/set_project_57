import Navbar from "../components/Navbar";
import UploadForm from "../components/UploadForm";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
      <Navbar />
      
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-black opacity-5"></div>
        <div className="absolute inset-0" style={{
          backgroundImage: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.05"%3E%3Cpath d="M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
        }}></div>
        <div className="max-w-7xl mx-auto px-6 py-20 text-center relative z-10">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 animate-fade-in">
            AI-Powered Product Image Evaluator
          </h1>
          <p className="text-xl text-primary-100 mb-10 max-w-3xl mx-auto leading-relaxed animate-slide-in">
            Ensure your product images meet professional ecommerce quality standards with our comprehensive AI analysis.
            Check resolution, lighting, background, sharpness, and more in seconds.
          </p>
          <div className="flex flex-wrap justify-center gap-6">
            <div className="bg-white/10 backdrop-blur-sm px-8 py-4 rounded-xl shadow-lg transform hover:scale-105 transition-transform duration-200">
              <div className="text-3xl font-bold mb-1">10+</div>
              <div className="text-sm text-primary-100 font-medium">Quality Checks</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm px-8 py-4 rounded-xl shadow-lg transform hover:scale-105 transition-transform duration-200">
              <div className="text-3xl font-bold mb-1">Instant</div>
              <div className="text-sm text-primary-100 font-medium">Analysis</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm px-8 py-4 rounded-xl shadow-lg transform hover:scale-105 transition-transform duration-200">
              <div className="text-3xl font-bold mb-1">AI-Powered</div>
              <div className="text-sm text-primary-100 font-medium">Suggestions</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        {/* Features Cards */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="card p-8 hover:scale-105 transition-transform duration-300">
            <div className="w-14 h-14 bg-primary-100 rounded-xl flex items-center justify-center mb-5 shadow-soft">
              <svg className="w-7 h-7 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-3 text-gray-800">Quality Verification</h3>
            <p className="text-gray-600 leading-relaxed">
              Automatic checks for resolution, blur, brightness, contrast, and sharpness to ensure professional quality standards.
            </p>
          </div>
          
          <div className="card p-8 hover:scale-105 transition-transform duration-300">
            <div className="w-14 h-14 bg-secondary-100 rounded-xl flex items-center justify-center mb-5 shadow-soft">
              <svg className="w-7 h-7 text-secondary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-3 text-gray-800">Ecommerce Standards</h3>
            <p className="text-gray-600 leading-relaxed">
              Validates aspect ratio, background cleanliness, and detects watermarks for marketplace compliance and best practices.
            </p>
          </div>
          
          <div className="card p-8 hover:scale-105 transition-transform duration-300">
            <div className="w-14 h-14 bg-warning-100 rounded-xl flex items-center justify-center mb-5 shadow-soft">
              <svg className="w-7 h-7 text-warning-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-3 text-gray-800">Smart Suggestions</h3>
            <p className="text-gray-600 leading-relaxed">
              Get actionable improvement tips to enhance your product images and boost conversion rates with detailed insights.
            </p>
          </div>
        </div>

        {/* Upload Section */}
        <UploadForm />
      </div>
    </div>
  );
}