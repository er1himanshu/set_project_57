import Navbar from "../components/Navbar";
import UploadForm from "../components/UploadForm";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <div className="p-8">
        <h1 className="text-3xl font-bold mb-6">AI Image Quality Analysis Dashboard</h1>
        <UploadForm />
      </div>
    </div>
  );
}