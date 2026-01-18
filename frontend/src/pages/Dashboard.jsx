import Navbar from "../components/Navbar";
import UploadForm from "../components/UploadForm";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <div className="p-8">
        <UploadForm />
      </div>
    </div>
  );
}