export default function ResultsTable({ results }) {
  if (!results || results.length === 0) {
    return (
      <div className="p-6 bg-white rounded shadow text-center text-gray-500">
        No results yet. Upload some images to see analysis results.
      </div>
    );
  }

  return (
    <div className="bg-white rounded shadow overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-gray-200">
            <th className="p-3 border">ID</th>
            <th className="p-3 border">Filename</th>
            <th className="p-3 border">Resolution</th>
            <th className="p-3 border">Blur</th>
            <th className="p-3 border">Brightness</th>
            <th className="p-3 border">Status</th>
            <th className="p-3 border">Reason</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.id} className="text-center border-t hover:bg-gray-50">
              <td className="p-3 border">{r.id}</td>
              <td className="p-3 border">{r.filename}</td>
              <td className="p-3 border">{r.width}x{r.height}</td>
              <td className="p-3 border">{r.blur_score.toFixed(2)}</td>
              <td className="p-3 border">{r.brightness_score.toFixed(2)}</td>
              <td className={`p-3 border font-bold ${r.passed ? "text-green-600" : "text-red-600"}`}>
                {r.passed ? "PASS" : "FAIL"}
              </td>
              <td className="p-3 border">{r.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}