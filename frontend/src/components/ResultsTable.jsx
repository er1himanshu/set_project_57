export default function ResultsTable({ results }) {
  return (
    <table className="w-full border">
      <thead>
        <tr className="bg-gray-200">
          <th>ID</th>
          <th>Filename</th>
          <th>Resolution</th>
          <th>Blur</th>
          <th>Brightness</th>
          <th>Status</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r) => (
          <tr key={r.id} className="text-center border-t">
            <td>{r.id}</td>
            <td>{r.filename}</td>
            <td>{r.width}x{r.height}</td>
            <td>{r.blur_score.toFixed(2)}</td>
            <td>{r.brightness_score.toFixed(2)}</td>
            <td className={r.passed ? "text-green-600" : "text-red-600"}>
              {r.passed ? "PASS" : "FAIL"}
            </td>
            <td>{r.reason}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}