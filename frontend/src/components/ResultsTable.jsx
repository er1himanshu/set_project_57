export default function ResultsTable({ results }) {
  if (!results || results.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
        <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
        </svg>
        <p className="text-gray-600 font-medium text-lg mb-2">No results yet</p>
        <p className="text-gray-500">Upload some images to see analysis results here</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
              <th className="px-6 py-4 text-left text-sm font-semibold">ID</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">Filename</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">Resolution</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">Quality Score</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">Background</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">Status</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">Issues</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, index) => (
              <tr 
                key={r.id} 
                className={`border-b border-gray-100 hover:bg-indigo-50 transition-colors ${
                  index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                }`}
              >
                <td className="px-6 py-4 text-sm text-gray-700 font-medium">#{r.id}</td>
                <td className="px-6 py-4 text-sm text-gray-700">
                  <div className="flex items-center">
                    <svg className="w-4 h-4 text-gray-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="truncate max-w-xs">{r.filename}</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm">
                  <div className="text-gray-700 font-medium">{r.width}×{r.height}</div>
                  <div className="text-xs text-gray-500">{r.aspect_ratio?.toFixed(2)} ratio</div>
                </td>
                <td className="px-6 py-4 text-sm">
                  <div className="space-y-1">
                    <div className="flex items-center">
                      <span className="text-xs text-gray-500 w-16">Blur:</span>
                      <span className={`text-xs font-semibold ${r.blur_score >= 100 ? 'text-green-600' : 'text-red-600'}`}>
                        {r.blur_score?.toFixed(0)}
                      </span>
                    </div>
                    <div className="flex items-center">
                      <span className="text-xs text-gray-500 w-16">Sharp:</span>
                      <span className={`text-xs font-semibold ${r.sharpness_score >= 50 ? 'text-green-600' : 'text-red-600'}`}>
                        {r.sharpness_score?.toFixed(0)}
                      </span>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm">
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                      <div 
                        className={`h-2 rounded-full ${
                          r.background_score >= 0.7 ? 'bg-green-500' : 
                          r.background_score >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${(r.background_score * 100).toFixed(0)}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-gray-600">{(r.background_score * 100).toFixed(0)}%</span>
                  </div>
                  {r.has_watermark && (
                    <div className="text-xs text-orange-600 mt-1">⚠ Watermark</div>
                  )}
                </td>
                <td className="px-6 py-4">
                  {r.passed ? (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
                      <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      PASS
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-800">
                      <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                      FAIL
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 text-sm">
                  {r.passed ? (
                    <span className="text-green-600 font-medium">✓ All checks passed</span>
                  ) : (
                    <div className="text-red-700">
                      <div className="font-medium mb-1">{r.reason}</div>
                      {r.improvement_suggestions && (
                        <details className="text-xs text-gray-600 mt-2">
                          <summary className="cursor-pointer text-indigo-600 hover:text-indigo-800">View suggestions</summary>
                          <div className="mt-2 p-2 bg-blue-50 rounded text-left">
                            {r.improvement_suggestions.split(';').map((s, i) => (
                              <div key={i} className="mb-1">• {s.trim()}</div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Footer */}
      <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
        <p className="text-sm text-gray-600">
          Showing <span className="font-semibold">{results.length}</span> result{results.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  );
}