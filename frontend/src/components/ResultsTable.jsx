export default function ResultsTable({ results }) {
  if (!results || results.length === 0) {
    return (
      <div className="card p-16 text-center animate-fade-in">
        <svg className="w-20 h-20 text-slate-400 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
        </svg>
        <p className="text-slate-700 font-bold text-xl mb-2">No Results Yet</p>
        <p className="text-slate-500 text-base">Upload product images to see comprehensive analysis results</p>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden animate-fade-in">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gradient-to-r from-slate-800 via-slate-700 to-slate-900 text-white">
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">ID</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Filename</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Resolution</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Quality Score</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Background</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Image-Text Match</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Status</th>
              <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">Issues</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, index) => (
              <tr 
                key={r.id} 
                className={`border-b border-slate-100 hover:bg-primary-50/50 transition-all duration-200 ${
                  index % 2 === 0 ? 'bg-white' : 'bg-slate-50/50'
                }`}
              >
                <td className="px-6 py-5 text-sm text-slate-900 font-bold">#{r.id}</td>
                <td className="px-6 py-5 text-sm text-slate-700">
                  <div className="flex items-center">
                    <svg className="w-5 h-5 text-primary-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="truncate max-w-xs font-medium">{r.filename}</span>
                  </div>
                </td>
                <td className="px-6 py-5 text-sm">
                  <div className="text-slate-800 font-bold">{r.width}×{r.height}</div>
                  <div className="text-xs text-slate-600 font-medium">{r.aspect_ratio?.toFixed(2)} ratio</div>
                </td>
                <td className="px-6 py-5 text-sm">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-600 font-medium w-16">Blur:</span>
                      <span className={`text-xs font-bold ${r.blur_score >= 100 ? 'text-success-600' : 'text-danger-600'}`}>
                        {r.blur_score?.toFixed(0)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-600 font-medium w-16">Sharp:</span>
                      <span className={`text-xs font-bold ${r.sharpness_score >= 50 ? 'text-success-600' : 'text-danger-600'}`}>
                        {r.sharpness_score?.toFixed(0)}
                      </span>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-5 text-sm">
                  <div className="flex items-center">
                    <div className="w-full bg-slate-200 rounded-full h-2.5 mr-2">
                      <div 
                        className={`h-2.5 rounded-full transition-all ${
                          r.background_score >= 0.7 ? 'bg-success-500' : 
                          r.background_score >= 0.4 ? 'bg-warning-500' : 'bg-danger-500'
                        }`}
                        style={{ width: `${(r.background_score * 100).toFixed(0)}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-slate-700 font-bold">{(r.background_score * 100).toFixed(0)}%</span>
                  </div>
                  {r.has_watermark && (
                    <div className="text-xs text-warning-600 mt-2 font-bold">⚠ Watermark</div>
                  )}
                </td>
                <td className="px-6 py-5 text-sm">
                  {r.similarity_score !== null && r.similarity_score !== undefined ? (
                    <div>
                      {r.has_mismatch ? (
                        <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-bold bg-warning-100 text-warning-800 border border-warning-300">
                          <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                          </svg>
                          Mismatch
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-bold bg-success-100 text-success-800 border border-success-300">
                          <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          Match
                        </span>
                      )}
                      <div className="mt-2 text-xs text-slate-600 font-medium">
                        Score: {(r.similarity_score * 100).toFixed(1)}%
                      </div>
                    </div>
                  ) : (
                    <span className="text-xs text-slate-500 italic">No description</span>
                  )}
                </td>
                <td className="px-6 py-5">
                  {r.passed ? (
                    <span className="badge-success text-sm py-1.5 px-4">
                      <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      PASS
                    </span>
                  ) : (
                    <span className="badge-danger text-sm py-1.5 px-4">
                      <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                      FAIL
                    </span>
                  )}
                </td>
                <td className="px-6 py-5 text-sm">
                  {r.passed ? (
                    <span className="text-success-600 font-bold">✓ All checks passed</span>
                  ) : (
                    <div className="text-danger-700">
                      <div className="font-bold mb-2">{r.reason}</div>
                      {r.improvement_suggestions && (
                        <details className="text-xs text-slate-700 mt-3">
                          <summary className="cursor-pointer text-primary-600 hover:text-primary-800 font-bold">View suggestions</summary>
                          <div className="mt-3 p-3 bg-primary-50 rounded-lg text-left border border-primary-200">
                            {r.improvement_suggestions.split(';').map((s, i) => (
                              <div key={i} className="mb-2 last:mb-0 flex items-start">
                                <span className="text-primary-600 mr-2">•</span>
                                <span className="font-medium">{s.trim()}</span>
                              </div>
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
      <div className="bg-slate-50 px-6 py-5 border-t border-slate-200">
        <p className="text-sm text-slate-700 font-medium">
          Showing <span className="font-bold text-primary-600">{results.length}</span> result{results.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  );
}