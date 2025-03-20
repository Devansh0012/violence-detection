export default function AlertPanel({ alerts }: { alerts: string[] }) {
  return (
    <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Security Alerts</h2>
        {alerts.length > 0 && (
          <span className="bg-red-600 text-white text-xs font-bold px-2 py-1 rounded-full">
            {alerts.length}
          </span>
        )}
      </div>
      
      <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
        {alerts.map((alert, index) => {
          // Extract timestamp if present in the format "Violence detected at 10:30:45 AM with 87.5% confidence"
          const timestampMatch = alert.match(/Violence detected at ([\d:]+\s?[APap][Mm]?) with/);
          const timestamp = timestampMatch ? timestampMatch[1] : "";
          
          // Extract confidence if present
          const confidenceMatch = alert.match(/with (\d+\.\d+)% confidence/);
          const confidence = confidenceMatch ? confidenceMatch[1] : "";
          
          return (
            <div key={index} className="bg-red-500/20 p-3 rounded-lg border border-red-500/50">
              <div className="flex items-start space-x-3">
                <div className="mt-1 text-red-400 text-xl">⚠️</div>
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <p className="text-red-400 font-medium">Violence Detected</p>
                    {timestamp && (
                      <span className="text-xs text-gray-400">{timestamp}</span>
                    )}
                  </div>
                  {confidence && (
                    <p className="text-sm text-red-300 mt-1">Confidence: {confidence}%</p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        
        {alerts.length === 0 && (
          <div className="text-center py-8 text-gray-400">
            <div className="text-3xl mb-2">✓</div>
            <p>No alerts detected</p>
            <p className="text-xs mt-1 text-gray-500">The system will notify you when violence is detected</p>
          </div>
        )}
      </div>
    </div>
  )
}