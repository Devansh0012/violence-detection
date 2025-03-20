export default function AlertPanel({ alerts }: { alerts: string[] }) {
    return (
      <div className="bg-gray-800 p-6 rounded-xl h-[400px] overflow-y-auto">
        <h2 className="text-xl font-semibold mb-4">Security Alerts</h2>
        
        <div className="space-y-3">
        {alerts.map((alert, index) => (
                    <div key={index} className="bg-red-500/20 p-3 rounded-lg border border-red-500/50">
                        <div className="flex items-center gap-2 text-red-400">
                            <span className="text-sm">⚠️</span>
                            <div className="flex-1">
                                <p className="text-sm">{alert}</p>
                                <img 
                                    src={`/detections/${alert.split(' ')[4]}.jpg`} 
                                    className="mt-2 rounded w-20 h-20 object-cover"
                                    alt="Detection snapshot"
                                />
                            </div>
                        </div>
                    </div>
                ))}
          
          {alerts.length === 0 && (
            <div className="text-center py-4 text-gray-400">
              No alerts detected
            </div>
          )}
        </div>
      </div>
    )
  }