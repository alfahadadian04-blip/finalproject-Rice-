import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Trash2 } from "lucide-react";

export type ScanHistoryItem = {
  id: string;
  label: string;
  confidence: number;
  timestamp: string;
};

type HistoryPanelProps = {
  history: ScanHistoryItem[];
  onDelete: (id: string) => void;
  onClearAll: () => void;
};

function confidenceStyle(confidence: number) {
  const percentage = confidence * 100;
  if (percentage >= 90) {
    return {
      badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
      text: "High",
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-600" />,
    };
  }

  if (percentage < 80) {
    return {
      badge: "bg-red-100 text-red-700 border-red-200",
      text: "Low",
      icon: <AlertTriangle className="h-4 w-4 text-red-600" />,
    };
  }

  return {
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    text: "Medium",
    icon: <AlertTriangle className="h-4 w-4 text-amber-600" />,
  };
}

export function HistoryPanel({ history, onDelete, onClearAll }: HistoryPanelProps) {
  return (
    <section className="rounded-xl border border-white/40 bg-white/70 p-4 text-slate-800 shadow-md backdrop-blur-sm">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Scan History</h2>
        <p className="text-xs text-slate-600">View All History</p>
      </div>

      <div className="mb-4 flex justify-end">
        <button
          type="button"
          onClick={onClearAll}
          disabled={history.length === 0}
          className="rounded-md border border-slate-300 bg-white/90 px-3 py-1.5 text-xs font-medium text-slate-800 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          Clear All History
        </button>
      </div>

      <div className="space-y-3">
        <AnimatePresence>
          {history.map((item) => {
            const style = confidenceStyle(item.confidence);
            return (
              <motion.article
                key={item.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: 24 }}
                className="flex items-center justify-between rounded-lg border border-slate-300 bg-white/90 p-3"
              >
                <div>
                  <p className="font-semibold text-slate-900">{item.label}</p>
                  <p className="text-xs text-slate-600">{item.timestamp}</p>
                  <div
                    className={`mt-2 inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs ${style.badge}`}
                  >
                    {style.icon}
                    {style.text} - {(item.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onDelete(item.id)}
                  className="inline-flex items-center gap-1 rounded-md border border-red-300 bg-red-500 px-2 py-1 text-xs text-white transition hover:bg-red-600"
                  aria-label="Delete history item"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete
                </button>
              </motion.article>
            );
          })}
        </AnimatePresence>

        {history.length === 0 && (
          <p className="rounded-xl border border-dashed border-slate-300 bg-white/60 p-4 text-sm text-slate-600">
            No scan records yet. Run a scan to populate history automatically.
          </p>
        )}
      </div>
    </section>
  );
}
