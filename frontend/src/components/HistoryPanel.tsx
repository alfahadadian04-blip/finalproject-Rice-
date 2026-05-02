import { AnimatePresence, motion } from "framer-motion";
import { Trash2 } from "lucide-react";

export type ScanHistoryItem = {
  id: string;
  imageUrl: string;
  fileName: string;
  label: string;
  confidence: number;
  timestamp: string;
};

type HistoryPanelProps = {
  history: ScanHistoryItem[];
  onDelete: (id: string) => void;
  onClearAll: () => void;
};

export function HistoryPanel({ history, onDelete, onClearAll }: HistoryPanelProps) {
  return (
    <section className="flex h-full min-h-0 flex-col rounded-xl border border-slate-200 bg-slate-50 p-3 text-slate-800">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Scan History</h2>
        <button
          type="button"
          onClick={onClearAll}
          disabled={history.length === 0}
          className="rounded-md border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Clear
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-auto">
        <AnimatePresence>
          {history.map((item) => {
            return (
              <motion.article
                key={item.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: 24 }}
                className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-3 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="flex min-w-0 items-center gap-3">
                  <img
                    src={item.imageUrl}
                    alt={`Scanned rice leaf for ${item.label}`}
                    className="h-16 w-16 shrink-0 rounded-lg border border-slate-300 object-cover"
                  />
                  <div className="min-w-0">
                    <p className="truncate font-semibold text-slate-900">{item.label}</p>
                    <p className="truncate text-xs text-slate-500">{item.fileName}</p>
                    <p className="text-xs text-slate-600">{item.timestamp}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold text-emerald-600">{(item.confidence * 100).toFixed(2)}%</p>
                  <button
                    type="button"
                    onClick={() => onDelete(item.id)}
                    className="inline-flex items-center gap-1 rounded-md border border-red-300 bg-red-500 px-2 py-1 text-xs text-white transition hover:bg-red-600"
                    aria-label="Delete history item"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete
                  </button>
                </div>
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
