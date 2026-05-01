import { ImagePlus, Leaf, LoaderCircle } from "lucide-react";

type PredictionResponse = {
  label: string;
  confidence: number;
  all_scores: Record<string, number>;
};

type DashboardProps = {
  previewUrl: string;
  result: PredictionResponse | null;
  error: string;
  loading: boolean;
  onFileChange: (file: File | null) => void;
  onScan: () => void;
};

export function Dashboard({
  previewUrl,
  result,
  error,
  loading,
  onFileChange,
  onScan,
}: DashboardProps) {
  const sortedScores = result
    ? Object.entries(result.all_scores).sort((left, right) => right[1] - left[1])
    : [];

  return (
    <section className="space-y-3 rounded-xl border border-white/40 bg-white/70 p-4 text-slate-800 shadow-md backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Start Scanning</h2>
        <div className="rounded-lg bg-emerald-100 p-2 text-emerald-700">
          <ImagePlus className="h-4 w-4" />
        </div>
      </div>
      <p className="text-sm text-slate-700">
        Upload or capture a photo of your rice leaf.
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-slate-300 bg-white/70 px-3 py-2 text-sm text-slate-800 transition hover:bg-white">
          Choose Image
          <input
            className="hidden"
            type="file"
            accept="image/*"
            onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
          />
        </label>

        <button
          type="button"
          onClick={onScan}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-md border border-emerald-700 bg-emerald-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? (
            <>
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Scanning...
            </>
          ) : (
            <>
              <Leaf className="h-4 w-4" />
              Run Scan
            </>
          )}
        </button>
      </div>

      {previewUrl && (
        <img
          src={previewUrl}
          alt="Uploaded rice preview"
          className="max-h-56 w-full rounded-xl border border-slate-300 object-cover"
        />
      )}

      {error && <p className="rounded-xl bg-red-500/20 p-3 text-sm text-red-100">{error}</p>}

      {result && (
        <div className="space-y-2 rounded-xl border border-slate-300 bg-white/75 p-4">
          <h3 className="font-semibold text-slate-900">Latest Result</h3>
          <p className="text-slate-800">
            <span className="font-semibold">Prediction:</span> {result.label}
          </p>
          <p className="text-slate-800">
            <span className="font-semibold">Confidence:</span>{" "}
            {(result.confidence * 100).toFixed(2)}%
          </p>
          <ul className="mt-2 space-y-1 text-sm text-slate-700">
            {sortedScores.map(([label, score]) => (
              <li key={label}>
                {label}: {(score * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
