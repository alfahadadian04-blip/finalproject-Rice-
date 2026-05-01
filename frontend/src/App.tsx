import { useEffect, useState } from "react";
import { Activity, Camera, Clock4, Leaf, ScanLine } from "lucide-react";
import { Dashboard } from "./components/Dashboard";
import { HistoryPanel, type ScanHistoryItem } from "./components/HistoryPanel";
import backgroundImage from "./assets/image_b4047c.png";

type PredictionResponse = {
  label: string;
  confidence: number;
  all_scores: Record<string, number>;
};

const API_URL = import.meta.env.VITE_API_URL ?? "/predict";
const REQUEST_TIMEOUT_MS = 15000;

const NAV_ITEMS = ["Home", "Scan", "Encyclopedia", "History"] as const;

function App() {
  const [activeTab, setActiveTab] = useState<(typeof NAV_ITEMS)[number]>("Home");
  const [clock, setClock] = useState<string>(new Date().toLocaleString());
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<ScanHistoryItem[]>([]);

  useEffect(() => {
    const timer = setInterval(() => {
      setClock(new Date().toLocaleString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onFileChange = (selected: File | null) => {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setError("");
  };

  const onScan = async () => {
    if (!file) {
      setError("Upload an image before running prediction.");
      return;
    }

    setLoading(true);
    setError("");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const formData = new FormData();
      formData.append("image", file);
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Prediction failed");
      }

      const data = (await response.json()) as PredictionResponse;
      setResult(data);
      setHistory((previous) => [
        {
          id: crypto.randomUUID(),
          label: data.label,
          confidence: data.confidence,
          timestamp: new Date().toLocaleString(),
        },
        ...previous,
      ]);
      setActiveTab("History");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request timed out. Please try again.");
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unexpected error while contacting API.");
      }
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  return (
    <main
      className="min-h-screen w-screen bg-cover bg-center text-white"
      style={{ backgroundImage: `url(${backgroundImage})` }}
    >
      <div className="min-h-screen w-full bg-black/30 px-3 py-4 md:px-5">
        <div className="mx-auto w-full max-w-6xl rounded-2xl border border-white/30 bg-white/15 p-3 backdrop-blur-md">
          <header className="mb-2 flex items-center justify-between rounded-xl border border-white/25 bg-black/35 px-3 py-2">
            <div className="flex items-center gap-2 text-xs md:text-sm">
              <Leaf className="h-4 w-4 text-emerald-300" />
              <h1 className="font-medium">WMSU Rice Disease Detection</h1>
            </div>
            <div className="inline-flex items-center gap-2 rounded-full bg-white/85 px-3 py-1 text-[10px] text-slate-700 md:text-xs">
              <Clock4 className="h-3.5 w-3.5" />
              {clock}
            </div>
          </header>

          <nav className="mb-3 flex flex-wrap gap-2 rounded-xl border border-white/25 bg-black/35 p-2">
            {NAV_ITEMS.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setActiveTab(item)}
                className={`rounded-md px-3 py-1.5 text-xs transition ${
                  activeTab === item
                    ? "bg-emerald-700 text-white"
                    : "bg-white/75 text-slate-700 hover:bg-white"
                }`}
              >
                {item}
              </button>
            ))}
          </nav>

          <section className="space-y-3">
            {(activeTab === "Home" || activeTab === "Scan") && (
              <section className="grid gap-3 rounded-xl border border-white/40 bg-white/70 p-4 text-slate-800 lg:grid-cols-12">
                <div className="lg:col-span-8">
                  <h2 className="text-3xl font-semibold">Protect Your Rice Crops with AI</h2>
                  <p className="mt-2 text-xs leading-relaxed text-slate-700 md:text-sm">
                    WMSU Rice Disease Detection: Utilizing YOLO11-cls Nano for instant diagnostics
                    to support Western Mindanao farmers.
                  </p>
                  <button
                    type="button"
                    onClick={() => setActiveTab("Scan")}
                    className="mt-4 rounded-md bg-emerald-700 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-800"
                  >
                    Start Scanning Now
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2 lg:col-span-4">
                  <div className="rounded-lg border border-slate-200 bg-white/80 p-3 text-center">
                    <p className="text-2xl font-bold">{history.length}</p>
                    <p className="text-[11px] text-slate-500">Total Scans</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-white/80 p-3 text-center">
                    <p className="text-2xl font-bold">{result ? `${(result.confidence * 100).toFixed(0)}%` : "--"}</p>
                    <p className="text-[11px] text-slate-500">Accuracy</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-white/80 p-3 text-center">
                    <p className="text-2xl font-bold">{history.length > 0 ? history.length : "--"}</p>
                    <p className="text-[11px] text-slate-500">Diagnoses</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-white/80 p-3 text-center">
                    <p className="text-2xl font-bold">{result ? "1" : "--"}</p>
                    <p className="text-[11px] text-slate-500">Avg Reports</p>
                  </div>
                </div>
              </section>
            )}

            {(activeTab === "Home" || activeTab === "Scan") && (
              <section className="grid gap-3 lg:grid-cols-2">
                <Dashboard
                  previewUrl={previewUrl}
                  result={result}
                  error={error}
                  loading={loading}
                  onFileChange={onFileChange}
                  onScan={onScan}
                />
                <section className="rounded-xl border border-white/40 bg-white/70 p-4 text-slate-800 shadow-md backdrop-blur-sm">
                  <div className="flex items-center justify-between">
                    <h3 className="text-2xl font-semibold">Disease Encyclopedia</h3>
                    <div className="rounded-lg bg-emerald-100 p-2 text-emerald-700">
                      <Camera className="h-4 w-4" />
                    </div>
                  </div>
                  <p className="mt-2 text-sm text-slate-700">Browse our comprehensive disease library.</p>
                  <button
                    type="button"
                    onClick={() => setActiveTab("Encyclopedia")}
                    className="mt-3 text-sm font-medium text-slate-800 underline"
                  >
                    Learn more
                  </button>
                </section>
              </section>
            )}

            {activeTab === "Encyclopedia" && (
              <section className="rounded-xl border border-white/40 bg-white/70 p-4 text-slate-800 shadow-md backdrop-blur-sm">
                <h2 className="text-2xl font-semibold">Disease Encyclopedia</h2>
                <p className="mt-2 text-sm text-slate-700">
                  Encyclopedia content module is ready for disease profiles, treatment guidance,
                  and prevention best practices.
                </p>
              </section>
            )}

            {(activeTab === "Home" || activeTab === "History") && (
              <HistoryPanel
                history={history}
                onDelete={(id) => setHistory((previous) => previous.filter((item) => item.id !== id))}
                onClearAll={() => setHistory([])}
              />
            )}
          </section>

          <footer className="mt-3 flex items-center gap-2 text-[10px] text-white/95 md:text-xs">
          <ScanLine className="h-4 w-4" />
          Built with FastAPI, YOLO11-cls Nano, React, Tailwind, Lucide, and Framer Motion.
          </footer>
        </div>
      </div>
    </main>
  );
}

export default App;
