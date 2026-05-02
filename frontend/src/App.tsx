import { useEffect, useState } from "react";
import { Clock4, ScanLine } from "lucide-react";
import { HistoryPanel, type ScanHistoryItem } from "./components/HistoryPanel";
import backgroundImage from "./assets/image_b4047c.png";
import headerLogo from "./assets/header-logo.png";

type PredictionResponse = {
  label: string;
  confidence: number;
  all_scores: Record<string, number>;
  is_reliable?: boolean;
  message?: string;
  has_camera_metadata?: boolean;
};

const API_URL = import.meta.env.VITE_API_URL ?? "/predict";
const REQUEST_TIMEOUT_MS = 15000;

const NAV_ITEMS = ["Home", "Scan", "Encyclopedia", "History"] as const;
const CLASSIFICATIONS = [
  {
    name: "Leaf Blight",
    symptoms: "Yellowing leaf edges that become dry and brown.",
    action: "Improve drainage, avoid leaf injury, and remove heavily infected leaves.",
  },
  {
    name: "Rice Blast",
    symptoms: "Diamond-shaped lesions with gray or white centers.",
    action: "Use resistant varieties and apply balanced nitrogen fertilizer.",
  },
  {
    name: "Rice Leaffolder",
    symptoms: "Folded leaves with scraped white streaks.",
    action: "Field monitoring and targeted control when infestation is high.",
  },
  {
    name: "Rice Stripes",
    symptoms: "Yellow-white stripes and stunted growth.",
    action: "Control vectors early and maintain healthy seedling conditions.",
  },
  {
    name: "Rice Tungro",
    symptoms: "Stunted plants and orange-yellow leaf discoloration.",
    action: "Rogue infected plants and manage green leafhopper vectors.",
  },
  {
    name: "Healthy Leaf",
    symptoms: "Uniform green color and no visible lesions or streaking.",
    action: "Maintain best practices and regular monitoring.",
  },
] as const;

function App() {
  const [activeTab, setActiveTab] = useState<(typeof NAV_ITEMS)[number]>("Home");
  const [clock, setClock] = useState<string>(new Date().toLocaleString());
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const [history, setHistory] = useState<ScanHistoryItem[]>([]);
  const averageAccuracy =
    history.length > 0
      ? history.reduce((total, item) => total + item.confidence, 0) / history.length
      : null;

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
    setFileName(selected.name);
    setPreviewUrl(URL.createObjectURL(selected));
    setError("");
    setWarning("");
  };

  const onPasteImage = async () => {
    if (!navigator.clipboard || !navigator.clipboard.read) {
      setError("Clipboard image paste is not supported in this browser.");
      return;
    }

    try {
      const clipboardItems = await navigator.clipboard.read();
      for (const item of clipboardItems) {
        const imageType = item.types.find((type) => type.startsWith("image/"));
        if (!imageType) continue;
        const blob = await item.getType(imageType);
        const pastedFile = new File([blob], `pasted-image.${imageType.split("/")[1] || "png"}`, {
          type: imageType,
        });
        onFileChange(pastedFile);
        return;
      }
      setError("No image found in clipboard. Copy an image first, then click Paste Image.");
    } catch {
      setError("Clipboard access was blocked. Allow clipboard permission and try again.");
    }
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
      const historyImageUrl = URL.createObjectURL(file);
      setResult(data);
      setWarning(data.message ?? "");
      setHistory((previous) => [
        {
          id: crypto.randomUUID(),
          imageUrl: historyImageUrl,
          fileName: fileName || "uploaded-image.jpg",
          label: data.label,
          confidence: data.confidence,
          timestamp: new Date().toLocaleString(),
        },
        ...previous,
      ]);
    } catch (err) {
      setWarning("");
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

  const onDeleteHistoryItem = (id: string) => {
    setHistory((previous) => {
      const item = previous.find((entry) => entry.id === id);
      if (item) {
        URL.revokeObjectURL(item.imageUrl);
      }
      return previous.filter((entry) => entry.id !== id);
    });
  };

  const onClearAllHistory = () => {
    setHistory((previous) => {
      previous.forEach((item) => URL.revokeObjectURL(item.imageUrl));
      return [];
    });
  };

  const latestResult = history[0] ?? null;
  const sortedScores = result
    ? Object.entries(result.all_scores).sort((left, right) => right[1] - left[1])
    : [];

  return (
    <main
      className="h-screen w-screen overflow-hidden bg-cover bg-center p-2 text-white md:p-3"
      style={{ backgroundImage: `url(${backgroundImage})` }}
    >
      <div className="mx-auto flex h-full w-full max-w-6xl flex-col rounded-3xl border border-white/30 bg-slate-900/35 p-2 backdrop-blur-sm md:p-3">
        <header className="mb-2 flex items-center justify-between rounded-2xl border border-white/20 bg-slate-800/45 px-3 py-2">
          <div className="flex items-center gap-2">
            <img src={headerLogo} alt="WMSU Rice Disease Detection logo" className="h-9 w-9 rounded-md object-cover" />
            <div>
              <h1 className="text-sm font-semibold md:text-base">WMSU Rice Disease Detection</h1>
              <p className="text-[10px] text-white/80 md:text-xs">Realtime AI Rice Leaf Diagnosis</p>
            </div>
          </div>
          <div className="inline-flex items-center gap-2 rounded-full bg-white/85 px-3 py-1 text-[10px] text-slate-700 md:text-xs">
            <Clock4 className="h-3.5 w-3.5" />
            {clock}
          </div>
        </header>

        <nav className="mb-2 grid grid-cols-2 gap-2 rounded-2xl border border-white/20 bg-slate-800/45 p-2 md:grid-cols-4">
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setActiveTab(item)}
              className={`rounded-lg px-3 py-2 text-xs font-medium transition md:text-sm ${
                activeTab === item
                  ? "bg-emerald-600 text-white shadow"
                  : "bg-white/85 text-slate-700 hover:bg-white"
              }`}
            >
              {item}
            </button>
          ))}
        </nav>

        <section className="min-h-0 flex-1 overflow-hidden rounded-2xl border border-white/20 bg-white/70 p-2 text-slate-800 md:p-3">
          {activeTab === "Home" && (
            <section className="grid h-full gap-3 overflow-auto lg:grid-cols-3">
              <article className="rounded-xl border border-slate-200 bg-slate-50 p-3 lg:col-span-2">
                <h2 className="text-2xl font-semibold">About AI Diagnostic System</h2>
                <p className="mt-2 text-sm text-slate-700">
                  An advanced AI-powered diagnostic tool specifically designed to help Western Mindanao farmers identify rice leaf diseases in real-time using deep learning models.
                </p>
                <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3">
                  <h3 className="text-base font-semibold">How this system works</h3>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
                    <li>Upload a rice leaf image in the Scan tab.</li>
                    <li>YOLO-based model classifies likely disease category.</li>
                    <li>Confidence scores help compare possible diagnoses.</li>
                    <li>Each successful prediction is saved in History.</li>
                  </ul>
                </div>
              </article>

              <article className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <h2 className="text-2xl font-semibold">System Metrics</h2>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-center">
                    <p className="text-2xl font-bold">{history.length}</p>
                    <p className="text-xs text-slate-500">Total Scans</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-center">
                    <p className="text-2xl font-bold">{averageAccuracy !== null ? `${(averageAccuracy * 100).toFixed(0)}%` : "--"}</p>
                    <p className="text-xs text-slate-500">Accuracy</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-center">
                    <p className="text-2xl font-bold">{history.length}</p>
                    <p className="text-xs text-slate-500">Diagnoses</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-center">
                    <p className="text-2xl font-bold">{latestResult ? `${(latestResult.confidence * 100).toFixed(0)}%` : "--"}</p>
                    <p className="text-xs text-slate-500">Avg Reports</p>
                  </div>
                </div>
              </article>
            </section>
          )}

          {activeTab === "Scan" && (
            <section className="grid h-full gap-3 overflow-hidden lg:grid-cols-3">
              <article className="rounded-xl border border-slate-200 bg-slate-50 p-3 lg:col-span-2">
                <h2 className="text-2xl font-semibold">Start Scanning</h2>
                <p className="mt-2 text-sm text-slate-700">Upload or capture a photo of your rice leaf.</p>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 hover:bg-slate-100">
                    Choose Image
                    <input
                      className="hidden"
                      type="file"
                      accept="image/*"
                      capture="environment"
                      onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
                    />
                  </label>
                  <button
                    type="button"
                    onClick={onPasteImage}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-800 hover:bg-slate-100"
                  >
                    Paste Image
                  </button>
                  <button
                    type="button"
                    onClick={onScan}
                    disabled={loading}
                    className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {loading ? "Scanning..." : "Run Scan"}
                  </button>
                </div>
                <div className="mt-3 h-[32vh] min-h-[200px] overflow-hidden rounded-xl border border-slate-200 bg-white">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Uploaded rice preview"
                      className="h-full w-full object-cover object-center"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-slate-400">
                      Image preview will appear here
                    </div>
                  )}
                </div>
                {error && (
                  <p className="mt-3 rounded-lg border border-red-200 bg-red-50 p-2 text-sm text-red-600">
                    {error}
                  </p>
                )}
              </article>

              <article className="space-y-3 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-3">
                <section className="rounded-lg border border-slate-200 bg-white p-3">
                  <h3 className="text-lg font-semibold">Latest Result</h3>
                  <div className="mt-2 border-b border-slate-200 pb-2 text-sm">
                    <p className="text-slate-600">Prediction</p>
                    <p className="text-2xl font-semibold">{result?.label ?? "--"}</p>
                    <p className="text-slate-600">Confidence: {result ? `${(result.confidence * 100).toFixed(2)}%` : "--"}</p>
                    {result?.is_reliable === false && result?.message && (
                      <p className="mt-1 text-xs text-amber-700">{result.message ?? "Low reliability result."}</p>
                    )}
                  </div>
                  <ul className="mt-2 max-h-[150px] space-y-1 overflow-auto text-sm">
                    {sortedScores.map(([label, score]) => (
                      <li key={label} className="flex items-center justify-between rounded border border-slate-200 px-2 py-1">
                        <span>{label}</span>
                        <span className="font-semibold">{(score * 100).toFixed(2)}%</span>
                      </li>
                    ))}
                  </ul>
                </section>

                <section className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-700">
                  <h4 className="text-base font-semibold text-slate-900">Realtime Test Flow</h4>
                  <ul className="mt-2 list-disc space-y-1 pl-5">
                    <li>Home: Overview and session scan summary</li>
                    <li>Scan: Upload image and run inference</li>
                    <li>Encyclopedia: Disease guidance library available</li>
                    <li>History: {history.length} scan records stored in session</li>
                  </ul>
                </section>
              </article>
            </section>
          )}

          {activeTab === "Encyclopedia" && (
            <section className="h-full overflow-auto">
              <h2 className="text-2xl font-semibold">Disease Encyclopedia</h2>
              <p className="mt-1 text-sm text-slate-700">
                Quick references to support real-time diagnosis interpretation.
              </p>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                {CLASSIFICATIONS.map((item) => (
                  <article key={item.name} className="rounded-lg border border-slate-200 bg-white p-3">
                    <h3 className="text-lg font-semibold">{item.name}</h3>
                    <p className="mt-1 text-sm text-slate-700">
                      <span className="font-semibold text-slate-900">Symptoms:</span> {item.symptoms}
                    </p>
                    <p className="mt-1 text-sm text-slate-700">
                      <span className="font-semibold text-slate-900">Action:</span> {item.action}
                    </p>
                  </article>
                ))}
              </div>
            </section>
          )}

          {activeTab === "History" && (
            <HistoryPanel history={history} onDelete={onDeleteHistoryItem} onClearAll={onClearAllHistory} />
          )}
        </section>

        <footer className="mt-2 flex items-center gap-2 rounded-xl border border-white/20 bg-emerald-900/60 px-3 py-1.5 text-[11px] text-white/95 md:text-xs">
          <ScanLine className="h-4 w-4" />
          Built with FastAPI, YOLO11-cls Nano, React, Tailwind, Lucide, and Framer Motion.
        </footer>
      </div>
    </main>
  );
}

export default App;
