import { useMemo, useState } from "react";

type PredictionResponse = {
  label: string;
  confidence: number;
  all_scores: Record<string, number>;
};

const API_URL = "http://127.0.0.1:8000/predict";
const REQUEST_TIMEOUT_MS = 15000;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const sortedScores = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.all_scores).sort((a, b) => b[1] - a[1]);
  }, [result]);

  const onFileChange = (selected: File | null) => {
    if (!selected) return;

    if (!selected.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }

    setFile(selected);
    setResult(null);
    setError("");
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const onSubmit = async () => {
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
        const detail = body.detail || "Prediction failed";
        throw new Error(detail);
      }

      const data = (await response.json()) as PredictionResponse;
      setResult(data);
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
    <main style={{ maxWidth: 920, margin: "0 auto", padding: 24 }}>
      <h1>Rice Variety and Condition Classifier</h1>
      <p style={{ color: "#475569" }}>
        Upload a rice image and classify it with YOLO11-cls.
      </p>

      <section
        style={{
          background: "white",
          padding: 16,
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          marginTop: 16,
        }}
      >
        <input
          type="file"
          accept="image/*"
          onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
        />
        <button
          onClick={onSubmit}
          disabled={loading}
          style={{ marginLeft: 12, padding: "8px 14px" }}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>

        {previewUrl && (
          <div style={{ marginTop: 14 }}>
            <img
              src={previewUrl}
              alt="Uploaded preview"
              style={{ maxWidth: "100%", maxHeight: 320, borderRadius: 8 }}
            />
          </div>
        )}
      </section>

      {error && (
        <p style={{ color: "#b91c1c", marginTop: 16, fontWeight: 600 }}>{error}</p>
      )}

      {result && (
        <section
          style={{
            marginTop: 18,
            background: "white",
            padding: 16,
            borderRadius: 12,
            border: "1px solid #e2e8f0",
          }}
        >
          <h2 style={{ marginTop: 0 }}>Prediction Result</h2>
          <p>
            <strong>Label:</strong> {result.label}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
          </p>

          <div style={{ marginTop: 8 }}>
            <div
              style={{
                width: "100%",
                background: "#e2e8f0",
                borderRadius: 999,
                overflow: "hidden",
                height: 12,
              }}
            >
              <div
                style={{
                  width: `${Math.max(0, Math.min(result.confidence * 100, 100))}%`,
                  height: "100%",
                  background: "#16a34a",
                }}
              />
            </div>
          </div>

          <h3>All Class Scores</h3>
          <ul style={{ paddingLeft: 18 }}>
            {sortedScores.map(([name, score]) => (
              <li key={name}>
                {name}: {(score * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

export default App;
