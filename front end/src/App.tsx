import React, { useState } from "react";
import { Home, ScanLine, BookOpen, History, Leaf, AlertTriangle, Clock, Image as ImageIcon, Gauge, ChevronRight, Upload, X, CheckCircle2, Trash2 } from "lucide-react";

interface PredictionResult {
  prediction: string;
  confidence: number;
  top3: { class: string; confidence: number }[];
}

// Symptom database for each disease
const diseaseSymptoms: Record<string, string[]> = {
  "Healthy": ["No visible symptoms", "Normal leaf color", "Healthy tissue"],
  "Leaf Blight": ["Water-soaked lesions", "Browning along leaf margins", "Yellow halos around spots"],
  "Rice Blast": ["Diamond-shaped lesions", "Grayish-white centers", "Brown borders", "Blast pattern on leaves"],
  "Rice Leaffolder": ["Folded leaf blades", "Larval feeding damage", "Scratches on leaf surface"],
  "Rice Stripes": ["Yellow stripe patterns", "Stunted growth signs", "Discolored leaf veins"],
  "Rice Tungro": ["Orange-yellow discoloration", "Stunted plants", "Reduced tillering"]
};

// Severity levels based on confidence
const getSeverity = (confidence: number, isHealthy: boolean): { level: string; color: string; desc: string } => {
  if (isHealthy) return { level: "None", color: "bg-green-500", desc: "Healthy plant detected" };
  if (confidence >= 85) return { level: "High", color: "bg-red-500", desc: "Disease detected with high confidence" };
  if (confidence >= 60) return { level: "Medium", color: "bg-yellow-500", desc: "Disease detected with moderate confidence" };
  return { level: "Low", color: "bg-orange-500", desc: "Possible disease - further analysis recommended" };
};

// Navbar Component
const Navbar = ({ activeTab, setActiveTab }: { activeTab: string; setActiveTab: (tab: string) => void }) => (
  <nav className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
    <div className="max-w-6xl mx-auto px-3 sm:px-4 lg:px-6">
      <div className="flex justify-between items-center h-14">
        <div className="flex items-center gap-2">
          <div className="bg-green-600 p-1.5 rounded-md">
            <Leaf className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm sm:text-base font-bold text-gray-900 leading-tight">WMSU Rice Leaf Disease Detection</h1>
            <p className="text-xs text-gray-500 hidden sm:block">Western Mindanao State University</p>
          </div>
        </div>
        <div className="flex items-center gap-0.5 sm:gap-1">
          {[
            { id: "home", label: "Home", icon: Home },
            { id: "scan", label: "Scan", icon: ScanLine },
            { id: "encyclopedia", label: "Encyclopedia", icon: BookOpen, mobileLabel: "Info" },
            { id: "history", label: "History", icon: History }
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-1 px-2 sm:px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                activeTab === item.id
                  ? "bg-green-600 text-white"
                  : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
              }`}
            >
              <item.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{item.label}</span>
              <span className="sm:hidden">{item.mobileLabel || item.label.slice(0, 4)}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  </nav>
);

// Progress Bar Component
const ProgressBar = ({ value, colorClass }: { value: number; colorClass: string }) => (
  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
    <div
      className={`h-full rounded-full transition-all duration-1000 ${colorClass}`}
      style={{ width: `${value}%` }}
    />
  </div>
);

// Severity Indicator Component
const SeverityIndicator = ({ severity }: { severity: { level: string; color: string; desc: string } }) => (
  <div className="mt-4">
    <div className="flex justify-between text-sm mb-2">
      <span className="text-gray-600 font-medium">Severity Level</span>
      <span className={`font-bold ${
        severity.level === "None" ? "text-green-600" :
        severity.level === "High" ? "text-red-600" :
        severity.level === "Medium" ? "text-yellow-600" : "text-orange-600"
      }`}>{severity.level}</span>
    </div>
    <div className="relative">
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${severity.color}`}
          style={{
            width: severity.level === "None" ? "10%" :
                   severity.level === "Low" ? "33%" :
                   severity.level === "Medium" ? "66%" : "100%"
          }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  </div>
);

// Upload Section Component
const UploadSection = ({ onImageUpload, preview, loading }: {
  onImageUpload: (file: File) => void;
  preview: string | null;
  loading: boolean;
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      onImageUpload(file);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md p-4 sm:p-6">
      <h2 className="text-lg sm:text-xl font-bold text-gray-900 mb-1">Upload Rice Leaf Image</h2>
      <p className="text-sm text-gray-500 mb-4">Drag and drop or click to select an image</p>

      {!preview ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-6 sm:p-8 text-center transition-all ${
            isDragging ? "border-green-500 bg-green-50" : "border-gray-300 hover:border-green-400 hover:bg-gray-50"
          }`}
        >
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onImageUpload(file);
            }}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="cursor-pointer block">
            <div className="bg-green-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
              <Upload className="h-6 w-6 text-green-600" />
            </div>
            <p className="text-sm sm:text-base font-medium text-gray-900 mb-1">Click to upload or drag and drop</p>
            <p className="text-xs text-gray-500">PNG, JPG, JPEG up to 10MB</p>
          </label>
        </div>
      ) : (
        <div className="relative">
          <img
            src={preview}
            alt="Preview"
            className="w-full max-h-64 object-contain rounded-lg"
          />
          <button
            onClick={() => window.location.reload()}
            className="absolute top-2 right-2 bg-white rounded-full p-1.5 shadow-md hover:bg-gray-100"
          >
            <X className="h-4 w-4 text-gray-600" />
          </button>
        </div>
      )}

      {loading && (
        <div className="mt-4 flex items-center justify-center gap-2">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-green-600" />
          <span className="text-sm text-gray-600">Analyzing...</span>
        </div>
      )}
    </div>
  );
};

// Results Card Component
const ResultsCard = ({ result }: { result: PredictionResult }) => {
  const isHealthy = result.prediction.toLowerCase() === "healthy";
  const severity = getSeverity(result.confidence, isHealthy);
  const symptoms = diseaseSymptoms[result.prediction] || ["No specific symptoms recorded"];

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className={`p-4 sm:p-5 ${isHealthy ? "bg-green-50" : "bg-red-50"}`}>
        <div className="flex items-start gap-2 sm:gap-3">
          <div className={`p-1.5 rounded-md ${isHealthy ? "bg-green-100" : "bg-red-100"}`}>
            {isHealthy ? (
              <CheckCircle2 className="h-5 w-5 text-green-600" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-red-600" />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <h3 className={`text-base sm:text-lg font-bold ${isHealthy ? "text-green-800" : "text-red-800"}`}>
              {result.prediction}
            </h3>
            <p className={`text-xs sm:text-sm mt-0.5 ${isHealthy ? "text-green-600" : "text-red-600"}`}>
              {severity.desc}
            </p>
          </div>
        </div>

        <div className="mt-4">
          <div className="flex justify-between items-end mb-1.5">
            <span className="text-xs sm:text-sm font-medium text-gray-700">Confidence Score</span>
            <span className={`text-xl sm:text-2xl font-bold ${
              result.confidence >= 80 ? "text-green-600" :
              result.confidence >= 60 ? "text-yellow-600" : "text-orange-600"
            }`}>{result.confidence.toFixed(1)}%</span>
          </div>
          <ProgressBar
            value={result.confidence}
            colorClass={result.confidence >= 80 ? "bg-green-500" : result.confidence >= 60 ? "bg-yellow-500" : "bg-orange-500"}
          />
        </div>

        {!isHealthy && <SeverityIndicator severity={severity} />}
      </div>

      {/* Top 3 Predictions */}
      {result.top3 && result.top3.length > 0 && (
        <div className="p-4 sm:p-5 border-t border-gray-100 bg-gray-50">
          <h4 className="text-sm sm:text-base font-semibold text-gray-900 mb-3">Top 3 Predictions</h4>
          <div className="space-y-2">
            {result.top3.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between bg-white rounded-md p-2 shadow-sm">
                <div className="flex items-center gap-2">
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                    idx === 0 ? "bg-green-500 text-white" :
                    idx === 1 ? "bg-yellow-500 text-white" : "bg-gray-400 text-white"
                  }`}>
                    {idx + 1}
                  </span>
                  <span className="text-sm text-gray-700">{item.class}</span>
                </div>
                <span className="text-sm font-medium text-gray-900">{item.confidence}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="p-4 sm:p-5 border-t border-gray-100">
        <h4 className="text-sm sm:text-base font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-orange-500" />
          Visual Symptoms Detected
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {symptoms.map((symptom, idx) => (
            <div key={idx} className="flex items-center gap-2 bg-gray-50 rounded-md p-2">
              <div className="w-1.5 h-1.5 rounded-full bg-orange-400 flex-shrink-0" />
              <span className="text-xs sm:text-sm text-gray-700 truncate">{symptom}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Sidebar Component
const Sidebar = ({ onScanAnother, result }: { onScanAnother: () => void; result: PredictionResult | null }) => (
  <div className="space-y-4">
    {/* Actions */}
    <div className="bg-white rounded-xl shadow-md p-4">
      <h3 className="text-sm sm:text-base font-semibold text-gray-900 mb-3">Actions</h3>
      <button
        onClick={onScanAnother}
        className="w-full flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium py-2.5 px-4 rounded-lg transition-colors"
      >
        <ScanLine className="h-4 w-4" />
        Scan Another
      </button>
    </div>

    {/* Important Note */}
    <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
      <h3 className="text-sm sm:text-base font-semibold text-blue-900 mb-2">Important Note</h3>
      <p className="text-xs text-blue-700 leading-relaxed">
        This is an AI-based assessment. For critical decisions, please consult with an agricultural expert.
      </p>
    </div>

    {/* Scan Statistics */}
    {result && (
      <div className="bg-white rounded-xl shadow-md p-4">
        <h3 className="text-sm sm:text-base font-semibold text-gray-900 mb-3">Scan Statistics</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2 text-gray-600">
              <Clock className="h-4 w-4" />
              <span className="text-xs">Analysis Time</span>
            </div>
            <span className="text-xs font-medium text-gray-900">2.3s</span>
          </div>
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2 text-gray-600">
              <ImageIcon className="h-4 w-4" />
              <span className="text-xs">Image Quality</span>
            </div>
            <span className="text-xs font-medium text-green-600">Excellent</span>
          </div>
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2 text-gray-600">
              <Gauge className="h-4 w-4" />
              <span className="text-xs">Model Version</span>
            </div>
            <span className="text-xs font-medium text-gray-900">v3.2.1</span>
          </div>
        </div>
      </div>
    )}
  </div>
);

// History item type
interface HistoryItem {
  id: string;
  date: string;
  preview: string;
  prediction: string;
  confidence: number;
  top3: Array<{class: string; confidence: number}>;
}

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState("scan");
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    const saved = localStorage.getItem('rice_disease_history');
    return saved ? JSON.parse(saved) : [];
  });

  // Compress image to reduce localStorage size
  const compressImage = (dataUrl: string, maxWidth = 200): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const scale = maxWidth / img.width;
        canvas.width = maxWidth;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.7)); // Compress to 70% quality
      };
      img.onerror = () => resolve(dataUrl); // Fallback to original if compression fails
      img.src = dataUrl;
    });
  };

  // Save history to localStorage with compressed image
  const saveToHistory = async (previewUrl: string, result: PredictionResult) => {
    // Compress image before saving to prevent localStorage quota exceeded
    const compressedPreview = await compressImage(previewUrl);

    const newItem: HistoryItem = {
      id: Date.now().toString(),
      date: new Date().toLocaleString(),
      preview: compressedPreview,
      prediction: result.prediction,
      confidence: result.confidence,
      top3: result.top3 || []
    };
    const updated = [newItem, ...history].slice(0, 20); // Keep last 20 scans to save space
    setHistory(updated);
    localStorage.setItem('rice_disease_history', JSON.stringify(updated));
  };

  const clearHistory = () => {
    if (window.confirm('Are you sure you want to delete ALL scan history? This cannot be undone.')) {
      setHistory([]);
      localStorage.removeItem('rice_disease_history');
    }
  };

  const deleteHistoryItem = (id: string) => {
    if (window.confirm('Are you sure you want to delete this scan?')) {
      const updated = history.filter(item => item.id !== id);
      setHistory(updated);
      localStorage.setItem('rice_disease_history', JSON.stringify(updated));
    }
  };

  const handleImageUpload = async (file: File) => {
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", file);

      console.log("Sending request to API at http://127.0.0.1:5005/predict");
      const response = await fetch("http://127.0.0.1:5005/predict", {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
      }

      const data = await response.json();
      console.log("Prediction result:", data);
      
      // Ensure data has required fields
      if (!data.prediction || data.confidence === undefined) {
        throw new Error("Invalid response from server");
      }
      
      // Format the result properly
      const formattedResult: PredictionResult = {
        prediction: data.prediction,
        confidence: data.confidence,
        top3: data.top3 || []
      };
      
      setResult(formattedResult);
      
      // Save to history with compressed image
      const previewUrl = preview || URL.createObjectURL(file);
      await saveToHistory(previewUrl, formattedResult);
    } catch (err: any) {
      console.error("Fetch error:", err);
      setError(err.message || "Failed to get prediction. Make sure the backend is running on port 5005.");
    } finally {
      setLoading(false);
    }
  };

  const handleScanAnother = () => {
    setPreview(null);
    setResult(null);
    setError(null);
  };

  // Home View
  if (activeTab === "home") {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
          <div className="text-center">
            <div className="bg-green-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
              <Leaf className="h-10 w-10 text-green-600" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-3">
              WMSU Rice Leaf Disease Detection
            </h1>
            <p className="text-base sm:text-lg text-gray-600 mb-6 max-w-xl mx-auto">
              AI-powered system for detecting rice leaf diseases. Upload an image to get instant analysis.
            </p>
            <button
              onClick={() => setActiveTab("scan")}
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg text-base transition-colors"
            >
              Start Scanning
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Encyclopedia View
  if (activeTab === "encyclopedia") {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
        <div className="max-w-5xl mx-auto px-3 sm:px-4 py-4 sm:py-6">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-4">Disease Encyclopedia</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {Object.entries(diseaseSymptoms).map(([disease, symptoms]) => (
              <div key={disease} className="bg-white rounded-lg shadow-md p-4">
                <h3 className="text-base font-bold text-gray-900 mb-2">{disease}</h3>
                <ul className="space-y-1">
                  {symptoms.map((symptom, idx) => (
                    <li key={idx} className="text-xs sm:text-sm text-gray-600 flex items-start gap-1.5">
                      <ChevronRight className="h-3 w-3 text-green-500 flex-shrink-0 mt-0.5" />
                      <span className="truncate">{symptom}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // History View
  if (activeTab === "history") {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
        <div className="max-w-4xl mx-auto px-3 sm:px-4 py-4 sm:py-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl sm:text-2xl font-bold text-gray-900">Scan History</h2>
            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-sm text-red-600 hover:text-red-700 font-medium px-3 py-1.5 rounded-md hover:bg-red-50 transition-colors"
              >
                Clear History
              </button>
            )}
          </div>
          
          {history.length === 0 ? (
            <div className="bg-white rounded-lg shadow-md p-6 sm:p-8 text-center text-gray-500">
              <History className="h-10 w-10 mx-auto mb-3 text-gray-400" />
              <p className="text-sm">No scan history available yet. Start by scanning a rice leaf!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((item) => (
                <div key={item.id} className="bg-white rounded-lg shadow-md p-3 sm:p-4">
                  <div className="flex gap-3">
                    <img
                      src={item.preview}
                      alt="Scan"
                      className="w-20 h-20 object-cover rounded-md flex-shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="text-xs text-gray-500">{item.date}</p>
                          <h3 className={`font-semibold text-sm sm:text-base ${
                            item.prediction.toLowerCase() === 'healthy' 
                              ? 'text-green-600' 
                              : 'text-red-600'
                          }`}>
                            {item.prediction}
                          </h3>
                        </div>
                        <span className={`text-sm font-bold ${
                          item.confidence >= 80 ? 'text-green-600' :
                          item.confidence >= 60 ? 'text-yellow-600' : 'text-orange-600'
                        }`}>
                          {item.confidence.toFixed(1)}%
                        </span>
                      </div>
                      
                      {item.top3 && item.top3.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-gray-100">
                          <p className="text-xs text-gray-500 mb-1">Top predictions:</p>
                          <div className="flex gap-2 flex-wrap">
                            {item.top3.slice(0, 3).map((pred, idx) => (
                              <span
                                key={idx}
                                className={`text-xs px-2 py-0.5 rounded-full ${
                                  idx === 0 
                                    ? 'bg-green-100 text-green-700' 
                                    : 'bg-gray-100 text-gray-600'
                                }`}
                              >
                                {pred.class} ({pred.confidence}%)
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    {/* Delete Button */}
                    <button
                      onClick={() => deleteHistoryItem(item.id)}
                      className="flex-shrink-0 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-colors ml-2"
                      title="Delete this scan"
                    >
                      <Trash2 className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Scan View (Main)
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <div className="max-w-5xl mx-auto px-3 sm:px-4 lg:px-6 py-4 sm:py-6">
        {/* Header */}
        <div className="text-center mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900">Detection Results</h2>
          <p className="text-xs sm:text-sm text-gray-500 mt-1">Analysis completed on {new Date().toLocaleDateString()}</p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-3xl mx-auto mb-4 bg-red-50 border border-red-200 text-red-700 px-3 py-2 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* Main Content Grid - Responsive */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
          {/* Left Column - Upload & Results */}
          <div className="lg:col-span-2 space-y-4">
            <UploadSection
              onImageUpload={handleImageUpload}
              preview={preview}
              loading={loading}
            />

            {result && <ResultsCard result={result} />}
          </div>

          {/* Right Column - Sidebar */}
          <div className="lg:col-span-1">
            <Sidebar onScanAnother={handleScanAnother} result={result} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;