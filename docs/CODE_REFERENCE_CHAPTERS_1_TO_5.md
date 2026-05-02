# Code reference: Chapters 1–5

This document describes the **rafsan** rice leaf disease detection system as implemented in the repository  
[github.com/alfahadadian04-blip/rafsan](https://github.com/alfahadadian04-blip/rafsan.git).

It is organized into five chapters: overview, backend, frontend, model and evaluation tooling, then build and operations.

---

## Chapter 1 — Project overview and repository map

### 1.1 Purpose

The system lets users **upload or paste a rice leaf image** in the browser, sends it to a **Python API**, and receives a **disease class label** with **confidence scores** and **reliability hints** (for example low resolution or missing camera metadata). The same server can host the **web UI** as static files so one process serves both UI and API.

### 1.2 Technology stack

| Layer | Technology |
|--------|------------|
| Web UI | React 18, TypeScript, Vite, Tailwind CSS, Framer Motion, Lucide icons |
| API | FastAPI, Uvicorn |
| ML | Ultralytics YOLO11 **classification** (`yolo11n-cls.pt`) |
| Images | Pillow (decode, EXIF transpose, preprocessing) |

### 1.3 Top-level layout (source you care about)

| Path | Role |
|------|------|
| `backend/` | FastAPI application, model weights, `requirements.txt` |
| `frontend/` | React source (`src/`), Vite config, `package.json` |
| `evaluate_accuracy.py` | Offline validation script (top-1 / top-5 on folder datasets) |
| `run-fullstack.bat` | Windows helper: build frontend, start Uvicorn |
| `README.md` | Quick run instructions |
| `.gitignore` | Excludes `myenv/`, `node_modules/`, `dist/`, large datasets, `runs/`, etc. |

Large folders such as `dataset/`, `dataset_original_split/`, `runs/`, and `myenv/` are intentionally **not** part of the GitHub source tree for size and reproducibility reasons; you keep them locally for training and evaluation.

### 1.4 Design choice: single origin

In production-style use, the browser loads the app from the **same host and port** as `/predict`. The frontend defaults to **`/predict`** (relative URL), which avoids CORS configuration for that deployment mode. The backend still enables permissive CORS for other deployment patterns.

---

## Chapter 2 — Backend (`backend/`)

### 2.1 Entry module

All server behavior lives in **`backend/main.py`**. The ASGI app instance is named **`app`** and is what Uvicorn loads (`main:app`).

### 2.2 Application lifespan and model loading

- A **lifespan** context runs when the server starts and stops.
- On startup it calls **`ModelSingleton.load()`**, which constructs a **`YOLO`** instance from **`backend/yolo11n-cls.pt`** (or the filename alone if the path is missing).
- On shutdown it releases the singleton reference.

This avoids reloading the model on every HTTP request.

### 2.3 HTTP endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Simple JSON `{"status": "ok"}` for liveness checks |
| `POST` | `/predict` | Multipart upload field **`image`**: returns JSON with label, scores, reliability fields |

If **`frontend/dist`** exists, additional routes serve the SPA:

- **`/assets/...`** — static build assets from `frontend/dist/assets`
- **`/`** — `index.html`
- **`/{path}`** — file from `dist` if it exists, otherwise **`index.html`** for client-side routing

### 2.4 `/predict` pipeline (conceptual order)

1. Validate **`Content-Type`** is an image type.
2. Read raw bytes; reject empty uploads.
3. Open with Pillow; apply **`ImageOps.exif_transpose`** so orientation matches how the user sees the photo.
4. Derive **camera metadata flags** from EXIF (e.g. make/model) for messaging.
5. Convert to RGB; record **low resolution** if below configured minimum width/height.
6. Compute **grayscale brightness spread**; reject images that are effectively too flat/dark (quality gate).
7. Build **multiple inference views** (original, center-crop resize, mirror, mild contrast boost) and run **`model.predict`** on each.
8. **Average** class probabilities across views that returned valid `probs`.
9. Compute **top-1**, **margin vs second class**, and **entropy** over the averaged distribution.
10. Set **`is_reliable`** from thresholds on confidence, margin, entropy, and resolution; build a human-readable **`message`** (warnings vs “passed checks”).
11. Return JSON: **`label`**, **`confidence`**, **`all_scores`**, **`is_reliable`**, **`message`**, **`has_camera_metadata`**, **`has_low_resolution`**, **`ensemble_views`**.

Constants such as minimum image size, confidence floor, margin, and entropy cap are defined at the top of **`main.py`** so you can tune behavior without changing control flow.

### 2.5 Dependencies

**`backend/requirements.txt`** lists runtime packages: FastAPI, Uvicorn, Ultralytics, Pillow, `python-multipart` (for file uploads).

---

## Chapter 3 — Frontend (`frontend/`)

### 3.1 Bootstrap

- **`frontend/index.html`** — Vite HTML shell.
- **`frontend/src/main.tsx`** — mounts **`App`** into the DOM root with React 18 `createRoot`.
- **`frontend/src/index.css`** — global styles (including Tailwind layers if configured there).

### 3.2 Main application: `App.tsx`

**`frontend/src/App.tsx`** is the primary UI module:

- **Navigation**: tabs — Home, Scan, Encyclopedia, History (`NAV_ITEMS`).
- **Clock**: header shows a ticking local date/time.
- **Scan**: file input, optional **paste image from clipboard**, preview URL, **Scan** button calling **`POST`** with **`FormData`** and multipart field name **`image`** (same as the FastAPI parameter name in `backend/main.py`).
- **API URL**: `import.meta.env.VITE_API_URL ?? "/predict"` so production uses same-origin **`/predict`**.
- **Timeout**: `AbortController` with a fixed millisecond budget to avoid hung requests.
- **State**: loading, error, warning (server message / reliability copy), **`result`**, **`history`** (session-only list of scans with object URLs for thumbnails).
- **Encyclopedia**: static structured copy for each disease class aligned with the model’s taxonomy.
- **History**: delegates list UI to **`HistoryPanel`**; supports delete one and clear all with **`URL.revokeObjectURL`** to avoid leaks.

### 3.3 History component: `HistoryPanel.tsx`

**`frontend/src/components/HistoryPanel.tsx`** renders the scrollable history list and action buttons. It receives **`history`**, **`onDelete`**, and **`onClearAll`** from **`App`** (presentational component pattern).

### 3.4 Assets

Images under **`frontend/src/assets/`** (for example background and header logo) are imported in **`App.tsx`** and bundled by Vite into hashed filenames under **`dist/assets/`** at build time.

### 3.5 Build tooling

- **`vite.config.ts`** — Vite + React plugin.
- **`tailwind.config.js`**, **`postcss.config.js`** — Tailwind pipeline.
- **`package.json`** — scripts: `dev`, `build`, `preview`.

---

## Chapter 4 — Model weights and offline evaluation

### 4.1 Weights file

The deployed classifier weights are expected at **`backend/yolo11n-cls.pt`**. The API loads this path on startup (see Chapter 2). Replacing this file with a newly trained **`best.pt`** (same architecture) is the standard way to ship an updated model without code changes.

### 4.2 `evaluate_accuracy.py`

**`evaluate_accuracy.py`** (repository root) is a **batch evaluation** utility, not used at runtime by the website:

- Loads a **`.pt`** model (default: `backend/yolo11n-cls.pt`).
- For each dataset root you pass (default: `dataset` and `dataset_original_split`), runs Ultralytics **`model.val(data=..., split="val")`** when a **`val/`** split exists.
- Prints **Top-1** and **Top-5** accuracy as percentages.

Use it to compare checkpoints against your local folder datasets. Paths are ignored by Git when datasets are absent on another machine.

### 4.3 Training outputs

Fine-tuning with Ultralytics typically writes under **`runs/`** (ignored in Git in your setup). Those artifacts are **documentation of experiments**, not dependencies of the running web app.

---

## Chapter 5 — Building, running, and architecture summary

### 5.1 Developer / operator workflow

1. Create a Python virtual environment and install **`backend/requirements.txt`**.
2. In **`frontend/`**, run **`npm install`** then **`npm run build`** to produce **`frontend/dist/`**.
3. From **`backend/`**, start **`uvicorn main:app --host … --port …`**.

On Windows, **`run-fullstack.bat`** automates steps 2–3 (with an optional **`nobuild`** argument if `dist` already exists). See **`README.md`** for the same steps in plain commands.

### 5.2 Runtime architecture (one sentence)

**One Uvicorn worker process** hosts **FastAPI**, loads **YOLO** once, exposes **`/predict`** and **`/health`**, and optionally serves the **Vite-built React app** from **`frontend/dist`** so users open a single URL for both UI and inference.

### 5.3 Data persistence

Scan **history** is kept in **React state** in the browser for the current session; refreshing the page clears it unless you later add `localStorage`/backend persistence.

### 5.4 Suggested extensions (not implemented in code today)

- User accounts and server-side history database.
- Separate dev server for Vite with **`VITE_API_URL`** pointing at the API.
- Git LFS or release assets for large datasets and optional weight variants.

---

*Document generated to match the codebase layout in [alfahadadian04-blip/rafsan](https://github.com/alfahadadian04-blip/rafsan.git). Update this file when you change endpoints, env vars, or folder conventions.*
