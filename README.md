# Rice Classifier Website

Production-ready rice variety and condition classifier powered by FastAPI + YOLO11-cls, with a React frontend served as a website from the same backend process.

## Project Structure

- `backend/` - FastAPI app and trained model weights
- `frontend/` - React app source

## Documentation

- **Professor PDF format (~40+ pages in Word at 12pt/1.5 spacing):** [docs/SYSTEM_DOCUMENTATION_PROFESSOR_FORMAT_EXPANDED.md](docs/SYSTEM_DOCUMENTATION_PROFESSOR_FORMAT_EXPANDED.md) — long-form thesis-style body + appendices; regenerate with `myenv\Scripts\python.exe docs\build_long_professor_doc.py`.
- **Professor PDF format (concise edition):** [docs/SYSTEM_DOCUMENTATION_PROFESSOR_FORMAT.md](docs/SYSTEM_DOCUMENTATION_PROFESSOR_FORMAT.md) — same template alignment, shorter read.
- **Thesis-style (Chapters 1–5):** [docs/SYSTEM_DOCUMENTATION_THESIS_FORMAT_CHAPTERS_1_TO_5.md](docs/SYSTEM_DOCUMENTATION_THESIS_FORMAT_CHAPTERS_1_TO_5.md) — introduction, RRL, methodology, results, summary/recommendations.
- **Code reference (Chapters 1–5):** [docs/CODE_REFERENCE_CHAPTERS_1_TO_5.md](docs/CODE_REFERENCE_CHAPTERS_1_TO_5.md) — repository map and implementation walkthrough.

## Run as a Website (Production-style)

1. Install backend dependencies:

```bash
myenv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

2. Build frontend static files:

```bash
cd frontend
npm install
npm run build
```

3. Start website server from `backend/`:

```bash
cd ../backend
../myenv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Open in browser:

- Website: `http://localhost:8000`
- API endpoint: `http://localhost:8000/predict`

## Notes

- Frontend uses same-origin API by default (`/predict`) for deployment simplicity.
- Place trained weights at `backend/yolo11n-cls.pt`.
