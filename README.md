# Rice Classifier Website

Production-ready rice variety and condition classifier powered by FastAPI + YOLO11-cls, with a React frontend served as a website from the same backend process.

## Project Structure

- `backend/` - FastAPI app and trained model weights
- `frontend/` - React app source

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
