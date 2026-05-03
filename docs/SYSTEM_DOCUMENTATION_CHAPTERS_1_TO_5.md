# System Documentation — Chapters 1 to 5

**System title:** WMSU Rice Disease Detection  

**Repository:** [https://github.com/alfahadadian04-blip/rafsan.git](https://github.com/alfahadadian04-blip/rafsan.git)  

**Implementation summary:** FastAPI backend (`backend/main.py`), Ultralytics **YOLO11n-cls** weights at `backend/yolo11n-cls.pt`, React + TypeScript + Vite frontend (`frontend/src/`). API version **2026.1.0**.  

**Document purpose:** Academic-style chapters 1–5 describing the **actual** behavior of the software as implemented in this repository.  

**Last updated:** May 2026.

---

## Chapter 1 — Introduction

### 1.1 Background of the Study

Rice (*Oryza sativa* L.) is a staple crop for a large share of the world’s population. Farmers, students, and extension workers often inspect leaves first, because discoloration, lesions, stripes, folding, or stunting can signal disease complexes, insect damage, or poor growing conditions. Traditional diagnosis depends on experience and reference guides. Digital cameras on phones are now common, so the same leaf photographs can be processed by machine learning models that suggest which **label** in a fixed list best matches the visible pattern, provided the model was trained on representative images and its limits are explained clearly.

This project implements **WMSU Rice Disease Detection**, a browser-oriented application that combines a deep learning **image classifier** with a simple web interface. A user uploads or pastes a photograph of a rice leaf, receives a predicted class with a numeric confidence score and a full score table, and can read short educational notes in an encyclopedia-style section of the app. The source code is kept in a public Git repository so that thresholds, dependencies, and API behavior can be verified directly instead of relying only on high-level descriptions.

### 1.2 Problem Statement

A model that scores well on a clean validation folder may still behave unpredictably on dark, blurry, or heavily compressed images taken in the field. Neural classifiers also tend to return a single “best” class even when several classes receive similar scores. The problem addressed by this work is how to deliver a **practical web-based** rice leaf classification service that (1) runs on a stack suitable for student deployment (Python API plus static React build), and (2) reports not only the top label and probabilities but also **structured cues** about input quality and prediction sharpness, so users are less likely to treat every output as equally trustworthy.

### 1.3 Objectives of the Study

The **general objective** is to design, implement, and document a web-based rice leaf condition classification system backed by a deep learning model trained on rice leaf imagery, with clear evaluation on held-out validation data.

The **specific objectives** are: (1) to use a **YOLO11 classification (nano)** model that discriminates among **six** classes aligned with the application (Leaf Blight, Rice Blast, Rice Leaffolder, Rice Stripes, Rice Tungro, Healthy Leaf); (2) to implement a **REST API** (`POST /predict`) that validates uploads, applies preprocessing and **multi-view inference**, and returns JSON including reliability-related fields; (3) to implement a **React** client with Home, Scan, Encyclopedia, and History tabs, including clipboard image paste and session-local history; (4) to provide a script **`evaluate_accuracy.py`** that prints **top-1** and **top-5** validation accuracy for local `dataset/` and `dataset_original_split/` roots when present; and (5) to ground training and reporting on the public **Rice Leaf Disease Dataset** hosted on Mendeley Data [1].

### 1.4 Scope and Limitations

The system is limited to the **six** trained classes; it does not identify rice varieties, growth stages, or specific pathogens beyond what those labels encode. Inference is **single-image** over HTTP; there is no batch satellite or drone pipeline. Scan **history** exists only in the browser for the current session and is lost on page reload. The API enables **CORS** for all origins in code (`allow_origins=["*"]`), which is convenient for classroom demos but is not ideal for public production without a reverse proxy and stricter policy.

Outputs are **advisory** only: they support learning and triage, not legal phytosanitary certification or automatic pesticide prescriptions. Training-time augmentation and exact train commands live in the Ultralytics workflow on the developer machine; the repository’s runtime artifact assumed by the server is **`backend/yolo11n-cls.pt`**.

### 1.5 Significance of the Study

The work helps **students** connect plant health vocabulary to measurable model behavior and accuracy figures. It gives **instructors** a self-contained example of full-stack ML deployment (load model at startup, serve JSON, consume from React). **Researchers** can reuse the same pipeline to compare other checkpoints or calibration methods while keeping the API contract stable. For **agricultural extension** demonstrations, the encyclopedia tab links model outputs to short symptom and management notes, reinforcing that software should complement field experts rather than replace them.

---

## Chapter 2 — Related Review of Literature and Technologies

### 2.1 Image-Based Plant Disease and Rice Leaf Classification

Deep learning for image-based plant disease detection has been an active research area for many years; a widely cited example is the study by Mohanty *et al.* that used large labeled plant image sets with convolutional networks [2]. Work focused on rice leaves has similarly demonstrated that deep convolutional architectures can separate disease categories when training data are well structured [3]. Review-level discussion emphasizes that **dataset diversity** and **domain shift** strongly affect whether laboratory results transfer to farmer-uploaded photos [4]. Those themes justify using a public rice leaf corpus [1] and adding explicit checks in the inference API when user images differ from training conditions.

### 2.2 Transfer Learning and Compact Models

Transfer learning—starting from weights learned on broad natural-image tasks and adapting to a smaller agricultural dataset—typically improves sample efficiency compared with training all layers from scratch [5]. Compact models reduce memory and latency, which matters when inference runs on **CPU-only** classroom machines. This project uses the **nano** variant of YOLO11 **classification** (`yolo11n-cls`), exposed through the Ultralytics library [6], as a deliberate trade-off between capacity and deployment simplicity.

### 2.3 Reliability, Calibration, and the User Interface

Raw classifier scores are not always well **calibrated** as true probabilities; work such as Guo *et al.* discusses calibration of modern neural networks [7]. From a systems perspective, it is therefore reasonable to expose **margin** (gap between the top two classes), **entropy** of the distribution, and simple **input-quality** measures, and to phrase uncertain cases carefully in the returned `message` string. The implemented backend follows this idea with fixed numeric thresholds documented in Chapter 3.

### 2.4 Web Stack Technologies

**FastAPI** provides a typed, asynchronous Python web framework suited to file uploads and JSON APIs [8]. **React** with **Vite** is a common toolchain for interactive single-page applications [9], [10]. In this repository, when `frontend/dist` exists after `npm run build`, the FastAPI app can serve the built static files and the `/assets` directory from the same process as `/predict`, which yields a **single-origin** deployment suitable for local demonstrations (`http://localhost:8000`).

### 2.5 Comparison with Typical Prior Work and Role of This System

Many academic papers report accuracy on a fixed test set but distribute only a notebook or weights file without a maintained web client. Commercial apps may offer polished branding but hide model version and decision rules. This **rafsan** implementation differs in concrete, inspectable ways: the **`build_inference_views`** function defines exactly four deterministic views whose probabilities are **averaged** before choosing the top class; the **`/predict`** response always includes **`all_scores`**, **`is_reliable`**, **`message`**, **`has_camera_metadata`**, **`has_low_resolution`**, and **`ensemble_views`**; and the **Encyclopedia** tab in `App.tsx` uses the same six class names as the model head, so the UI and the network stay aligned. The contribution is therefore best understood as **applied systems integration** with transparent behavior, rather than as a new neural architecture.

---

## Chapter 3 — Methodology

### 3.1 System Architecture

The system follows a layered layout. The **client layer** is the React application compiled to static files under `frontend/dist/`. The **service layer** is `backend/main.py`, a FastAPI application that exposes `GET /health` (returns `{"status": "ok"}`) and `POST /predict` for classification. If the distribution folder is present, the same process serves the SPA and static assets so users visit one origin for both UI and API. The **model layer** is a singleton `ModelSingleton` that loads `YOLO` once during application lifespan from `backend/yolo11n-cls.pt` if the file exists, otherwise from the configured filename string so Ultralytics can resolve pretrained weights. No server-side database is used; history is stored only in browser state.

### 3.2 Technologies Used

The backend dependencies listed in `backend/requirements.txt` are **FastAPI**, **Uvicorn**, **Ultralytics**, **python-multipart**, and **Pillow**. The frontend uses **React 18**, **TypeScript**, **Vite 6**, **Tailwind CSS 3**, **Lucide React** icons, and **Framer Motion** (see `frontend/package.json`). Model training and validation are performed with the Ultralytics ecosystem; the shipped artifact for the running website is the `.pt` classification file beside `main.py`.

### 3.3 Data Collection

The project’s rice leaf photographs are drawn from the **Rice Leaf Disease Dataset** published on Mendeley Data [1]. After download, images are organized for Ultralytics **classification** training into folder trees `train/<class_name>/` and `val/<class_name>/`, where each subfolder name matches one of the six labels the classifier must learn. In the development workspace used for this document, two parallel roots were maintained: **`dataset/`** (with 3106 training images and 1532 validation images across six classes, as reported by Ultralytics during `evaluate_accuracy.py`) and **`dataset_original_split/`** (2499 train and 629 validation images), allowing comparison between an augmented or reorganized split and a more conservative split. Academic citation of the image source should use reference [1]:

[1] Mendeley Data, "Rice Leaf Disease Dataset," Available: https://data.mendeley.com/datasets/vwv3nry3wr/1

### 3.4 Data Preprocessing

**Training-time** resizing and augmentation are governed by Ultralytics defaults and any custom settings the trainer used when producing `yolo11n-cls.pt`; those steps are not hard-coded in `main.py`. **Inference-time** preprocessing is fully specified in `predict`. Uploaded bytes must have an `image/*` content type and non-empty body. Pillow opens the file; **`ImageOps.exif_transpose`** corrects orientation from EXIF. The image is converted to **RGB**. EXIF tags 271 and 272 are read to set **`has_camera_metadata`**. Width and height below **224** pixels set **`has_low_resolution`** to true (the reliability rule still uses this flag even though the image is not rejected solely for size). A grayscale copy is used to compute pixel standard deviation; if it is below **18.0**, the handler responds with HTTP **400** and the message asking for a clearer, well-lit photo, which prevents nearly flat images from entering the model.

### 3.5 Model and Algorithm Used

The classifier is **YOLO11n-cls** [6]. For each accepted image, **`build_inference_views`** returns four tensors as PIL images: the RGB canvas after EXIF correction; a **square center crop** resized back to the original width and height with bicubic resampling; a **horizontal mirror**; and a **contrast-enhanced** copy with factor **1.08**. Each view is passed through `model.predict(..., verbose=False)`. Class probability vectors from `prediction.probs.data` are accumulated and divided by the number of valid views to produce an **averaged** distribution. The predicted **`label`** is the class name at the argmax of that average; **`confidence`** is that class’s score. The implementation also computes the margin between the top and second class and the **Shannon entropy** of the averaged distribution (with a small epsilon inside the logarithm for stability). **`is_reliable`** is true only when the top score is at least **0.68**, the margin is at least **0.12**, entropy is at most **1.35**, and the image is not low-resolution by the 224-pixel rule; otherwise the same argmax label is still returned but **`is_reliable`** is false and **`message`** collects human-readable warnings (missing camera metadata, low resolution, or weak confidence).

### 3.6 System Workflow

Step **one**, the operator builds the frontend with `npm run build` inside `frontend/` and starts Uvicorn on `main:app` from the `backend/` directory, optionally after installing Python dependencies into the project virtual environment. Step **two**, at startup the lifespan context loads the YOLO weights into memory so the first request does not pay the full disk read penalty alone. Step **three**, the user opens the site in a browser; if static files are mounted, the Home tab loads from `index.html`. Step **four**, on the Scan tab the user chooses an image or uses **Paste Image**, which uses the Clipboard API when available. Step **five**, pressing **Run Scan** builds `FormData` with the field name **`image`** and sends `POST` to `/predict` (or to the URL in environment variable **`VITE_API_URL`** during split development), with a **15-second** abort timeout implemented via `AbortController` in `App.tsx`. Step **six**, the server validates the upload, runs preprocessing and the four-view prediction pipeline, and returns JSON. Step **seven**, the client displays the predicted label, numeric confidence, sorted **`all_scores`**, and warning text when **`is_reliable`** is false, and appends the scan to **History** with a blob URL thumbnail. Step **eight**, the Encyclopedia tab shows static symptom and action paragraphs keyed by the same six class names for interpretation. Step **nine**, for offline metrics the developer runs **`python evaluate_accuracy.py`** from the repository root, which loads `backend/yolo11n-cls.pt` and calls `model.val(data=<root>, split="val")` for each existing dataset root, printing top-1 and top-5 percentages.

---

## Chapter 4 — Results and Discussion

### 4.1 Quantitative Validation Outputs

Validation was executed with **`evaluate_accuracy.py`** on the machine used for this write-up (Ultralytics **8.4.43**, **Python 3.14.3**, **PyTorch 2.11.0+cpu**, Intel **Core i5-1155G7**, CPU inference). The script reported **YOLO11n-cls** with about **1.53 million** parameters and approximately **3.2 GFLOPs** for the fused model summary line emitted during validation.

On the **`dataset/`** root, the validator scanned **1532** images in **`val/`** across six classes and printed **top-1 accuracy 86.55%** and **top-5 accuracy 99.74%**. On **`dataset_original_split/`**, it scanned **629** validation images and printed **top-1 accuracy 81.40%** and **top-5 accuracy 99.84%**. The gap in top-1 performance between the two roots is expected when splits differ in size, augmentation, or class balance; top-5 staying near ceiling for only **six** classes shows that the true label almost always appears among the strongest few responses even when the argmax is wrong.

These numbers are **not** hard-coded in the application; they are produced by rerunning the script whenever weights or splits change. If a dataset folder is missing, the script exits with an error for the default configuration rather than fabricating metrics.

### 4.2 Runtime Behavior and API Outputs

The live **`/predict`** endpoint returns a JSON object with keys **`label`**, **`confidence`**, **`all_scores`** (a map from class name to averaged probability), **`is_reliable`**, **`message`**, **`has_camera_metadata`**, **`has_low_resolution`**, and **`ensemble_views`** (the count of views that returned valid probability tensors, normally four). Extremely dark or flat images are rejected before inference with HTTP 400, which is visible in the Scan tab as an error message parsed from the response body. When predictions complete, the UI lists all classes sorted by score so students can see near-ties between visually similar conditions.

### 4.3 Strengths and Limitations

**Strengths** include end-to-end reproducibility from repository to running site, small model footprint suitable for CPU deployment, deterministic multi-view averaging that reduces sensitivity to mild framing and contrast changes, and explicit reliability messaging. **Limitations** include dependence on the six-label taxonomy from training data [1], no guarantee of generalization to unseen farms or cultivars, session-only history, open CORS in the source configuration, and fixed reliability thresholds that were chosen as engineering constants rather than learned from user studies.

---

## Chapter 5 — Conclusion and Recommendation

### 5.1 Conclusion

This study implemented and documented **WMSU Rice Disease Detection**, a working integration of the Rice Leaf Disease Dataset [1], a **YOLO11n-cls** classifier, a **FastAPI** inference service with preprocessing and reliability logic, and a **React** client with scan, encyclopedia, and history features. Offline validation through **`evaluate_accuracy.py`** yielded **86.55%** top-1 and **99.74%** top-5 accuracy on the larger `dataset/` validation split, and **81.40%** top-1 with **99.84%** top-5 on `dataset_original_split/`, demonstrating strong ranking behavior with top-1 accuracy that depends on the exact split in use. The live API’s quality gate and JSON fields address part of the Chapter 1 problem statement by making uncertainty and input issues visible instead of hiding them behind a single percentage.

### 5.2 Recommendations

For **future research**, publishing per-class precision and recall, confusion matrices, and calibration curves would strengthen formal reporting; conducting surveys or interviews with agriculture students about the clarity of **`message`** text would improve human factors evidence. For **future engineering**, adding HTTPS, authentication or rate limits as needed, returning a **`model_version`** or checksum in JSON, tightening CORS, and optionally persisting history in a small database would move the same codebase toward production hardening. For **data practice**, any new imagery merged from local field campaigns should be documented alongside [1] so readers can tell which results come from the public benchmark and which from supplementary collections.

---

## References

[1] Mendeley Data, "Rice Leaf Disease Dataset," Available: https://data.mendeley.com/datasets/vwv3nry3wr/1  

[2] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016. Available: https://www.frontiersin.org/articles/10.3389/fpls.2016.01419  

[3] P. K. Sethy *et al.*, "Detection and classification of rice leaf diseases using trained deep CNNs," *IEEE Access*, vol. 8, pp. 107359–107371, 2020. Available: https://ieeexplore.ieee.org/document/9095358  

[4] J. G. A. Barbedo, "Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification," *Computers and Electronics in Agriculture*, vol. 153, pp. 46–54, 2018. Available: https://www.sciencedirect.com/science/article/pii/S0168169917310591  

[5] S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, 2010. Available: https://ieeexplore.ieee.org/document/5288526  

[6] Ultralytics, "YOLO11 Documentation," Available: https://docs.ultralytics.com/models/yolo11/  

[7] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017. Available: https://proceedings.mlr.press/v70/guo17a.html  

[8] FastAPI, "FastAPI documentation," Available: https://fastapi.tiangolo.com/  

[9] Meta Open Source, "React — A JavaScript library for building user interfaces," Available: https://react.dev/  

[10] Vite, "Vite documentation," Available: https://vite.dev/  

---

*Implementation source: [https://github.com/alfahadadian04-blip/rafsan.git](https://github.com/alfahadadian04-blip/rafsan.git). Validation figures in Chapter 4 were produced by `evaluate_accuracy.py` on the author’s development machine in May 2026.*
