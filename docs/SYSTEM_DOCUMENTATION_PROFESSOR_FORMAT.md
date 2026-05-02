# Image-Based Rice Leaf Disease Classification Using Deep Learning and a Full-Stack Web System

**Institution (template):** Western Mindanao State University — College of Computing Studies — Department of Information Technology — Republic of the Philippines  

**System repository:** [alfahadadian04-blip/rafsan](https://github.com/alfahadadian04-blip/rafsan.git)  

**Document purpose:** This file follows the **section structure and depth** of the reference course document (*Machine-Learning.pdf*, local path `D:/Bagopoito/Machine-Learning.pdf`) while describing **this project’s** implementation: a browser-based **WMSU Rice Disease Detection** application using **YOLO11 classification**, **FastAPI**, and **React** (not the reference paper’s drought/pest/healthy three-class ResNet18 + Flask + React Native stack). Fill in **Submitted by:** and **Date:** on your title page when binding.

**Need ~40 pages in Word/PDF?** Use the generated long edition: [`SYSTEM_DOCUMENTATION_PROFESSOR_FORMAT_EXPANDED.md`](./SYSTEM_DOCUMENTATION_PROFESSOR_FORMAT_EXPANDED.md) (see `docs/build_long_professor_doc.py`).

---

# Chapter 1 — Introduction

## 1.1 Background of the Study

Rice (*Oryza sativa*) is a staple crop whose yield and quality are threatened by multiple **biotic stresses**, including fungal and bacterial diseases, insect-related damage, and viral complexes that often first appear as **visible leaf symptoms**. Early recognition of abnormal leaf appearance supports timely field scouting, rational use of inputs, and coordination with agricultural extension services.

Traditional diagnosis relies on **manual inspection** and expert judgment. While authoritative, it is **time-consuming**, **subjective** under variable lighting, and **hard to scale** when experts are few relative to the number of farmers and students who need guidance. Similar symptoms can also suggest different underlying problems, which increases the risk of misclassification when decisions are based on a quick visual guess alone.

The widespread availability of **smartphone and digital-camera imagery** makes **image-based analysis** a practical complement to field visits. Color, texture, shape, and lesion patterns can be encoded numerically and mapped to **discrete condition labels** using **machine learning**. In particular, **deep convolutional models** can learn hierarchical features directly from pixels, reducing dependence on hand-crafted descriptors while still benefiting from careful **preprocessing** (orientation, lighting-aware transforms, and quality checks).

This project implements and documents such a system: a **six-class rice leaf condition classifier** exposed through a **REST API** and a **responsive web client**, with explicit handling of **prediction uncertainty** for non-ideal images (e.g., pasted or low-resolution photos).

## 1.2 Purpose of the Review / Study

The purpose of this documentation is twofold:

1. **Scholarly:** To situate the project within established work on **plant disease classification**, **transfer learning**, and **deployed ML services**, and to state clearly how the present artifact differs from hybrid classical pipelines (e.g., handcrafted texture features plus shallow classifiers).

2. **Engineering:** To record **design decisions, data flow, training and validation practice, deployment pattern, and evaluation metrics** so that the system can be **replicated, extended, or audited** by another researcher or developer.

The implementation emphasizes a **single deployable stack** (Python server serving both **inference** and the **built web UI**) to reduce operational complexity for academic demonstration and small-scale hosting.

## 1.3 Research Questions

The primary purpose of this study (software artifact) is to **develop and evaluate** a web-accessible system capable of classifying rice leaf images into **six predefined categories** with transparent confidence and reliability feedback.

1. How effectively does a **fine-tuned YOLO11 classification (nano)** model separate the six target classes on **held-out validation folders**?

2. How can **preprocessing and test-time multi-view inference** (EXIF correction, multiple transformed views, averaged probabilities) improve **robustness** compared to a single forward pass on the raw upload?

3. How should **reliability** be communicated when inputs are **out-of-distribution** (e.g., missing camera metadata, low resolution, ambiguous softmax distributions)?

4. What is the **end-to-end latency** of the inference path for a typical uploaded image on commodity CPU hardware, and is it acceptable for interactive classroom or extension demos?

5. To what extent can a **browser-based interface** with an embedded **disease encyclopedia** and **session scan history** support **learning and triage** without replacing certified agronomists?

## 1.4 Scope

The scope of this project includes:

- **Target classes (six):** Leaf Blight, Rice Blast, Rice Leaffolder, Rice Stripes, Rice Tungro, and Healthy Leaf — as reflected in the client encyclopedia and the trained classifier head.

- **Input modality:** Single **RGB leaf photograph** per prediction request (`multipart/form-data`, field name `image`).

- **Core software:**  
  - **Backend:** Python **FastAPI** + **Uvicorn**, **Ultralytics YOLO**, **Pillow**.  
  - **Frontend:** **React** + **TypeScript** + **Vite** + **Tailwind CSS** + **Framer Motion** + **Lucide** icons.  
  - **Weights:** `backend/yolo11n-cls.pt` loaded at server startup.

- **Evaluation tooling:** Root-level **`evaluate_accuracy.py`** for **top-1 / top-5** validation on YOLO-style `train/` and `val/` directory trees when those datasets exist locally.

- **Automation:** **`run-fullstack.bat`** to build `frontend/dist` and start the API with static hosting.

### Delimitations

- **Taxonomy:** The system does **not** detect every possible rice pest or nutrient disorder; it is limited to the **six trained labels**. Other conditions may be mislabeled or forced into the nearest class.

- **Image-level classification only:** There is **no lesion segmentation**, **no bounding-box localization**, and **no counting** of insects.

- **No satellite or hyperspectral** inputs; only standard RGB photographs.

- **No persistent multi-user database** in the repository: scan **history** is kept in **browser session state** and is cleared on full page reload unless extended by future work.

- **Training data provenance:** Large corpora may be **gitignored**; replication assumes access to **comparable labeled images** organized per class.

- **Regulatory:** Outputs are **advisory** and not a substitute for national plant protection regulations, pesticide labels, or expert field diagnosis.

---

# Chapter 2 — Review of Related Literature

Rice remains central to food security in many regions; yield loss from diseases and pests is a recurring theme in agronomic literature [12], [15]. **Image-based plant phenotyping** has gained traction because it can provide **repeatable, rapid** estimates of stress indicators compared with purely manual scoring [11], [17].

**Convolutional neural networks** learn spatial hierarchies from pixels and, when **pretrained** on large-scale natural image corpora, often **transfer** effectively to agricultural domains with smaller labeled sets [20], [29], [30]. Residual architectures [20], [26] addressed optimization difficulties in very deep networks; more recent **unified families** (including YOLO variants) also offer **classification heads** suitable for whole-image labeling when localization is not required, trading some architectural specialization for **training ecosystem integration and deployment speed** [19].

Classical pipelines combining **texture descriptors** (e.g., GLCM-based features) with **kernel machines** remain pedagogically important and can perform well under controlled imaging [4], [10], [14]; however, **end-to-end deep learning** reduces manual feature design at the cost of **data hunger** and **opacity**, motivating **post-hoc interpretability** tools (e.g., class activation concepts [27]) where thesis work requires them.

**Deployment** literature distinguishes **edge** inference from **centralized** serving; this project follows a **stateless REST** pattern (load model once, score per request) common in lightweight ML microservices [37], implemented here with **FastAPI** rather than Flask. **Security and ethics** for advisory agronomic tools emphasize **confidence disclosure**, **uncertainty handling**, and **non-substitution** for professional judgment [12].

**Synthesis for this project:** Prior work supports (a) **deep learned features** for leaf imagery, (b) **transfer learning** for data efficiency, (c) **explicit uncertainty or thresholding** for responsible user messaging, and (d) **consistent train/inference preprocessing**. The implemented system instantiates these principles using **YOLO11-cls**, **EXIF-aware decoding**, **multi-view test-time averaging**, and **rule-based reliability flags** on top of the softmax output.

**Theoretical / conceptual lens:** Information flows from **observed image** → **validated tensor representation** → **learned posterior over classes** → **policy layer** (thresholds on confidence, margin, entropy, resolution) → **presentation layer** (JSON + UI). This separates **epistemic** limitations (model fit) from **communicative** choices (how strongly to endorse the top class).

---

# Chapter III — Methodology

## 3.1 Research Design

This study follows an **applied developmental and experimental** design:

- **Developmental:** A full-stack software artifact (API + web UI) was engineered to completion against stated functional requirements.

- **Experimental (ML):** Model quality is measured by **quantitative validation metrics** (top-1 and top-5 accuracy on held-out `val/` splits) using the same Ultralytics toolchain employed in training.

The design compares **engineering alternatives** at a high level (e.g., single-process deployment vs. separate dev servers) rather than comparing YOLO to classical SVM on the same codebase; the latter can be proposed as **future work**.

## 3.2 System Development Methodology

To manage complexity, work was organized into a **planning phase** followed by **iterative deliverables** (conceptually aligned with agile sprints):

### Phase 0: Requirement Analysis and Backlog

- **Scope:** Six-way rice leaf classification, browser UI, reliability messaging, reproducible evaluation script.  
- **Deliverables:** Class list and UI navigation map; repository layout (`backend/`, `frontend/`); choice of **YOLO classification** for integration with Ultralytics training and export.

### Sprint 1: Data Pipeline and Baseline Model

- **Goal:** Curate or obtain labeled images; enforce **YOLO-style** folder structure (`train/<class>/`, `val/<class>/`).  
- **Deliverables:** Cleaning rules (duplicates, corrupt files, minimum usability); augmentation as configured in Ultralytics training; baseline training run logs under local `runs/` (not necessarily in Git).

### Sprint 2: Model Training, Fine-Tuning, and Selection

- **Goal:** Improve **out-of-sample** behavior (e.g., fine-tuning experiments, early stopping).  
- **Deliverables:** `best.pt` or equivalent promoted to **`backend/yolo11n-cls.pt`**; documented validation accuracy via **`evaluate_accuracy.py`**.

### Sprint 3: Backend API Integration

- **Goal:** Load weights **once** at startup; expose **stateless** inference.  
- **Deliverables:**  
  - **FastAPI** application (`backend/main.py`) with **`POST /predict`**.  
  - **`GET /health`** for monitoring.  
  - **JSON** responses including `label`, `confidence`, `all_scores`, `is_reliable`, `message`, and auxiliary flags (`has_camera_metadata`, `has_low_resolution`, `ensemble_views`).  
  - **Preprocessing:** EXIF transpose, brightness-variance quality gate, multi-view generation and **probability averaging**.

### Sprint 4: Frontend Development and System Testing

- **Goal:** Usable interface for scan, encyclopedia, and history.  
- **Deliverables:** React SPA (`frontend/src/App.tsx`, `HistoryPanel.tsx`); `npm run build` output; manual end-to-end tests (upload, paste, error paths); optional timing notes for inference on target hardware.

## 3.3 System Architecture Design

### 3.3.1 Overall Architecture

The system follows a **client–server** pattern:

| Layer | Technology | Responsibility |
|--------|------------|----------------|
| **Presentation** | React (Vite build) | Tabs (Home, Scan, Encyclopedia, History), image pick/paste, results, warnings, session history. |
| **Application / service** | FastAPI + Uvicorn | Routing, validation, static files for `frontend/dist`, CORS configuration. |
| **Intelligence** | Ultralytics YOLO11n-cls | Whole-image classification; softmax class probabilities. |
| **Persistence** | *(client session only)* | In-memory React state and object URLs for thumbnails; **no** server-side scan database in the described codebase. |

In **production-style** deployment, the browser and API share **one origin** (e.g. `http://localhost:8000`), so the default client uses a **relative** `/predict` URL.

### 3.3.2 ML Pipeline Architecture

Stages align with standard supervised learning practice:

1. **Acquisition:** Collect RGB images labeled into the six classes.  
2. **Partitioning:** Train/validation split via class subfolders.  
3. **Training / fine-tuning:** Ultralytics classification training with chosen epochs, augmentations, and early stopping (see local run `args.yaml` / `results.csv` when available).  
4. **Serialization:** Weights saved as `.pt` for Ultralytics `YOLO(path)`.  
5. **Integration:** Model singleton loaded in FastAPI **lifespan**.  
6. **Inference:** Decode upload → quality checks → multi-view predict → average probabilities → thresholded reliability → JSON.

## 3.4 Data Collection

- **Sources (typical):** Institutional field captures, public agricultural image corpora, or curated teaching datasets — subject to **licensing** and **ethical** use for academic purposes only.  
- **Format:** JPEG/PNG RGB images organized by class folders.  
- **Volume:** Depends on local corpus; validation scripts expect non-empty `val/` per dataset root.  
- **Ethical considerations:** No personal identifiable information in images; labels should reflect **consent** and **context** where human subjects appear in backgrounds (ideally cropped to leaves). Predictions are **advisory**; agronomist consultation is recommended for chemical or varietal decisions.

## 3.5 Data Processing

### Class labeling

Operational definitions align with the six UI encyclopedia entries (symptoms and recommended farmer actions are displayed for **education**, not as automated prescriptions binding the model).

### Data cleaning

Remove unreadable files, severe blur where labels are untrustworthy, and misfiled images. Ensure **three-channel** RGB input to the vision stack.

### Handling missing or invalid values

Non-image uploads, empty bodies, undecodable bytes, or failed quality gates return **HTTP 4xx** with explicit error messages rather than being imputed.

### Train, validation, and test splits

Primary reporting uses Ultralytics **`val`** on the provided roots. A separate **holdout test** set may be added for thesis rigor; document its path and lock it before final experiments.

### Data augmentation

Augmentation is applied during **training** per Ultralytics configuration (not hard-coded in `main.py`); inference uses deterministic server-side views for **test-time** diversity.

## 3.6 Feature Engineering

The deployed model uses **end-to-end deep learning**: high-level features are learned by convolutional blocks rather than hand-crafted GLCM or HOG vectors. **Engineered aspects** appear instead as **input transformations** (EXIF correction, optional contrast and mirror views) and **post-processing rules** on the probability vector.

## 3.7 Machine Learning Model Development

### 3.7.1 Model selection

**Selected model:** **YOLO11n-cls** (nano width) for balance of **parameter count**, **inference time**, and **integration** with the Ultralytics training stack.

**Alternatives (conceptual):** Larger CNNs or vision transformers may improve accuracy at higher compute cost; classical SVM/RF on handcrafted features may suit extremely small data but underperform on subtle spatial patterns for fine-grained leaf damage [4], [14].

### 3.7.2 Model training

Training hyperparameters (optimizer, epochs, augmentations, early stopping) are defined in the **Ultralytics experiment** that produced `yolo11n-cls.pt`. The API does **not** retrain on requests; it only performs **forward inference**.

### 3.7.3 Model evaluation metrics

| Metric | Role |
|--------|------|
| **Top-1 accuracy** | Primary scalar for correct class on validation folders. |
| **Top-5 accuracy** | Useful when class count grows; with six classes it is often near saturation. |
| **Reliability rate (engineering)** | Fraction of predictions passing internal thresholds (`is_reliable`); distinct from statistical recall. |
| **Confusion matrix** | Recommended thesis artifact from Ultralytics validation exports. |

**Offline script:** `evaluate_accuracy.py` prints dataset-wise top-1 and top-5 after `model.val(...)`.

### 3.7.4 Confusion matrix and per-class discussion *(thesis placeholder)*

After exporting a confusion matrix from your best run, discuss **dominant confusions** (e.g., visually similar classes) and tie findings to **augmentation** or **additional data** plans.

## 3.8 System Implementation

| Component | Stack |
|-----------|--------|
| Language | Python 3.x (project venv), TypeScript (frontend) |
| API | FastAPI, `python-multipart`, Uvicorn |
| ML | Ultralytics, PyTorch (transitive), Pillow |
| Frontend | React 18, Vite, Tailwind, Framer Motion, Lucide |
| Version control | Git; remote [GitHub rafsan](https://github.com/alfahadadian04-blip/rafsan.git) |

**Consistency rule:** Preprocessing at inference must **match** training expectations (RGB, spatial size handled inside Ultralytics predict).

## 3.9 Model Deployment

- **Artifact:** `backend/yolo11n-cls.pt`.  
- **Process:** `uvicorn main:app --host 0.0.0.0 --port 8000` (or `127.0.0.1` for local-only).  
- **Throughput:** Depends on CPU/GPU; YOLO11n-cls is sized for **interactive** use.  
- **Scaling:** Stateless API instances can be **horizontally scaled** behind a reverse proxy; sticky sessions not required for inference (history is client-side).

## 3.10 Testing Strategy

| Test type | Focus |
|-----------|--------|
| **Unit-level** | Image validation branches, reliability threshold logic. |
| **Integration** | `POST /predict` returns JSON; `GET /health` returns OK. |
| **System** | Build `dist`, serve via FastAPI, full browser scan flow. |
| **Model** | `evaluate_accuracy.py` on frozen `val/` splits. |

## 3.11 Performance Evaluation

Report **validation top-1/top-5** from `evaluate_accuracy.py` **after each model change**. Optionally log **p50/p95 latency** for `/predict` under target hardware. Update the table in **Chapter IV** accordingly.

## 3.12 Ethical and Security Considerations

- **Advisory use:** UI and API messages should not be read as legal or medical directives.  
- **Privacy:** Do not log raw images on shared servers without consent and policy.  
- **Bias:** Class imbalance in training data skews metrics; report **per-class** metrics where possible.  
- **Security:** For public deployment, use **HTTPS**, rate limiting, and dependency patching; restrict CORS if the API is not same-origin.

## 3.13 Tools and Technologies

| Category | Tools |
|----------|--------|
| IDE / editor | VS Code, Cursor, or equivalent |
| Python env | `venv` / `myenv`, `pip install -r backend/requirements.txt` |
| Node | `npm` for `frontend` |
| ML | Ultralytics YOLO, PyTorch backend |
| Docs | Markdown in `docs/` |

## 3.14 Proposed User Interfaces (Mapped to Implementation)

| Reference UI concept | Implementation |
|---------------------|----------------|
| Home / overview | Home tab with session summary and flow description. |
| Scan / upload | Scan tab: file input, **paste from clipboard**, preview, Analyze. |
| Stress / disease guide | Encyclopedia tab: six cards (symptoms + actions). |
| History | History tab: thumbnails, delete one, clear all; object URLs revoked on delete. |
| Low confidence warning | Server `message` + `is_reliable`; client distinguishes **error** vs **warning** styling where applicable. |

**Testing walkthrough (for thesis appendix):** (1) healthy leaf image, (2) each disease class sample if available, (3) dark/blurry image to observe quality gate or warning behavior, (4) pasted screenshot to observe metadata / reliability messaging.

---

# Chapter IV — Presentation, Analysis, and Interpretation of Data

## 4.1 Data presentation

**Table 1. Example offline validation (run `evaluate_accuracy.py` to refresh).**

| Dataset root (local) | Metric | Example value* |
|----------------------|--------|----------------|
| `dataset` | Top-1 | ≈ 86.6% |
| `dataset` | Top-5 | ≈ 99.7% |
| `dataset_original_split` | Top-1 | ≈ 81.4% |
| `dataset_original_split` | Top-5 | ≈ 99.8% |

\*Replace with your machine’s current console output after retraining.

**Figures (recommended):** Confusion matrix heatmap; sample UI screenshots (Scan, Encyclopedia, History); optional reliability vs. resolution scatter from a labeled audit set.

## 4.2 Analysis

The gap between dataset roots in Table 1 suggests **distribution shift** or different **split/augmentation** policies. **Top-5** near ceiling with six classes indicates the true label usually appears among the top few logits even when top-1 fails.

## 4.3 Interpretation

Results support the Chapter 1 goal: a **usable** classifier with **documented offline performance** and a **runtime reliability layer** for inputs that differ from training photography. Findings should be interpreted **only** within the delimited six-class task.

---

# Chapter V — Summary, Conclusions, and Recommendations

## 5.1 Summary of findings

A **full-stack** rice leaf disease classification system was implemented with **FastAPI**, **React**, and **YOLO11n-cls**, including **multi-view averaged inference** and **reliability messaging**. Offline validation commands are documented via **`evaluate_accuracy.py`**.

## 5.2 Conclusions

The artifact meets its scoped objectives for **teaching, demonstration, and triage-oriented decision support**, with explicit limits on **taxonomy coverage** and **professional substitution**.

## 5.3 Recommendations

- Add a **frozen test set** and **per-class precision/recall**.  
- Run **user acceptance** sessions with extension workers.  
- Consider **Grad-CAM** or similar for thesis interpretability [27].  
- For production: **HTTPS**, logging policy, model versioning in API responses.

---

# Abstract

Rice leaf diseases reduce productivity when diagnosis is delayed. This project develops an **image-based six-class classification system** using a **fine-tuned YOLO11 classification (nano)** model, served through a **FastAPI** backend with a **React** web client. Preprocessing incorporates **EXIF orientation correction**, **image quality screening**, and **multi-view probability averaging**; responses include **confidence**, **full class score vectors**, and **reliability flags** to mitigate over-trust on difficult inputs. Offline validation is supported by a **scripted Ultralytics validation** workflow. The resulting system is suitable as an **academic and extension demonstration** of responsible, browser-accessible agricultural AI, while remaining **delimited** from full agronomic certification.

---

# References *(starter set — extend to match your institution’s citation style)*

[4] Arivazhagan, S., Latha, P. D. S. V., & Jeya, R. (2022). Rice leaf disease detection and classification using SVM with GLCM features. *Journal of Agricultural Informatics*, 13(2), 45–58.  

[10] Kabir, M., & Akinci, T. C. (2024). Revealing GLCM metric variations across a plant disease dataset. *Electronics*, 13(4), 782.  

[11] Laraswati, A. A., et al. (2021). Image-based phenotyping and selection index for rice under drought stress. *Plant Breeding and Biotechnology*, 9(4), 272–286.  

[12] Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.  

[14] Sanyal, P., et al. (2023). Textural feature extraction for mineral deficiency detection in rice leaves using GLCM and neural networks. *Smart Agricultural Technology*, 4, 100185.  

[15] Sethy, P. K., et al. (2020). Detection and classification of rice leaf diseases using trained deep CNNs. *IEEE Access*, 8, 107359–107371.  

[17] Vishal, M. K., et al. (2020). Image-based phenotyping of diverse rice genotypes. *arXiv*.  

[19] Yi, W., et al. (2025). Rice disease detection based on multi-scale dynamic feature fusion. *Frontiers in Plant Science*, 16.  

[20] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.  

[27] Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*.  

[29] Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*, 22(10), 1345–1359.  

[30] Barbedo, J. G. A. (2018). Impact of dataset size and variety on deep learning and transfer learning for plant disease classification. *Computers and Electronics in Agriculture*, 153, 46–53.  

[37] Grinberg, M. (2018). *Flask Web Development* (2nd ed.). O’Reilly. *(general REST ML serving patterns; this project uses FastAPI.)*  

**Ultralytics YOLO documentation** — https://docs.ultralytics.com/  

**FastAPI documentation** — https://fastapi.tiangolo.com/  

---

*Format alignment: section numbering and methodology depth modeled on **Machine-Learning.pdf** (`file:///D:/Bagopoito/Machine-Learning.pdf`). Content describes the **rafsan** repository implementation.*
