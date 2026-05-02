# System Documentation — Chapters 1 to 5

**Official repository:** [https://github.com/alfahadadian04-blip/rafsan.git](https://github.com/alfahadadian04-blip/rafsan.git)  

**System title:** WMSU Rice Disease Detection  

**Document type:** End-to-end system description (introduction through conclusions) aligned with common thesis chapter numbering.

---

## Chapter 1 — Introduction

### 1.1 Background of the study

Rice (*Oryza sativa* L.) supports food security for a large share of the world’s population. In the field, **leaf symptoms** often appear before yield loss becomes irreversible: lesions, stripes, chlorosis, folding, and deformation patterns associated with pathogens, viruses, insect damage, or abiotic stress. Correct **early orientation** toward possible causes helps farmers, students, and extension staff prioritize scouting, sampling, and (where appropriate) expert consultation.

**Traditional diagnosis** relies on trained observers. It remains the gold standard for legal and agronomic certification but is **difficult to scale** when the ratio of experts to farms is low, when visits are costly, or when learners need repeated practice with diverse imagery. **Digital photographs** are now ubiquitous; they can be analyzed with **machine learning** models that map pixels to **discrete diagnostic labels**, provided training data are representative and limitations are communicated honestly.

This project implements such a system as an **open, versioned software repository** ([rafsan on GitHub](https://github.com/alfahadadian04-blip/rafsan.git)): a **six-class rice leaf condition classifier** using a **YOLO11 classification (nano)** model, exposed through a **FastAPI** web API and consumed by a **React** single-page application. The same server process may serve the built frontend, enabling **single-origin** deployment suitable for laboratories and demonstration kiosks.

### 1.2 Statement of the problem

There is a recurring gap between **offline benchmark accuracy** on curated folders and **trustworthy field-facing software**. Uploads from phones vary in resolution, lighting, orientation (EXIF), and provenance (camera vs. screenshot). Models can be **overconfident** on out-of-distribution inputs. The problem addressed here is:

> How to deliver a **browser-accessible** rice leaf classification service that reports **not only a label and probabilities** but also **structured reliability cues**, while remaining maintainable as a small-team **full-stack** codebase on GitHub?

### 1.3 Objectives of the study

**General objective.** To design, implement, and document a web-based rice leaf disease classification system integrated with a deep learning model and clear user-facing guidance.

**Specific objectives.**

1. Train or fine-tune a **YOLO11n-cls** model for **six** outcome classes aligned with extension-style messaging.
2. Implement a **REST API** (`POST /predict`) with **EXIF-aware preprocessing**, **quality screening**, **multi-view inference**, and **averaged class probabilities**.
3. Implement **reliability logic** (confidence, margin, entropy, resolution) and human-readable **`message`** strings.
4. Build a **React** client with **Home**, **Scan**, **Encyclopedia**, and **History** flows, including clipboard image paste and session history.
5. Provide **offline validation** tooling (`evaluate_accuracy.py`) for **top-1 / top-5** reporting on local `val/` splits.
6. Publish sources and this documentation on **[GitHub](https://github.com/alfahadadian04-blip/rafsan.git)** for replication and review.

### 1.4 Significance of the study

| Audience | Benefit |
|----------|---------|
| **Students** | Hands-on link between plant pathology vocabulary and model outputs. |
| **Researchers** | Baseline pipeline for comparing architectures, calibration methods, or datasets. |
| **Extension / ICT for agriculture** | Demonstrable artifact for workshops; advisory framing reduces false authority. |
| **Developers** | Clear separation: `backend/main.py`, `frontend/src/`, weights path, evaluation script. |

### 1.5 Scope and delimitations

**In scope:** Six display classes (Leaf Blight, Rice Blast, Rice Leaffolder, Rice Stripes, Rice Tungro, Healthy Leaf); single-image HTTP inference; session-local history; documented constants in `backend/main.py`.

**Out of scope:** Fine-grained pest species ID beyond trained classes; **hyperspectral** or **satellite** inputs; **automated pesticide prescription**; **guaranteed regulatory compliance** in all jurisdictions; **persistent multi-user cloud accounts** in the baseline repository.

### 1.6 Assumptions

- Training labels in local datasets are sufficiently correct for supervised learning.
- Deployment hosts provide enough RAM to hold the YOLO model resident in memory.
- Users can read the current English UI strings unless localization is added later.

### 1.7 Definition of terms

| Term | Meaning in this system |
|------|-------------------------|
| **YOLO11n-cls** | Ultralytics YOLO **classification** variant, nano width; outputs class probabilities for the whole image. |
| **Top-1 accuracy** | Fraction of validation images where the highest-probability class equals the ground-truth folder label. |
| **Top-5 accuracy** | Fraction where the true class appears among the five largest probabilities. |
| **Multi-view inference** | Several deterministic image views (original, center-crop resize, mirror, mild contrast); probabilities **averaged** before argmax. |
| **`is_reliable`** | Boolean from thresholds on top probability, margin, entropy, and resolution (see Chapter 3). |
| **SPA** | Single-page application; built to `frontend/dist` and optionally served by FastAPI. |

---

## Chapter 2 — Review of related literature and conceptual basis

### 2.1 Plant disease imaging and agricultural deep learning

Image-based plant disease detection has matured into a standard research thread: labeled leaf imagery, convolutional feature learning, and held-out evaluation [12], [15]. Surveys emphasize **dataset diversity** and **domain shift** when moving from growth-chamber captures to farmer-uploaded photos [30].

### 2.2 Transfer learning and compact architectures

Transfer learning initializes deep networks from large-scale pretraining, then adapts final layers (or deeper blocks) to agricultural targets [29], [30]. Compact models (e.g., small CNN heads packaged in unified training frameworks) trade some accuracy for **latency** and **deployment simplicity**—relevant when inference runs on **CPU-only** classroom servers.

### 2.3 Uncertainty, calibration, and responsible UI

Human–computer interaction research warns against **over-trust** in opaque scores. Complementary signals—**entropy**, **margin**, **input quality**—can gate language in the interface. Agronomic tools should remain **advisory** relative to certified experts and national regulations.

### 2.4 Synthesis and gap addressed by this repository

Prior work motivates **deep learned features**, **transfer learning**, and **cautious presentation**. The **[rafsan](https://github.com/alfahadadian04-blip/rafsan.git)** repository contributes a **fully wired** example: FastAPI + React + Ultralytics YOLO11-cls, with **explicit reliability JSON** and a public **Git** history for assignments and capstones.

### 2.5 Conceptual framework

```text
[Leaf image] → [Decode + EXIF + quality gate] → [Multi-view YOLO predict]
      → [Average softmax] → [Reliability policy] → [JSON + UI encyclopedia/history]
```

---

## Chapter 3 — Methodology

### 3.1 Research design

**Applied software research** with **quantitative ML evaluation**: build the artifact, then measure **validation accuracy** using Ultralytics `val` via `evaluate_accuracy.py`. Qualitative UX evaluation (surveys, think-aloud) is recommended as future work.

### 3.2 Data and labeling

Images are organized in **YOLO classification layout**: `train/<class_name>/` and `val/<class_name>/`. Class names must match the model’s output head. Local dataset roots may include `dataset` and `dataset_original_split`; large corpora are typically **gitignored** on GitHub but retained on training machines.

### 3.3 Model training and selection

Training uses the **Ultralytics** ecosystem (hyperparameters recorded in run artifacts under `runs/` locally). The promoted checkpoint is copied or saved as **`backend/yolo11n-cls.pt`**, which the API loads at startup through `ModelSingleton`.

### 3.4 System architecture

| Layer | Artifact | Responsibility |
|--------|-----------|------------------|
| Presentation | `frontend/src/` (React) | Tabs, scan, encyclopedia, history, `fetch` to `/predict`. |
| Application | `backend/main.py` (FastAPI) | HTTP, validation, optional static hosting of `frontend/dist`. |
| Intelligence | `ultralytics.YOLO` | Class probabilities per view. |
| Session memory | Browser React state | History list; object URLs revoked on delete/clear. |

### 3.5 Inference algorithm (summary)

Implementation reference: `backend/main.py` in the [repository](https://github.com/alfahadadian04-blip/rafsan.git).

1. Validate `Content-Type` is `image/*`; reject empty body.  
2. Open bytes with Pillow; **`ImageOps.exif_transpose`** for orientation.  
3. Read EXIF make/model flags → `has_camera_metadata`.  
4. Convert to RGB; set `has_low_resolution` if width or height `< 224`.  
5. **Brightness spread** on grayscale: standard deviation must be **≥ 18.0**; else HTTP 400.  
6. Build **four views** (`build_inference_views`): original, square center crop resized to original size, horizontal mirror, contrast enhance 1.08×.  
7. `model.predict` each view; sum valid `probs` vectors; divide by count → **averaged** distribution.  
8. Compute **top-1**, **second class**, **margin**, **Shannon entropy** (with small epsilon).  
9. `is_reliable` iff top ≥ **0.68**, margin ≥ **0.12**, entropy ≤ **1.35**, and not low-resolution.  
10. Compose **`message`** from optional warnings (metadata, resolution, weak confidence).  
11. Return JSON keys: `label`, `confidence`, `all_scores`, `is_reliable`, `message`, `has_camera_metadata`, `has_low_resolution`, `ensemble_views`.

### 3.6 Frontend methodology

`frontend/src/App.tsx` posts **`FormData`** with field name **`image`**. Default URL is **`/predict`**; override with **`VITE_API_URL`** for split dev servers. **AbortController** enforces a **15-second** timeout.

### 3.7 Evaluation methodology

`evaluate_accuracy.py` loads default weights **`backend/yolo11n-cls.pt`**, runs `model.val(data=root, split="val")` for each configured dataset root, and prints **Top-1** and **Top-5** percentages. Re-run after any retrain to refresh thesis tables.

### 3.8 Ethical considerations

Plant images only; no PII in baseline design. Outputs are **not** a license to apply restricted chemicals without local rules. Clear **encyclopedia** text encourages best practices and expert follow-up.

---

## Chapter 4 — Presentation, analysis, and interpretation of results

### 4.1 Quantitative validation (illustrative)

Run **`python evaluate_accuracy.py`** from the repository root (with venv and dependencies installed). Example results from documented runs on comparable setups:

| Dataset root | Top-1 (example) | Top-5 (example) |
|----------------|-----------------|-----------------|
| `dataset` | ≈ 86.6% | ≈ 99.7% |
| `dataset_original_split` | ≈ 81.4% | ≈ 99.8% |

**Analysis.** The gap between roots suggests **distribution or split differences** (augmentation policy, relabeling, or class proportions). **Top-5** near ceiling with **six** classes indicates the true label usually ranks highly even when top-1 fails.

**Interpretation.** Metrics support the Chapter 1 objective of a **usable** classifier on held-out imagery, while Chapter 3’s **reliability layer** addresses inputs that metrics alone do not describe (screenshots, dark photos).

### 4.2 System behavior and qualitative observations

- **Quality gate** rejects extremely flat/dark images to avoid meaningless softmax outputs.  
- **Multi-view averaging** tends to **stabilize** predictions on minor pose and contrast variation.  
- **Metadata warnings** inform users that pasted images may behave differently from camera captures without blocking prediction outright where the pipeline still returns a result.

### 4.3 Limitations observed in engineering practice

- **Session-only history** resets on reload—document for users expecting cloud sync.  
- **CORS `*`** simplifies demos but should be tightened for production.  
- **CPU vs GPU** latency varies; document hardware used in any formal timing study.

---

## Chapter 5 — Summary, conclusions, and recommendations

### 5.1 Summary of findings

1. A **six-class** rice leaf classification stack was implemented and open-sourced at **[github.com/alfahadadian04-blip/rafsan](https://github.com/alfahadadian04-blip/rafsan.git)**.  
2. The backend combines **FastAPI**, **Pillow preprocessing**, **multi-view YOLO11n-cls inference**, and **structured reliability fields**.  
3. The frontend delivers **scan**, **encyclopedia**, and **history** experiences aligned with extension-style communication.  
4. **Offline validation** is reproducible via **`evaluate_accuracy.py`**.

### 5.2 Conclusions

The system **meets its scoped engineering goals**: reproducible code, documented API, and transparent scoring. It **does not** remove the need for **field experts** or compliance with **local plant protection** rules. Validation accuracy is **dataset-dependent**; all numeric claims should be **re-measured** on the reader’s frozen splits.

### 5.3 Recommendations

**Research**

- Publish a **frozen test set** and per-class precision/recall tables.  
- Add **confusion matrices** and calibration curves (ECE) to the thesis appendix.  
- Conduct **user studies** with extension audiences.

**Engineering**

- Add **HTTPS** reverse proxy, **rate limiting**, and **structured logging** for public deployment.  
- Version the **model file** in API responses (`model_version` field) for audit trails.  
- Optional **SQLite** or cloud DB for persistent history.

**Repository hygiene**

- Keep **[GitHub](https://github.com/alfahadadian04-blip/rafsan.git)** README synchronized with run commands and documentation links.  
- Pin dependency versions for long-term reproducibility.

---

## References (starter set — expand per your institution’s style)

\[12] Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.  

\[15] Sethy, P. K., et al. (2020). Detection and classification of rice leaf diseases using trained deep CNNs. *IEEE Access*, 8, 107359–107371.  

\[29] Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345–1359.  

\[30] Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. *Computers and Electronics in Agriculture*, 153, 46–53.  

**Ultralytics YOLO documentation:** https://docs.ultralytics.com/  

**FastAPI documentation:** https://fastapi.tiangolo.com/  

---

*End of Chapters 1–5. Canonical source: [https://github.com/alfahadadian04-blip/rafsan.git](https://github.com/alfahadadian04-blip/rafsan.git).*
