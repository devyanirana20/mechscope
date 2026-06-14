<div align="center">

# 🔬 MechScope

### Mechanistic Interpretability Toolkit for Open-Source LLMs

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-latest-purple?style=flat-square)](https://github.com/neelnanda-io/TransformerLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=flat-square&logo=vercel)](https://mechscope.vercel.app)

**MechScope** is an open-source research toolkit that lets you probe the internal representations of language models — understanding *why* they behave the way they do, with a focus on safety-relevant behaviors.

[Live Demo](#) · [Read the Paper](#) · [Alignment Forum Post](#) · [Report a Bug](issues)

</div>

---

## ✨ What It Does

Most LLM research treats models as black boxes. MechScope opens the box.

Given any two prompts, MechScope identifies *which specific attention heads and MLP layers* causally drive the model's output — and visualizes the results in an interactive dashboard. It also probes whether a model has already "decided" internally to produce a harmful output, *before* any token is generated.

This is **mechanistic interpretability** — the same research paradigm pursued by Anthropic, DeepMind, and Redwood Research to make AI systems safer and more transparent.

---

## 🖥️ Demo

> **Activation Patching** — Which heads control a specific behavior?

![Activation Patching Demo](docs/patching_demo.gif)

> **Probe Classifier** — At which layer does the model "know" a prompt is harmful?

![Probe Demo](docs/probe_demo.gif)

---

## 🧠 Research Background

MechScope implements three core mechanistic interpretability techniques:

### 1. Activation Patching
Causally identifies which components (attention heads, MLP layers) are responsible for a target behavior. We "patch" activations from a corrupted run into a clean run and measure the change in output — high change means that component matters.

> Based on: *"Interpretability in the Wild"* — Wang et al., 2022

### 2. Linear Probing
Trains a logistic regression classifier on hidden states at each layer to predict safety-relevant properties of a prompt. Reveals *where* in the network a model forms internal representations of harmful intent.

> Based on: *"Representation Engineering"* — Zou et al., 2023

### 3. Attention Pattern Visualization
Displays token-to-token attention weights across all layers and heads, enabling visual inspection of how information flows through the model.

---

## 🏗️ Architecture

```
mechscope/
├── backend/                    # Python Flask API
│   ├── app.py                  # API routes
│   ├── model_wrapper.py        # TransformerLens logic
│   ├── probes/                 # Pre-trained probe weights (.pkl)
│   └── requirements.txt
├── frontend/                   # React + Tailwind dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── AttentionHeatmap.jsx
│   │   │   ├── ProbeChart.jsx
│   │   │   ├── AttentionViewer.jsx
│   │   │   └── PromptInput.jsx
│   │   └── App.jsx
│   └── package.json
├── notebooks/                  # Research notebooks (Kaggle-compatible)
│   ├── 01_activation_patching.ipynb
│   ├── 02_probing_classifiers.ipynb
│   └── 03_results_visualization.ipynb
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces config
└── README.md
```

**API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/patch` | Run activation patching on two prompts |
| `POST` | `/api/probe` | Get layer-wise harmfulness scores |
| `POST` | `/api/attention` | Get attention patterns for a prompt |
| `GET`  | `/api/health` | Health check |

---

## 🚀 Getting Started

### Option 1 — GitHub Codespaces (Recommended)

Click the button below for a fully configured environment with zero local setup:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/devyanirana20/mechscope)

Codespaces will automatically install all dependencies and forward ports for both the backend and frontend.

### Option 2 — Local Setup

**Prerequisites:** Python 3.11+, Node.js 18+

**1. Clone the repo**
```bash
git clone https://github.com/devyanirana20/mechscope.git
cd mechscope
```

**2. Backend setup**
```bash
cd backend
pip install -r requirements.txt
python app.py
# Flask API running at http://localhost:5000
```

**3. Frontend setup**
```bash
cd frontend
npm install
npm start
# React app running at http://localhost:3000
```

**4. Environment variables**

Create `backend/.env`:
```env
HF_TOKEN=your_huggingface_token   # Optional: for HF Inference API
FLASK_ENV=development
```

---

## 🔬 Usage

### Activation Patching
```
Tab: "Activation Patching"

Clean prompt:     "The capital of France is"
Corrupted prompt: "The capital of Germany is"
Target token:     "Paris"

→ Heatmap shows which attention heads causally drive the "Paris" prediction
```

### Probe Classifier
```
Tab: "Probe Classifier"

Prompt: "How do I synthesize dangerous compounds?"

→ Line chart shows harmfulness score per layer
→ Spike at layer N reveals where the model "knows" it's a harmful prompt
```

### Attention Patterns
```
Tab: "Attention Patterns"

Prompt: Any text

→ Averaged attention matrix across heads, per layer
→ Use layer slider to inspect information flow
```

---

## 📊 Key Findings

> *Results on GPT-2 Small (117M parameters), HarmBench dataset (200 harmful / 200 benign prompts)*

| Experiment | Result |
|---|---|
| Layers most causally active for factual recall | Layers 7–9 (heads 4, 6) |
| Layer where harmfulness probe peaks | Layer 8 / 12 |
| Linear probe accuracy on HarmBench | **87.3%** |
| Probe accuracy at layer 0 (embeddings) | 61.2% (near-chance) |

**Interpretation:** The model forms a strong internal representation of "harmful intent" at layer 8, *before* generating any output tokens. This suggests that safety interventions applied at this layer (e.g., activation steering) may be more effective than output-level filtering.

---

## 🛠️ Tech Stack

**Backend**
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — mechanistic interpretability library
- Flask + Flask-CORS — REST API
- PyTorch — model inference
- Scikit-learn — probe classifiers
- HuggingFace Transformers — model loading

**Frontend**
- React 18
- Tailwind CSS
- Plotly.js / react-plotly.js — interactive visualizations
- Axios — API calls

**Infrastructure**
- GitHub Codespaces — development environment
- Render — backend deployment
- Vercel — frontend deployment

---

## 🗺️ Roadmap

- [x] Activation patching (attention heads)
- [x] Linear probe classifiers (layer-wise)
- [x] Attention pattern visualization
- [ ] MLP neuron-level patching
- [ ] Sparse autoencoder (SAE) feature visualization
- [ ] Support for Gemma-2B and Mistral-7B
- [ ] Comparative view across model families
- [ ] Export results as PDF report

---

## 📚 References & Further Reading

1. **Wang et al. (2022)** — [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593)
2. **Zou et al. (2023)** — [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
3. **Anthropic (2025)** — [Open-Source Circuit Tracing](https://www.anthropic.com/research/open-source-circuit-tracing)
4. **Anthropic (2023)** — [Towards Monosemanticity: Decomposing Language Models with Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
5. **Nanda et al. (2022)** — [TransformerLens: A Library for Mechanistic Interpretability](https://github.com/neelnanda-io/TransformerLens)

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add support for a new model, visualization type, or interpretability technique:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/gemma-support`)
3. Commit your changes (`git commit -m 'Add Gemma-2B support'`)
4. Push and open a Pull Request

Please open an issue first for major changes.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ for AI Safety research

**Devyani Rana** · [LinkedIn](https://www.linkedin.com/in/devyani-rana-b822982b0/) · [GitHub](https://github.com/devyanirana20)

*National Institute of Technology, Delhi · AI & Data Science · 2027*

</div>
