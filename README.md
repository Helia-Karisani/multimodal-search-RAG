# Multimodal Search and RAG Systems

This repository goes through six steps of building **multimodal AI systems**, from representation learning to multimodal **Retrieval-Augmented Generation (RAG)** and **recommender systems**. The models work across **text, images, audio, and video**.

---

## Project Purpose

Standard RAG systems add external text to a large language model's context. This project extends that to **multimedia data**, so the model can retrieve and reason over visual, audio, and text information together.

The six steps cover:

* **Contrastive learning** to embed multimodal data into shared vector spaces
* **Any-to-any multimodal search** across modalities
* **Visual instruction tuning**, where models reason jointly over text and images
* An **end-to-end multimodal RAG pipeline**
* **Industry applications** like document analysis, invoice understanding, and visual data extraction
* A **multi-vector recommender system** based on cross-modal similarity

---

## Repository Overview

```
Step-1 → Multimodal embeddings and contrastive learning
Step-2 → Multimodal search and retrieval
Step-3 → Large Multimodal Models (LMMs) and visual instruction tuning
Step-4 → End-to-end multimodal RAG system
Step-5 → Industry applications
Step-6 → Multi-vector multimodal recommender system
```

Each step has its own README, Jupyter notebooks, datasets, and helper scripts.
The `source/` folder has shared images, audio, and video files used across the steps.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Helia-Karisani/multimodal-search-RAG.git
cd multimodal-search-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install weaviate-client google-generativeai openai python-dotenv pandas numpy pillow matplotlib
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
EMBEDDING_API_KEY=your_palm_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Then open the notebook in the step folder you want.

---

## Tools & Frameworks

### AI & ML

* **TensorFlow / Keras**: contrastive learning and embedding visualization
* **PyTorch**: pretrained model integrations such as CLIP
* **Google Gemini API**: multimodal reasoning and image understanding
* **OpenAI Embeddings API**: text vectorization
* **Weaviate**: vector database for multimodal storage and retrieval
* **PaLM Multimodal API (`multi2vec-palm`)**: image and video embeddings

### Models

* **Gemini-1.5 Flash / Pro-Vision**: large multimodal models for vision-language reasoning

### Data & Processing

* **pandas**, **numpy**, **Pillow**, **base64**, **python-dotenv**

### Visualization

* **matplotlib**, **IPython.display**

### Environment

* Google Colab / Jupyter Notebook
* Python ≥ 3.10
