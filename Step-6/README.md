

---

# Step 6 – Multimodal Recommender System

### Connection to Previous Steps

Across Steps 1 – 5, we moved from building contrastive multimodal embeddings to constructing retrieval systems, reasoning with LMMs, generating multimodal answers, and applying them to industry documents.
Now, **Step 6** brings everything together — using those capabilities to power a **multivector recommender system** that finds movies similar in both description and visual style.

---

### Purpose

Build a **dual-embedding recommendation engine** that aligns text (titles & summaries) and images (posters) in shared vector spaces, enabling *any-to-any search* — for instance, recommending films with posters that “look like” a given image or whose synopsis matches a concept.

---

### Workflow Overview

#### 1 Setup – Environment and API Connections

The notebook starts by installing and importing the needed libraries:
`weaviate-client`, `google-generativeai`, and `openai`.
It loads environment variables via `dotenv` and connects to an **embedded Weaviate instance** configured with two modules:

* `multi2vec-palm` for image embeddings (Google PaLM multimodal API)
* `text2vec-openai` for text embeddings (OpenAI API)

---

#### 2 Creating the Movie Collection

A collection called **Movies** is created with the following fields:
`title`, `overview`, `vote_average`, `release_year`, `tmdb_id`, `poster`, and `poster_path`.
Two vector spaces are then defined:

* `txt_vector` for semantic text representation of titles + overviews.
* `poster_vector` for visual representation of poster images.
  Each movie is represented in both spaces so that either text or image can serve as the search anchor.

---

#### 3 Loading and Encoding Data

Movie metadata is read from `movies_data.json`, and each poster is loaded from the `posters/` folder.
A helper function `toBase64()` encodes poster images for vectorization:

```python
def toBase64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')
```

Each movie is batched into Weaviate using `batch.add_object()` and sent to both vectorizers for text and image embedding.

---

#### 4 Running Text-to-Text Recommendations

Queries like:

```python
movies.query.near_text(
    query="Movie about lovable cute pets",
    target_vector="txt_vector", limit=3)
```

retrieve semantically related titles (e.g., “101 Dalmatians”, “Beethoven”, “Chicken Run”).
Changing the query to “Epic super hero” returns action films like “Iron Man” and “Man of Steel” — demonstrating text-semantic recommendation over movie overviews.

---

#### 5 Running Text-to-Image and Image-to-Image Recommendations

By switching the target vector to `poster_vector`, the system retrieves visually similar posters:

```python
movies.query.near_text(
    query="Epic super hero",
    target_vector="poster_vector", limit=3)
```

or even directly searches with an image example:

```python
movies.query.near_image(
    near_image=toBase64("test/spooky.jpg"),
    target_vector="poster_vector", limit=3)
```

This shows the power of multivector spaces — the same database supports recommendation by visual similarity and semantic meaning alike.

---

#### 6 Results and Analysis

Even with authentication errors in some image calls, the text queries work smoothly, illustrating how recommendation can be driven by shared multimodal embeddings.
This architecture is extendable to music, art, or product recommendations — any domain where meaning exists beyond text alone.

---

### Outcome and Final Project Summary

Step 6 concludes the journey by transforming our multimodal pipeline into a **practical recommender system** that bridges language and vision.
From learning embeddings (Step 1) to retrieval (Step 2), reasoning (Step 3), RAG integration (Step 4), industry use (Step 5), and recommendation (Step 6) — the project evolves from *representation learning to real-world application.*
This final step demonstrates how multimodal AI can understand, search, reason, and recommend — closing the loop on intelligent context-aware systems.

---
