
# 🔎 Step 2 – Building a Multimodal Search Engine

### 🔗 Connection to Step 1

In **Step 1**, we learned how contrastive learning builds a *shared embedding space* where similar images lie close together.
**Step 2** takes that principle further — using those embedding concepts to create a **real multimodal search system** capable of finding related items across **images, videos, and text**.
Instead of only comparing images to images, we now integrate multiple data types into a single searchable database using **Weaviate**, an open-source vector database built for multimodal AI.

---

### 🎯 Purpose

This step demonstrates how to:

* Embed **images and videos** into a shared vector space using a pretrained **multimodal embedding model**,
* Store those embeddings inside a vector database,
* Query the system using **text, images, or videos** interchangeably, and
* Visualize how these different modalities coexist in one semantic space.

In short — this step turns the concept of “multimodal similarity” into a working **search engine**.

---

### ⚙️ Workflow Overview

#### 1️⃣ Setting Up the Environment

After loading required libraries and API keys, Weaviate is connected in **embedded mode** (running locally in memory).
The connection is configured with two key modules:

* `multi2vec-palm` – to generate vector embeddings for multimodal data (image + video),
* `backup-filesystem` – to store and restore saved vectors when needed.
  This setup forms the searchable vector store for all multimodal assets. 

---

#### 2️⃣ Creating the Collection (“Animals”)

A new collection named **`Animals`** is created to store embeddings and metadata for media items.
If an old copy exists, it is deleted and recreated fresh.
The collection is configured with the **Palm Multimodal Embedding model (`multimodalembedding@001`)** that maps images and videos into **1408-dimensional embeddings**.
These embeddings will later allow similarity search across modalities. 

---

#### 3️⃣ Preparing Helper Functions

A small helper, `toBase64(path)`, reads a file (image or video) and encodes it in base64 format — the format Weaviate expects for uploading binary data.
This makes it possible to send and store images or short videos directly from local files. 

---

#### 4️⃣ Inserting Images into the Database

From the **`source/animal_image`** folder, the notebook loops through each file and uploads it to the Weaviate “Animals” collection.
Each entry includes:

* `name` – the file name,
* `path` – file path for display,
* `image` – base64-encoded content,
* `mediaType` – a label marking it as “image”.

Batch processing (`rate_limit(100)`) ensures efficient uploads.
After completion, the script verifies there are no failed objects. 

---

#### 5️⃣ Inserting Videos

Next, the same process is repeated for videos stored in the **`source/video`** folder.
Each video is inserted one by one using `animals.data.insert()`, labeled with `mediaType = "video"`.
The notebook checks again for upload errors to confirm that the media assets (about 15 in total: 9 images + 6 videos) are successfully stored. 

---

#### 6️⃣ Verifying the Database

To confirm the content count and distribution, the code runs an **aggregation query** (`animals.aggregate.over_all(group_by="mediaType")`) which lists how many items of each type exist — verifying that images and videos are balanced and correctly stored. 

---

#### 7️⃣ Building Multimodal Search Helpers

Several functions make searching and displaying results easy:

* `json_print()` – nicely formats metadata in readable JSON,
* `display_media()` – renders either an image or video inline,
* `url_to_base64()` and `file_to_base64()` – convert media from URLs or local files into base64 strings for queries.
  These utilities abstract away the low-level encoding details so the user can focus on search results. 

---

#### 8️⃣ Running Text-to-Media Search

The first live query demonstrates **semantic retrieval using text input**.
By searching for a phrase such as:

```python
animals.query.near_text(
    query="dog playing with stick",
    return_properties=['name','path','mediaType'],
    limit=3
)
```

Weaviate compares the text embedding against stored media embeddings and returns the closest matching items — images and videos of dogs in action.
This shows that embeddings from different modalities can be meaningfully compared. 

---

#### 9️⃣ Searching by Image

Using an image file (`test/test-cat.jpg`) as the query, the notebook calls `near_image()` to find related images and videos within the same embedding space.
It returns a ranked list of similar results, rendered automatically by the helper functions.
A second version of this query retrieves an image from a **web URL**, converts it with `url_to_base64()`, and performs the same search — demonstrating how external sources can be used as input. 

---

#### 🔟 Searching by Video

Finally, the code performs **video-to-media search** using `near_media()` with `media_type=NearMediaType.VIDEO`.
Given a short clip (e.g., `test-meerkat.mp4`), Weaviate returns the most visually similar videos or related images, completing the “any-to-any” search capability. 

---

#### 11️⃣ Visualizing the Multimodal Embedding Space

To explore how the model clusters data internally, the notebook restores a saved backup (`resources-img-and-vid`) containing a larger collection of image and video embeddings.
It then extracts vectors and their `mediaType` labels into a DataFrame and applies **UMAP** for 2D projection.
The resulting visualization shows two main clusters (images vs videos) with overlaps where semantics align — a clear picture of how multimodal embeddings organize data. 

---

#### 12️⃣ Interactive Exploration and Cleanup

An interactive UMAP plot allows zooming and highlighting across clusters.
Finally, the connection to Weaviate is closed (`client.close()`), cleaning up the embedded database. 

---

### 🧠 Outcome and Connection to Next Steps

At the end of Step 2, the project has evolved from learning image representations to **performing real multimodal retrieval** — connecting text, image, and video in one unified vector space.
This serves as the backbone for Step 3, where the same concept expands into **Large Multimodal Models (LMMs)** that reason across modalities using language, not just similarity.

