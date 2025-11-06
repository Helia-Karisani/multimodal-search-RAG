

# 🔍 Step 4 – Multimodal Retrieval-Augmented Generation (RAG)

### 🔗 Connection to Steps 1–3

In **Step 1**, we built contrastive-learning embeddings that map multimodal inputs (like images and text) into shared vector spaces.
In **Step 2**, we used those embeddings to perform multimodal retrieval — finding relevant content across data types.
In **Step 3**, we advanced to reasoning with a Large Multimodal Model (LMM) that could interpret and describe images.

Now in **Step 4**, we **combine retrieval and reasoning** — creating a complete **multimodal RAG pipeline**.
Here, we first *retrieve* the most relevant image from a Weaviate vector database and then *generate* an explanation of that image using **Gemini 1.5 Flash**, producing a seamless “retrieve-then-reason” workflow.

---

### 🎯 Purpose

To build an **end-to-end Multimodal RAG system** that searches a pre-vectorized image + video dataset, retrieves contextually matching visual data, and uses an LMM to describe it in natural language.
This demonstrates how retrieval engines and generative vision models can work together.

---

### ⚙️ Workflow Overview

#### 1️⃣ Setup – Connecting Weaviate and Loading API Keys

The notebook starts by loading API keys from `.env` files using `dotenv`.
It configures both `EMBEDDING_API_KEY` (for Weaviate vector retrieval) and `GOOGLE_API_KEY` (for Gemini).
Then it launches a **local Weaviate embedded instance**:

```python
client = weaviate.connect_to_embedded(
    version="1.24.21",
    environment_variables={
        "ENABLE_MODULES": "backup-filesystem,multi2vec-palm",
        "BACKUP_FILESYSTEM_PATH": "/home/jovyan/work/L4/backups",
    },
    headers={"X-PALM-Api-Key": EMBEDDING_API_KEY}
)
```

This initializes a vector database capable of restoring and searching multimodal data.

---

#### 2️⃣ Restoring Pre-Vectorized Resources

To speed up experiments, a **13 K+ image and video dataset** is restored from a backup named `resources-img-and-vid`:

```python
client.backup.restore(
    backup_id="resources-img-and-vid",
    include_collections="Resources",
    backend="filesystem"
)
```

After a brief wait, Weaviate rebuilds the collection, which is verified by counting media types:

```python
image count ≈ 13 394  
video count ≈ 200
```

This step ensures that all embeddings are ready for retrieval without re-vectorization.

---

#### 3️⃣ Retrieving an Image from the Database

A custom function `retrieve_image(query)` is defined to search for visual matches to a text query within the `Resources` collection:

```python
response = resources.query.near_text(
    query=query,
    filters=Filter.by_property("mediaType").equal("image"),
    return_properties=["path"],
    limit=1
)
```

The function returns the path of the closest matching image. For example:

```python
img_path = retrieve_image("fishing with my buddies")
```

displays an image retrieved from the vector database based on semantic similarity.

---

#### 4️⃣ Describing the Retrieved Image with Gemini

Next, the Gemini API is configured for visual generation using `google-generativeai`:

```python
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
```

A helper function `call_LMM(image_path, prompt)` opens the image, sends it with a prompt to Gemini, and returns a Markdown-formatted description.
Example prompt:

```python
call_LMM(img_path, "Please describe this image in detail.")
```

The model analyzes the retrieved image and generates a rich natural-language summary of its contents .

---

#### 5️⃣ Combining Retrieval and Generation into a Single Pipeline

A final function `mm_rag(query)` connects everything:

```python
def mm_rag(query):
    SOURCE_IMAGE = retrieve_image(query)
    display(Image(SOURCE_IMAGE))
    description = call_LMM(SOURCE_IMAGE, "Please describe this image in detail.")
    return description
```

This one-liner performs both the retrieval and description steps in sequence — effectively a Multimodal RAG pipeline.
Example:

```python
mm_rag("paragliding through the mountains")
```

retrieves a mountain image and lets Gemini describe it automatically.

---

#### 6️⃣ Testing and Experimentation

Users are encouraged to try different queries (e.g., "diving under the sea" or "running in the rain").
Each run retrieves a semantically relevant image and produces a unique LMM description — showing how RAG extends from text to vision domains.
Finally, the Weaviate instance is closed to free resources:

```python
client.close()
```

---

### 🧩 Outcome and Connection to Next Steps

By the end of Step 4, we have a fully functional **Multimodal Retrieval-Augmented Generation system**.
It retrieves images from a vector database based on textual queries and uses a Large Multimodal Model to interpret them in detail.
This step marks the transition from visual reasoning to context-aware generation — laying the foundation for industry applications in Step 5, where AI will analyze real-world visual documents such as invoices and charts.

---

