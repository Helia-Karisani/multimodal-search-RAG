

# Step 3 – Large Multimodal Models (LMMs)

### Connection to Steps 1 & 2

Up to this point, the project has built the foundation of multimodal intelligence:

* **In Step 1**, we trained a contrastive-learning model that *learned how to embed images* into a semantic vector space.
* **In Step 2**, we built a multimodal search system that used those embeddings to connect **text, images, and videos** inside a unified retrieval database.

Now in **Step 3**, we take the next leap: moving from **retrieval** to **reasoning**.
Here, we use a **Large Multimodal Model (LMM)** — Google’s *Gemini 1.5 Flash* — to interpret, explain, and analyze visual information in natural language.
Instead of matching vectors, the LMM actually *understands* what’s in the images and can respond conversationally to open-ended questions.

---

### Purpose

This step demonstrates how to use an **LMM** to process images and answer questions about their content.
You’ll explore how the model “sees” visual data, how it interprets charts and diagrams, and even how it detects hidden patterns within images.
It provides an intuitive look into **visual instruction tuning** — the process that enables LMMs to connect visual perception with language understanding.

---

### Workflow Overview

#### 1 Environment Setup

The notebook begins by loading API credentials through environment variables.
The `google-generativeai` library is installed and configured with the user’s `GOOGLE_API_KEY`.
This key authenticates the notebook to use Google’s multimodal model via REST API:

```python
!pip install google-generativeai
%env GOOGLE_API_KEY=************
```

The library is then initialized using `genai.configure()` so that calls to `GenerativeModel("gemini-1.5-flash")` can be made directly.

---

#### 2 Helper Functions

A few utility functions are created to keep the interface readable:

* `to_markdown(text)` – converts AI responses into nicely formatted Markdown blocks.
* `call_LMM(image_path, prompt)` – the main function that:

  1. Loads an image with Pillow (`PIL.Image.open`),
  2. Sends both the **prompt** and **image** to Gemini,
  3. Returns a Markdown-formatted text response.

This function wraps the full visual reasoning pipeline, transforming raw pixels into descriptive natural language output.

---

#### 3 Analyzing Images with the LMM

The first experiment tests whether the model can describe a **stock market chart** (`SP-500-Index-Historical-Chart.jpg`).
Using:

```python
call_LMM("SP-500-Index-Historical-Chart.jpg", "Explain what you see in this image.")
```

The LMM interprets the chart — recognizing axes, trends, and patterns — and explains the historical behavior of the S&P 500 index.
This shows that LMMs can reason over **structured visual data**, not just natural photos.

---

#### 4 Interpreting Technical Figures

Next, the model is tested on a conceptual image (`clip.png`) illustrating relationships between text and image embeddings.
The prompt asks:

> “Explain what this figure is and where it is used.”

The LMM identifies it as a **CLIP architecture diagram** and explains its role in connecting visual and textual understanding — a direct conceptual bridge back to what we built in Steps 1 and 2.

---

#### 5 Decoding Hidden Messages in Images

The notebook then challenges the model with a subtle task — interpreting an image (`blankimage3.png`) containing a *hidden text message* blended into background color.
By prompting:

```python
call_LMM("blankimage3.png", "Read what you see on this image.")
```

the LMM successfully decodes and reads the faint hidden text, illustrating how well-tuned visual-language models perceive fine-grained pixel-level patterns.

---

#### 6 Understanding “How the Model Sees”

To demystify what’s happening behind the scenes, the notebook loads the same hidden-text image using NumPy and Matplotlib.
By thresholding color channels (`image_array[:,:,0]>120`), the code visualizes the red-channel intensity that the LMM might rely on.
This reveals that models *perceive visual data numerically* — as brightness matrices — not the way humans do.
It’s an educational visualization that contrasts *machine vision* with *human perception*.

---

#### 7 Creating Hidden Messages Yourself

Finally, a function `create_image_with_text()` lets users generate their own hidden-message images.
It overlays colored text (e.g., “Hello, world!”) on a matching background and saves it as `extra_output_image.png`.
The user can then call the same LMM function again:

```python
call_LMM("extra_output_image.png", "Read what you see on this image.")
```

and verify that Gemini can still read the concealed message.
The image is then reprocessed with NumPy to visualize the binary mask — demonstrating how AI models separate signal from background noise.

---

### Outcome and Connection to Next Steps

By the end of Step 3, we’ve evolved from simply **embedding** and **retrieving** multimodal content to **reasoning** about it.
The LMM can analyze complex visuals, interpret abstract diagrams, and extract hidden details — showing the jump from *searching* to *understanding*.

This directly sets the stage for **Step 4**, where these visual reasoning abilities are combined with **Retrieval-Augmented Generation (RAG)** — allowing an AI system to not only perceive multimodal inputs but also *generate contextually grounded answers* from retrieved evidence.


