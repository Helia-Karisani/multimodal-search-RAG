
---

# 🏭 Step 5 – Industry Applications of Multimodal AI

### 🔗 Connection to Previous Steps

So far, we’ve built the full multimodal foundation:

* **Step 1** created contrastive-learning embeddings for aligning visual and textual data.
* **Step 2** enabled multimodal retrieval using those embeddings.
* **Step 3** introduced reasoning with a Large Multimodal Model (LMM).
* **Step 4** connected retrieval and reasoning into a complete multimodal RAG pipeline.

Now, in **Step 5**, we take the system into the **real world** — showing how multimodal AI can extract, structure, and reason over data found in business documents such as invoices, tables, and flow charts.

---

### 🎯 Purpose

Demonstrate **practical industry use-cases** for multimodal AI: automatically parsing images that contain structured information, extracting text and data fields, and even generating executable logic from visual diagrams.

---

### ⚙️ Workflow Overview

#### 1️⃣ Setup – Environment and Model Initialization

The notebook loads the environment variables and API keys using `dotenv`, then configures Google’s Gemini API:

```python
import google.generativeai as genai
from google.api_core.client_options import ClientOptions
genai.configure(
    api_key=GOOGLE_API_KEY,
    transport="rest",
    client_options=ClientOptions(api_endpoint=os.getenv("GOOGLE_API_BASE")),
)
```

The model `gemini-1.5-flash` is used instead of the older `gemini-pro-vision`.
Two helper functions are defined:

* `to_markdown(text)` – formats output into Markdown blocks.
* `call_LMM(image_path, prompt, plain_text=False)` – opens an image, sends it with a prompt to Gemini, and returns either plain-text or Markdown output.

---

#### 2️⃣ Analyzing an Invoice (Image: `invoice.png`)

Purpose: Show how the LMM can extract **structured financial data** from a visual document.

```python
call_LMM("invoice.png",
  """Identify items on the invoice. 
  Output JSON with quantity, description, unit price, and amount.""")
```

Gemini parses the invoice layout and returns data in JSON form — quantities, descriptions, prices, and totals.
A follow-up query tests numeric reasoning:

```python
call_LMM("invoice.png",
  """How much would four sets pedal arms cost and 6 hours of labour?""",
  plain_text=True)
```

The model performs a contextual calculation, demonstrating multimodal math reasoning applied to document understanding.

---

#### 3️⃣ Extracting Tables from Images (Image: `prosus_table.png`)

Purpose: Test structured data extraction and semantic analysis.

```python
call_LMM("prosus_table.png",
  "Print the contents of the image as a markdown table.")
```

The LMM reconstructs the image into a properly formatted Markdown table.
A second prompt adds reasoning:

```python
call_LMM("prosus_table.png",
  """Analyse the contents as a markdown table. 
  Which business unit has the highest revenue growth?""")
```

Gemini identifies numeric trends and responds with a textual summary, showing how LLMs can both read and interpret structured visual data.

---

#### 4️⃣ Analyzing Flow Charts (Image: `swimlane-diagram-01.png`)

Purpose: Demonstrate visual logic understanding and code generation.

```python
call_LMM("swimlane-diagram-01.png",
  """Provide a summarized breakdown of the flow chart 
  as a numbered list.""")
```

The model summarizes the business process step by step.
A second prompt goes further:

```python
call_LMM("swimlane-diagram-01.png",
  """Analyse the flow chart and generate Python code that implements its logic.""")
```

Gemini outputs Python code (e.g., an `order_fulfillment()` function) capturing the process flow of clients, online shops, and courier companies:

```python
def order_fulfillment(client, online_shop, courier_company):
    order = client.place_order()
    payment = client.make_payment(order)
    if payment.status == "successful":
        online_shop.ship_order(order)
        courier_company.transport_order(order)
    else:
        online_shop.cancel_order(order)
        client.refund_order(order)
    online_shop.invoice_order(order)
```

This shows how an LMM can translate diagrammatic logic into readable, functional code.

---

### 🧩 Outcome and Connection to Next Steps

By the end of Step 5, we see multimodal AI applied to **real business scenarios** — from invoice data extraction and financial reasoning to table analysis and code generation from flow charts.
This step proves that the LMM is not just a reasoning engine but a powerful automation tool that bridges human readable visual documents with machine actionable outputs.

In **Step 6**, these concepts extend to **multimodal recommendation systems**, where retrieval and reasoning merge to personalize results across images and text.

---
