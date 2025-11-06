

# 🧩 Step 1 – Overview of Multimodality

### 🎯 Purpose

This first step lays the conceptual and technical foundation for the entire multimodal project.
Its aim is to **show how a neural network can learn to represent visual data (images) as numerical embeddings** that reflect similarity — a building block for multimodal retrieval and RAG systems.
To make this concrete, the step trains a small **contrastive-learning model** on the **MNIST** handwritten-digit dataset and visualizes how it organizes similar digits close together in vector space.

---

### 🧠 Workflow Overview

#### 1️⃣ Setting Up the Environment

The notebook begins by importing deep-learning libraries (`torch`, `torchvision`, `torch.utils.data`) and data-analysis tools (`numpy`, `pandas`, `matplotlib`, `plotly`, `umap-learn`, `sklearn`).
These provide everything needed for loading data, building a model, training it, and visualizing embeddings.
A helper file **`mnist_dataset.py`** inside the same folder prepares *anchor*, *positive*, and *negative* image samples required for contrastive learning.

---

#### 2️⃣ Preparing and Understanding the Data

The MNIST dataset is loaded from the **data** folder (`digit-recognizer/train.csv`).
Images are normalized and converted to tensors, then split into training and validation sets.
Two PyTorch **DataLoaders** (`trainLoader`, `valLoader`) are created to feed the data in small batches — enabling efficient training and shuffling.
Before training, a visualization function displays triplets of images to confirm that anchors are correctly paired with similar and dissimilar samples.
This ensures the contrastive task is set up properly before moving to modeling.

---

#### 3️⃣ Designing the Embedding Network

Next, the notebook defines the **`Network`** class — a small convolutional neural network that converts each 28 × 28 image into a 64-dimensional feature vector.
This model acts as an *encoder*: it doesn’t classify digits; it learns to **map visually similar digits to nearby points in vector space**.
By keeping the embedding dimension low (64), later visualization and interpretation become intuitive.

---

#### 4️⃣ Defining the Learning Objective

To train the encoder, a custom **`ContrastiveLoss`** function is implemented.
It compares pairs of image embeddings using cosine similarity and computes an MSE-based penalty:

* similar images (anchor + positive) should yield a similarity close to 1,
* dissimilar images (anchor + negative) should yield a similarity near 0.
  This loss directly teaches the network *what it means for two images to be alike or different* — the key concept behind all multimodal alignment.

---

#### 5️⃣ Configuring Training

The model, optimizer (`Adam`, lr = 0.005), and scheduler (`StepLR`) are initialized.
Training uses GPU if available and saves progress inside the **`checkpoints`** folder for every epoch.
Each training loop processes anchor/contrastive pairs, computes loss, performs back-propagation, and updates the weights.
After every epoch, the average loss is recorded to monitor learning quality.

---

#### 6️⃣ Running the Training Loop

The core function **`train_model()`** orchestrates all of this.
Across roughly 10 epochs, it repeatedly:

1. Loads a batch from the DataLoader,
2. Generates embeddings for anchors and contrastives,
3. Calculates contrastive loss,
4. Updates model parameters,
5. Saves a checkpoint (`model_epoch_X.pt`).
   By the final epoch, the model learns to clearly separate different digits in embedding space.

---

#### 7️⃣ Inspecting the Results

Once trained, the model’s embeddings are extracted for the entire dataset.
Two dimensionality-reduction techniques are applied sequentially:

* **PCA** reduces 64 → 3 dimensions for 3D visualization;
* **UMAP** further compresses to 2D while preserving neighborhood structure.
  Interactive plots (via Plotly) show that images of the same digit cluster tightly together — a clear sign that contrastive learning has succeeded.

---

#### 8️⃣ Visualizing the Learning Process

An animation (`contrastive_Training_100.mp4`) illustrates how the network gradually organizes points in the embedding space over 100 epochs.
Clusters corresponding to different digits emerge and separate, making the abstract concept of “embedding learning” tangible.

---

### 🧩 Outcome and Connection to Next Steps

By the end of Step 1, you have a working example of **representation learning**: the model converts raw images into meaningful numerical embeddings that capture semantic similarity.
This principle is the backbone of later stages — where the same idea extends from single-modality (images) to **multimodal** settings, enabling **text–image–audio alignment**, **cross-modal retrieval**, and ultimately **RAG with multimodal context**.

