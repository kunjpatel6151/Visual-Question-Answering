# Visual Question Answering (VQA)

## Project Overview

This project implements a Visual Question Answering (VQA) system using deep learning techniques. Given an image and a natural language question about the image, the model predicts the correct answer from a predefined vocabulary. This implementation leverages pretrained models for feature extraction and a custom attention mechanism for multimodal fusion.

## Dataset

This project uses the processed DAQUAR dataset. You can download it from Kaggle:
- [Processed DAQUAR Dataset](https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset)

## Architecture

### 1. Image Encoder: ResNet-50 (Spatial Features)
- **Model:** ResNet-50 (Pretrained on ImageNet)
- **Feature Extraction:** We extract spatial features from the final convolutional block (before global average pooling).
- **Dimensions:** Typically outputs a $2048 \times 7 \times 7$ tensor per image.
- **Purpose:** Instead of compressing the entire image into a single vector, this approach preserves spatial information, allowing the model to attend to specific regions (e.g., top-left vs. bottom-right) relevant to the question.

### 2. Question Encoder: BERT
- **Model:** BERT (Bidirectional Encoder Representations from Transformers)
- **Feature Extraction:** questions are tokenized and passed through a pretrained `bert-base-uncased` model.
- **Dimensions:** Uses the pooled output (CLS token) of size 768.
- **Purpose:** Provides a rich semantic representation of the question context.

### 3. Multimodal Fusion Strategy: Attention Mechanism

The core of this VQA system is the **Attention-based Fusion** strategy.

#### How it works:
1.  **Feature Projection:**
    -   Image features (`2048 -> hidden_dim`) and Question features (`768 -> hidden_dim`) are projected into a common embedding space using linear layers.
2.  **Joint Representation:**
    -   The projected features are combined (broadcasted addition) and passed through a non-linear activation (`tanh`) to create a joint representation of the question and each image region.
3.  **Attention Scoring:**
    -   A linear layer computes a scalar score for each of the $49$ ($7 \times 7$) spatial regions.
    -   A Softmax function normalizes these scores to produce an attention map (probabilities summing to 1).
4.  **Context Vector:**
    -   The original image features are weighted by these attention scores and summed. This results in a focused visual context vector that emphasizes the most relevant parts of the image.
5.  **Final Classification:**
    -   The focused visual context is concatenated with the question embedding.
    -   A classifier (MLP) predicts the final answer index.

#### Why this strategy was chosen:
-   **Focus:** Unlike simple concatenation or element-wise multiplication (which treat the entire image equally), attention allows the model to "focus" on specific objects or regions mentioned in the question (e.g., "What color is the *car*?").
-   **Interpretability:** Attention weights can be visualized to debug and understand what the model is looking at when making a decision.
-   **Performance:** Spatially-aware models generally outperform global feature models on VQA tasks because they retain local details necessary for answering fine-grained questions.

## Project Structure

-   `VQA_Image_Processing.ipynb`: Extracts and saves spatial features using ResNet-50.
-   `VQA_Text_Processing.ipynb`: Tokenizes questions and extracts BERT embeddings; builds answer vocabulary.
-   `VQA_Fusion.ipynb`: Loads precomputed features, defines the `AttentionVQA` model, and runs the training/evaluation loop.

## Setup & Usage

### Prerequisites
-   Python 3.x
-   PyTorch
-   Transformers (Hugging Face)
-   Pandas, NumPy, Pillow, Tqdm

### Workflow
1.  **Image Processing:** Run `VQA_Image_Processing.ipynb` to process your image dataset and save the `.pt` feature files.
2.  **Text Processing:** Run `VQA_Text_Processing.ipynb` to process questions/answers and save `.npy` feature files.
3.  **Training:** Open `VQA_Fusion.ipynb` to train the model. Ensure the paths to the saved features are correct.
