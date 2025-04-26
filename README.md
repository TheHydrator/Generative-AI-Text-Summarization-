# Generative-AI-Text-Summarization

<p align="center">
  <a href="https://huggingface.co/models">
    <img src="https://img.shields.io/badge/Built%20with-🤗%20Transformers-blue" alt="Huggingface Transformers">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Powered%20by-PyTorch-red" alt="PyTorch">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
  </a>
</p>

# 🧠 Generative AI Text Summarization using PEGASUS

## 📚 Project Overview
This project builds an abstractive text summarization pipeline using the **PEGASUS (google/pegasus-cnn_dailymail)** transformer model. The model summarizes dialogues from the **SAMSum** dataset, demonstrating the capabilities of Generative AI in producing high-quality human-like summaries.

## 🛠️ Technologies Used
- **Model:** PEGASUS (fine-tuned on CNN/DailyMail) – ~2.28 GB
- **Dataset:** SAMSum Dataset (Test Split – 819 dialogues)
- **Libraries and Tools:**
  - Huggingface Transformers
  - Datasets
  - Evaluate (for ROUGE metric calculation)
  - PyTorch
  - NLTK
  - tqdm

## ⚙️ Project Workflow
1. **Dataset Loading:** Loaded the SAMSum dataset using Huggingface `datasets` library.
2. **Model Loading:** Loaded the pre-trained PEGASUS model and tokenizer from `transformers`.
3. **Summarization:**
   - Tokenized dialogues efficiently.
   - Used batch inference for optimized performance.
   - Generated summaries using beam search (`num_beams=8`) with a length penalty.
4. **Evaluation:**
   - Computed ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.
   - Compared generated summaries against human references.
5. **Optimization:**
   - Implemented progressive batching.
   - Leveraged GPU acceleration for a 2x faster inference pipeline.

## 📈 Results
- Achieved high ROUGE scores indicating strong abstractive summarization performance.
- Reduced inference time significantly through batching and device optimization.
- Generated fluent and coherent summaries from raw dialogue inputs.

## 🔥 Key Highlights
- Worked with a **state-of-the-art** Generative AI model (PEGASUS).
- Efficiently handled a large model size (~2.28 GB).
- Automated end-to-end summarization and evaluation pipeline.
- Followed industry best practices in batching, evaluation, and device optimization.

## 🧩 Future Work
- Fine-tune PEGASUS specifically on the SAMSum dataset for improved performance.
- Deploy the summarization model through an interactive web application (e.g., Streamlit or FastAPI).

## 📍 How to Run
```bash
# Install required libraries
!pip install transformers[sentencepiece] datasets evaluate sacrebleu rouge_score py7zr tqdm nltk

# Download punkt tokenizer
import nltk
nltk.download('punkt')

# Load the model and dataset, and then run summarization and evaluation as per the provided scripts.


📎 References
https://arxiv.org/abs/1912.08777
https://huggingface.co/google/pegasus-cnn_dailymail
