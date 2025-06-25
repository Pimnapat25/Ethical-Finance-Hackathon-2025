# 📊 Financial Question Answering with Qwen-14B, Qwen-32B (RAG + Prompt Engineering)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline enhanced with structured **prompt engineering** to answer financial questions using the **Qwen-14B** (for Rise/Fall), **Qwen-32B** (for MCQ) large language model. It supports two major types of questions:
- **Multiple Choice (A–D)**
- **Rise/Fall Market Movement Prediction**

---

## 📁 Directory Overview

- `test.csv` and `multiple_choice.csv` and `rise_fall.csv` – Input question file
- `vectorstore/` – Prebuilt knowledge base for context retrieval
- `Finance_Hackathon_Pipeline.ipynb`, `vllm_qwen32.ipynb` – Exploratory and batch prompt generation logic

---

## 🧹 Step 1: Preprocessing

The system first identifies the **type of question** and **language** using regex-based heuristics.

### Question Type Detection

```python
def detect_question_type(query):
    # Returns: 'rise_fall', 'multiple_choice', or 'other'
```

### Language Detection

```python
def detect_language(query):
    # Returns: 'th', 'en', 'mix', or 'unknown'
```

Applied with:
```python
df["question_type"] = df["query"].apply(detect_question_type)
df["language"]      = df["query"].apply(detect_language)
```

---

## 🔍 Step 2: Retrieval-Augmented Generation (RAG)

For each question, we retrieve the **top-k** most relevant context documents using a vectorstore retriever (e.g., Chroma, FAISS).

```python
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])
```

This context is then used to construct Qwen compatible prompts.

---

## 🧠 Step 3: Prompt Engineering for Qwen

We use a chat-style format with `<thinking>` and `<output>` tags to enforce reasoning and strict answer formatting.

### 🅰️ Prompt Template for Multiple-Choice (A–D)

```text
<|im_start|>system
You are a financial assistant. Use the following context to carefully think through the question and select the best answer from A, B, C, or D.
<|im_end|>

<|im_start|>user
Context:
[retrieved context]

Question:
[MCQ question here]

<thinking>
<|im_end|>

<|im_start|>assistant
Step-by-step reasoning...
</thinking>

<output>
A
</output>
```

### 📈 Prompt Template for Rise/Fall Questions

```text
<|im_start|>system
You are a financial assistant. Given the context and the question, respond only with 'Rise' or 'Fall'.
<|im_end|>

<|im_start|>user
Context:
[retrieved context]

Question:
[Price movement question]

<thinking>
<|im_end|>

<|im_start|>assistant
Analyze trends and sentiment...
</thinking>

<output>
Rise
</output>
```

---

## 🧪 Step 4: Inference

You can run inference using `vLLM`, `transformers`, and LangChain. Sample wrapper:

```python
from vllm import LLM
output = llm.generate([prompt], sampling_params)
```

The response will be parsed by extracting content within the `<output>` tag using:

```python
import re
def extract_answer(output_text):
    match = re.search(r"<output>\s*(.*?)\s*</output>", output_text, flags=re.DOTALL)
    return match.group(1).strip() if match else "UNKNOWN"
```

---

## 📄 Output

A list of chain-of-thought prompts with clearly structured outputs, such as:

```text
<thinking>
The context mentions that marginal cost equals marginal revenue at the point of profit maximization.
</thinking>
<output>
B
</output>
```

---

## 📌 Notes

- Ensure your vectorstore contains finance-related documents (e.g., FinCoT, Finance-Instruct-500k).
- Prompts are deterministic and parsing-friendly. Do not alter tags unless necessary.
- Supports multilingual questions (Thai, English, or mixed).

---

## 📄 License

MIT License  
(c) 2025 Financial AI Research Group
