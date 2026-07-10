# Medical Diagnosis Assistant — GenAI Capstone Project

An end-to-end GenAI system that reasons about *possible* medical conditions from
reported symptoms, using Retrieval-Augmented Generation (RAG) grounded in a
medical knowledge base, with a locally fine-tuned embedding model and a
locally-served LLM via Ollama.

## What this project demonstrates

| Requirement | Where |
|---|---|
| Model selection & fine-tuning | `src/finetune_embeddings.py` — fine-tunes LLM with Triplet Loss on domain-specific medical triplets |
| Embedding generation & data insights |  PCA cluster plot + similarity heatmap over the knowledge base |
| RAG integration | `src/build_vectorstore.py` (FAISS + LangChain) + `src/rag_pipeline.py` |
| Advanced prompt engineering | `src/prompt_templates.py` — system priming, few-shot, retrieval grounding, chain-of-thought, guardrails |
| LLMOps & Responsible AI | `src/llmops.py` — structured logging, confidence guardrail |

## Project structure

```
medical_diagnosis_assistant/
├── data/
│   ├── medical_knowledge_base.csv   # 21 conditions: symptoms, description, action, severity
│   └── patient_histories.csv        # 10 synthetic patient cases for demos
├── src/
│   ├── data_prep.py                 # loading + triplet construction
│   ├── finetune_embeddings.py       # domain fine-tuning of the embedding model
│   ├── build_vectorstore.py         # FAISS + LangChain vector store
│   ├── prompt_templates.py          # advanced prompt engineering
│   ├── rag_pipeline.py              # end-to-end RAG pipeline (Ollama)
│   └── llmops.py                    # logging, guardrails, evaluation
├── notebooks/
│   └── Medical_Diagnosis_Assistant_Capstone.ipynb   # full executed walkthrough
├── models/                          # fine-tuned embedding model + FAISS index (generated)
├── outputs/                         # HTML export, plots, screenshots, query logs (generated)
├── ppt/
│   ├── Medical_Diagnosis_Assistant_Capstone.pptx    # slide deck with speaker notes
│   └── slide_notes_summary.txt                      # slide notes as plain text
├── requirements.txt
└── README.md
```

## Setup


pip install -r requirements.txt

open `notebooks/Medical_Diagnosis_Assistant_Capstone.ipynb` and run all cells —
it executes the entire pipeline end-to-end with visualizations and a patient-case demo.

## Submission artifacts

1. **Notebook** — `notebooks/Medical_Diagnosis_Assistant_Capstone.ipynb` 
2. **PPT** — `ppt/Medical_Diagnosis_Assistant_Capstone.pptx` (17 slides, code/output screenshots)
3. **Slide notes summary** — `ppt/slide_notes_summary.txt`
