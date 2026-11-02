# Advanced RAG: From Naive to Production-Ready

Companion notebooks for the **RAG Master Class**, a one-hour course on making retrieval-augmented generation (RAG) actually work.

▶️ **Watch the full course:** [Your RAG Is Naive. Fix It in 1 Hour (Full Course)](https://www.youtube.com/watch?v=hEG3-L4l5R4)

Naive RAG (split → embed → retrieve top-k → generate) breaks in predictable ways: chunks cut through ideas, tables and mixed content get lost, and small chunks retrieve well but answer badly. Each notebook here tackles one of those problems with runnable code and, where it applies, a [RAGAS](https://docs.ragas.io/) evaluation so you can compare approaches with numbers, not vibes.

## What's inside

| # | Topic | Notebook | Dataset |
|---|---|---|---|
| 1 | Fixed-size chunking | [`fixed_size_chunking.ipynb`](01-chunking/fixed_size_chunking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zahere-dev/advanced-rag/blob/main/01-chunking/fixed_size_chunking.ipynb) | [Medical case studies (synthetic)](01-chunking/data/medical_case_studies_synthetica_dataset.txt) |
| 1 | Recursive chunking | [`recursive_chunking.ipynb`](01-chunking/recursive_chunking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zahere-dev/advanced-rag/blob/main/01-chunking/recursive_chunking.ipynb) | [Medical case studies (synthetic)](01-chunking/data/medical_case_studies_synthetica_dataset.txt) |
| 2 | Multi-representation indexing | [`multi_representation_indexing.ipynb`](02-multi-representation-indexing/multi_representation_indexing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zahere-dev/advanced-rag/blob/main/02-multi-representation-indexing/multi_representation_indexing.ipynb) | [Immunotherapy in NSCLC (PDF)](02-multi-representation-indexing/data/Immunotherapy_in_Non-Small-Cell_Lung_Cancer.pdf) |

```
advanced-rag/
├── 01-chunking/
│   ├── fixed_size_chunking.ipynb
│   ├── recursive_chunking.ipynb
│   └── data/medical_case_studies_synthetica_dataset.txt
└── 02-multi-representation-indexing/
    ├── multi_representation_indexing.ipynb
    └── data/Immunotherapy_in_Non-Small-Cell_Lung_Cancer.pdf
```

## 1. Chunking

How you split text decides what the retriever can find, so it affects answer quality, latency, cost and hallucinations. Both notebooks run the same pipeline on the same synthetic medical case-study corpus: chunk (1000 chars, 200 overlap) → embed with `text-embedding-3-small` → Chroma → `gpt-4o-mini`. Each one then generates a 10-question test set and scores it with RAGAS.

- **Fixed-size chunking** cuts text into equal-length pieces. It's simple and fast, but it ignores meaning and can split a sentence in half.
- **Recursive chunking** tries paragraph breaks first, then line breaks, so chunks follow the document's natural boundaries.

Results from the notebook runs:

| Strategy | Context precision | Faithfulness | Answer relevancy | Context recall |
|---|---|---|---|---|
| Fixed-size | 1.0000 | 0.7385 | 0.9183 | 0.9667 |
| Recursive | 1.0000 | **0.8276** | 0.9167 | 0.9056 |

In the course, semantic chunking scored just **0.38** faithfulness on this dataset. The "smartest" strategy isn't automatically the best one, so measure before you choose.

## 2. Multi-representation indexing

Real documents mix prose and tables, and raw table text embeds poorly. This notebook:

1. Partitions the PDF into text and table elements with [Unstructured](https://unstructured.io/) (`partition_pdf`, using Tesseract and Poppler).
2. Has an LLM write a concise **summary** of every text chunk and table.
3. Indexes the **summaries** in Chroma and stores the **original elements** in a docstore, linked by ID through LangChain's `MultiVectorRetriever`.
4. Searches against the summaries, then returns the full original content to the LLM.

You search a compact representation that's easy to match, and the model answers from the complete source.

## Getting started

Each notebook was written for Google Colab and runs top to bottom there.

1. Click **Open in Colab** on the notebook you want.
2. Add your OpenAI key as a Colab secret named `OPENAI_API_KEY` (🔑 icon in the left sidebar). The notebooks read it with `userdata.get("OPENAI_API_KEY")`.
3. Upload the matching file from the `data/` folder to `/content/` (Files panel). The notebooks load it from there.
4. Run all cells.

**Running locally instead?** Replace `userdata.get(...)` with `os.environ["OPENAI_API_KEY"]`, point the file paths at `data/`, and install the system packages for notebook 2 (`tesseract-ocr`, `poppler-utils`).

**Stack:** LangChain · OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) · Chroma · Unstructured · RAGAS

## About

Built by **Zahiruddin Tavargere**.
If these notebooks helped, ⭐ the repo and [watch the full course](https://www.youtube.com/watch?v=hEG3-L4l5R4).
