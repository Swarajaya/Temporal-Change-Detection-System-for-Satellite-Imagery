A research-grade implementation of a Self-Evolving Temporal Retrieval-Augmented Generation system designed to address outdated knowledge, hallucinations, and static retrieval behavior in Large Language Models (LLMs).

This project introduces time-aware retrieval and a feedback-driven self-evolution mechanism that allows the system to continuously adapt its knowledge base without retraining the underlying language model.

📌 Motivation

Large Language Models are trained on static corpora and therefore struggle with:

Time-sensitive queries

Outdated factual knowledge

Hallucinations caused by stale retrieval

While Retrieval-Augmented Generation (RAG) improves factual grounding, most RAG systems ignore temporal relevance and lack autonomous knowledge adaptation.

This project addresses these gaps by introducing:

Explicit temporal modeling during retrieval

A self-evolving feedback loop that refines retrieval behavior over time

🚀 Key Contributions

Temporal Retrieval
Incorporates document timestamps using decay-based ranking to prioritize recent knowledge.

Self-Evolving Knowledge Base
Updates retrieval behavior using confidence-driven feedback without retraining the LLM.

Multi-Mode Evaluation
Supports baseline RAG, temporal RAG, and self-evolving temporal RAG for fair comparison.

Research-Ready Outputs
Generates reproducible experiments, ablation studies, and analysis figures suitable for academic publication.

🏗️ System Architecture

The system consists of the following major components:

Data Ingestion & Preprocessing

Raw text ingestion from arXiv, Wikipedia, and web sources

Cleaning, chunking, and metadata extraction

Embedding & Indexing

Sentence-level embeddings using transformer encoders

FAISS vector indexing for efficient retrieval

Temporal Retrieval Module

Time-decay weighting integrated with semantic similarity

LLM-Based Generation

Context-aware response generation using retrieved evidence

Self-Evolving Agent

Confidence evaluation

Knowledge reinforcement and update

📁 Project Structure
self-evolving-temporal-rag/
│
├── data/
│   ├── raw/                 # Raw text documents
│   ├── embeddings/          # embeddings.npy, texts.json, metadata.json
│   └── index/               # FAISS index
│
├── scripts/
│   ├── build_embeddings.py  # Build embeddings + metadata
│   ├── build_index.py       # Create FAISS index
│   ├── run_pipeline.py      # Run baseline / temporal / evolving RAG
│   └── evaluate.py          # Generate figures (Fig 6–13)
│
├── experiments/
│   ├── baseline/
│   ├── temporal/
│   ├── self_evolving/
│   └── results/             # All plots and metrics
│
├── logs/                    # Retrieval & evolution logs
├── paper/                   # LaTeX sections for research paper
├── docs/                    # Diagrams and methodology
└── README.md

📊 Experimental Results

The system generates the following research figures:

Figure	Description
Fig. 6	Baseline vs Temporal vs Self-Evolving Retrieval
Fig. 7	Hallucination Reduction
Fig. 8	Confidence Score Distribution
Fig. 9	Ranking Change due to Temporal Logic
Fig. 10	Accuracy vs Latency Trade-off
Fig. 11	Ablation Study
Fig. 12	Knowledge Base Growth
Fig. 13	Failure Case Analysis

All figures are saved under:

experiments/results/

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt


On Windows, FAISS must be installed as:

pip install faiss-cpu

▶️ Running the Pipeline
Step 1: Build Embeddings
python scripts/build_embeddings.py

Step 2: Build FAISS Index
python scripts/build_index.py

Step 3: Run Retrieval Pipeline
python scripts/run_pipeline.py --mode baseline
python scripts/run_pipeline.py --mode temporal
python scripts/run_pipeline.py --mode self_evolving

Step 4: Generate Evaluation Figures
python scripts/evaluate.py

🧪 Evaluation Metrics

Retrieval Accuracy

Hallucination Rate

Confidence Score

Temporal Freshness

Latency vs Accuracy

Knowledge Base Growth

⚠️ Limitations

Manual dataset curation

Dependency on timestamp metadata

Offline evaluation (no real-time ingestion)

Text-only knowledge sources

These limitations are discussed in detail in the accompanying research paper.

🔮 Future Extensions

Real-time web ingestion

Temporal embedding learning

Multimodal RAG (text + images + tables)

Reinforcement learning-based evolution

Human-in-the-loop validation

📄 Research Paper

This repository accompanies the research paper:

“A Self-Evolving Temporal Retrieval-Augmented Generation System for Time-Sensitive Knowledge Access”

All LaTeX sources are available in the paper/ directory.

👤 Author

Swarajaya Singh Sawant
Department of Computer Science
Dehradun, India