# Evaluation Dataset for Conversational Dataset Retrieval (CDR)

This repository contains code for **benchmarking Large Language Model (LLM)–based retrieval systems** using datasets from the **Open Data Portal of the European Union (europa.eu)**.

The goal of this project is to evaluate how well different retrieval approaches can identify relevant datasets in a **conversational setting**.

## Project Overview

Modern data portals contain thousands of datasets, making dataset discovery difficult for users.  
This project benchmarks retrieval performance by simulating conversational queries and measuring how effectively LLM-based and embedding-based methods retrieve the correct datasets.

The pipeline includes:

- Collecting open datasets from **europa.eu**
- Generating conversational queries
- Constructing ground-truth relevance sets
- Benchmarking multiple retrieval approaches
- Evaluating retrieval performance using standard metrics

## What This Code Does

- Uses **EU Open Data Portal datasets** as the evaluation corpus  
- Generates **queries representing conversational dataset search**
- Applies different retrieval techniques:
  - Lexical retrieval
  - Dense / embedding-based retrieval
  - LLM-assisted retrieval(Open and Closed LLM models)
  - RAG
- Compares retrieval performance across methods
- Computes evaluation metrics such as precision and recall



