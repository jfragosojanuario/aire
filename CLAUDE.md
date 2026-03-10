# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIRE is a construction cost estimation tool. It accepts natural language descriptions of construction elements (in English or Portuguese), extracts structured data via LLM + regex/fuzzy matching, and matches them against a pricing database using semantic similarity.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run app/main.py

# Run cost estimation directly
python scripts/cost_estimation.py

# Test the NLP pipeline interactively
python scripts/text_input.py
```

The `.env` file must contain `OPENAI_API_KEY` for the LLM extraction pipeline to work.

## Architecture

The pipeline has four stages:

1. **NLP Extraction** (`scripts/text_input.py`) — The core engine (~900 lines). Entry point is `generate_final_output()`. It uses OpenAI GPT (`gpt-4o-mini`) for structured JSON extraction, with regex + fuzzy matching as fallback. Extracts: element type, materials, dimensions (standardized to SI), quantities, construction category, and confidence scores.

2. **Configuration/Keywords** (`scripts/config/`) — Keyword mappings for materials, elements, units, and construction categories. Both English and Portuguese terms are mapped. Extend these when adding new material or element types.

3. **Semantic Matching** (`scripts/cost_estimation.py`) — Uses `sentence-transformers` to embed the extracted JSON and compute cosine similarity against the Excel database (`database/Data AIRE project_V0.xlsx`). Returns top-5 matches with unit prices.

4. **Web UI** (`app/main.py`) — Streamlit interface. The embedding model and database embeddings are cached with `@st.cache_resource` for performance.

## Key Data Flow

```
User text → llm_process_json() → structured JSON → keyword extraction → cosine similarity → top-5 products + prices
```

The Excel database columns that matter: `Domain`, `Subdomain`, `product_material`, `related_terms_pt`, `related_terms_en`, `technical_specs`, `unit_prices`, `measurement_units`.

## Test Data

`tests/generated_inputs.py` has 31 pre-generated JSON examples (various element/material/unit combinations). `tests/test_inputs.py` has one active test string. These are used for offline testing of `cost_estimation.py` without hitting the LLM API.
