# 🎌 AniSage (Anime RAG)

AniSage is an AI-powered anime recommendation engine and chatbot. It uses Retrieval-Augmented Generation (RAG) to provide highly personalized anime suggestions based on natural language queries.

The system combines semantic search capabilities (via FAISS and sentence-transformers) with the reasoning capabilities of Large Language Models (LLMs) via Groq or OpenAI backends.

## ✨ Features

- 🧠 **Smart Semantic Search**: Find anime using natural language queries like *"dark psychological thriller"* or *"lighthearted romance in a school setting"* rather than rigid category tags.
- 💬 **Conversational AI Assistant**: Chat with **AniSage** naturally to receive contextual, highly personalized anime recommendations along with explanations of *why* they fit your taste.
- ⚡ **Lightning Fast Retrieval**: Built on a locally optimized **FAISS** vector database to retrieve matches instantly from over 14,000 processed anime records.
- 🎯 **Rich & Clean Anime Dataset**: Uses a heavily processed, cross-referenced dataset sourced directly from both **MyAnimeList** and **AniList**, ensuring accurate scores, genres, and metadata.
- 🖥️ **Modern Web Interface**: A sleek, dark-mode focused React/Vite frontend featuring glassmorphism design, real-time response streaming, and beautiful hover animations.
- 🛠️ **Fully Hackable CLI**: A robust Python/Typer terminal application that makes regenerating embeddings, updating the dataset, or testing the RAG pipeline incredibly easy.
## 🏗️ Architecture & Phases

The project is structured into multiple distinct phases:

- **Phase 1: Data Collection & Processing**
  - Scrapes and aggregates thousands of anime entries from MyAnimeList (Jikan v4 API) and AniList (GraphQL).
  - Merges, cleans, and generates optimized "embedding text" suitable for high-quality semantic retrieval while discarding adult entries and low-quality summaries.
- **Phase 2: Vector Embeddings & Indexing**
  - Embeds anime metadata into dense vectors using local models (e.g., `all-MiniLM-L6-v2` via HuggingFace's `sentence-transformers`) or OpenAI's embeddings.
  - Builds highly optimized, locally-stored `FAISS` indexes (and optionally ChromaDB) for lightning-fast semantic search.
- **Phase 3: RAG Chain & Conversational Engine**
  - Combines FAISS retrieval with LLM completion to evaluate and recommend anime conversationally.
  - Interactive CLI allows you to chat with "AniSage" directly in the terminal.
- **Phase 4: FastAPI Backend**
  - Exposes the RAG engine and Semantic Search over a robust FastAPI server.
  - Provides endpoints for semantic search (`/search`), finding a random anime (`/anime/random`), looking up specific anime by MAL ID, and an interactive chat session model (`/chat`).
- **Phase 5: Web UI Frontend**
  - A beautiful, modern frontend built with React, Vite, TypeScript, and TailwindCSS.
  - Features a clean user interface to chat with AniSage and browse recommendations with rich visual cards.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- A Groq API key or OpenAI API key (for the LLM completions)

### Backend Setup

1. **Activate the Virtual Environment:**
   ```powershell
   # Activation from the root directory
   .\phase4\ragani\Scripts\Activate.ps1
   ```

2. **Set Up the Environment Variables:**
   - Create a `.env` file in the root directory (you can use `.env.example` as a reference if it exists).
   - Add your API Key: `GROQ_API_KEY=your_key_here` (or `OPENAI_API_KEY`).

3. **Install Dependencies (if not already installed):**
   ```powershell
   pip install -r requirements.txt
   ```
   *(Note: This project uses a CPU-only build of PyTorch for maximum stability on Windows).*

4. **Run the API Server:**
   ```powershell
   python main.py serve
   ```
   The backend will be available at `http://localhost:8000`. You can access the automatic interactive API documentation at `http://localhost:8000/docs`.

### Frontend Setup

1. **Navigate to the UI folder:**
   ```powershell
   cd phase5
   ```

2. **Install Node Dependencies:**
   ```powershell
   # If you encounter peer dependency conflicts with Vite 8, use legacy-peer-deps:
   npm install --legacy-peer-deps
   ```

3. **Start the Development Server:**
   ```powershell
   npm run dev
   ```
   The frontend will typically be accessible on `http://localhost:5173`.

## 🛠️ CLI Commands (main.py)

The `main.py` entrypoint acts as the orchestration CLI for the entire project.

**Phase 1 & 2 Data Commands:**
- `python main.py run-all` - Run the full Phase 1 pipeline (Collect Jikan → Collect AniList → Process).
- `python main.py stats` - View data quality statistics on your processed dataset.
- `python main.py embed` - Generate embeddings for the processed dataset.
- `python main.py build-index` - Build the FAISS index from the generated embeddings.

**Testing & Serving:**
- `python main.py search "dark fantasy with demons"` - Test the FAISS retrieval engine directly.
- `python main.py chat` - Start an interactive terminal-based chat session with AniSage.
- `python main.py serve` - Start the FastAPI web server.

## ⚠️ Notes for Windows Users

If you run into `[WinError 1114] A dynamic link library (DLL) initialization routine failed` related to PyTorch/FAISS when starting the `serve` or `chat` commands, it is usually a conflict with OpenMP or missing CUDA libraries on your system. 
This repository includes a custom DLL-loading fix that automatically injects the correct CPU `torch\lib` paths and sets `KMP_DUPLICATE_LIB_OK=TRUE` and `CUDA_VISIBLE_DEVICES=-1` to ensure stable initialization on Windows.
