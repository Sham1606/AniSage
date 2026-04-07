@echo off
setlocal enabledelayedexpansion

echo.
echo  ==========================================
echo   Anime RAG Project Structure Generator
echo  ==========================================
echo.

:: Set root project folder name
set ROOT=anime_rag

echo  Creating project: %ROOT%
echo.

:: ── ROOT ──────────────────────────────────────────────────────────────────────
mkdir %ROOT%

:: Root files
type nul > %ROOT%\.env.example
type nul > %ROOT%\.gitignore
type nul > %ROOT%\README.md
type nul > %ROOT%\requirements.txt
type nul > %ROOT%\main.py

:: ── PHASE 1: Data Collection & Processing ─────────────────────────────────────
mkdir %ROOT%\phase1
type nul > %ROOT%\phase1\__init__.py

mkdir %ROOT%\phase1\schemas
type nul > %ROOT%\phase1\schemas\__init__.py
type nul > %ROOT%\phase1\schemas\anime_schema.py

mkdir %ROOT%\phase1\collectors
type nul > %ROOT%\phase1\collectors\__init__.py
type nul > %ROOT%\phase1\collectors\jikan_collector.py
type nul > %ROOT%\phase1\collectors\anilist_collector.py

mkdir %ROOT%\phase1\processors
type nul > %ROOT%\phase1\processors\__init__.py
type nul > %ROOT%\phase1\processors\data_processor.py

mkdir %ROOT%\phase1\utils
type nul > %ROOT%\phase1\utils\__init__.py
type nul > %ROOT%\phase1\utils\helpers.py

mkdir %ROOT%\phase1\tests
type nul > %ROOT%\phase1\tests\__init__.py
type nul > %ROOT%\phase1\tests\test_phase1.py

mkdir %ROOT%\phase1\data\raw
mkdir %ROOT%\phase1\data\processed
mkdir %ROOT%\phase1\data\checkpoints

:: Add .gitkeep so empty folders are tracked by git
type nul > %ROOT%\phase1\data\raw\.gitkeep
type nul > %ROOT%\phase1\data\processed\.gitkeep
type nul > %ROOT%\phase1\data\checkpoints\.gitkeep

:: ── PHASE 2: Vector Embeddings ────────────────────────────────────────────────
mkdir %ROOT%\phase2
type nul > %ROOT%\phase2\__init__.py

mkdir %ROOT%\phase2\embeddings
type nul > %ROOT%\phase2\embeddings\__init__.py
type nul > %ROOT%\phase2\embeddings\embed_pipeline.py
type nul > %ROOT%\phase2\embeddings\embedding_models.py

mkdir %ROOT%\phase2\vectordb
type nul > %ROOT%\phase2\vectordb\__init__.py
type nul > %ROOT%\phase2\vectordb\chromadb_store.py
type nul > %ROOT%\phase2\vectordb\faiss_store.py

mkdir %ROOT%\phase2\tests
type nul > %ROOT%\phase2\tests\__init__.py
type nul > %ROOT%\phase2\tests\test_phase2.py

mkdir %ROOT%\phase2\data\faiss_index
mkdir %ROOT%\phase2\data\chroma_db
type nul > %ROOT%\phase2\data\faiss_index\.gitkeep
type nul > %ROOT%\phase2\data\chroma_db\.gitkeep

:: ── PHASE 3: RAG Orchestration ────────────────────────────────────────────────
mkdir %ROOT%\phase3
type nul > %ROOT%\phase3\__init__.py

mkdir %ROOT%\phase3\chains
type nul > %ROOT%\phase3\chains\__init__.py
type nul > %ROOT%\phase3\chains\rag_chain.py
type nul > %ROOT%\phase3\chains\query_rewriter.py

mkdir %ROOT%\phase3\prompts
type nul > %ROOT%\phase3\prompts\__init__.py
type nul > %ROOT%\phase3\prompts\system_prompt.txt
type nul > %ROOT%\phase3\prompts\prompt_templates.py

mkdir %ROOT%\phase3\memory
type nul > %ROOT%\phase3\memory\__init__.py
type nul > %ROOT%\phase3\memory\conversation_memory.py

mkdir %ROOT%\phase3\tests
type nul > %ROOT%\phase3\tests\__init__.py
type nul > %ROOT%\phase3\tests\test_phase3.py

:: ── PHASE 4: API Backend ──────────────────────────────────────────────────────
mkdir %ROOT%\phase4
type nul > %ROOT%\phase4\__init__.py

mkdir %ROOT%\phase4\api
type nul > %ROOT%\phase4\api\__init__.py
type nul > %ROOT%\phase4\api\main.py

mkdir %ROOT%\phase4\api\routes
type nul > %ROOT%\phase4\api\routes\__init__.py
type nul > %ROOT%\phase4\api\routes\chat.py
type nul > %ROOT%\phase4\api\routes\anime.py

mkdir %ROOT%\phase4\api\schemas
type nul > %ROOT%\phase4\api\schemas\__init__.py
type nul > %ROOT%\phase4\api\schemas\request_response.py

mkdir %ROOT%\phase4\api\middleware
type nul > %ROOT%\phase4\api\middleware\__init__.py
type nul > %ROOT%\phase4\api\middleware\rate_limiter.py

mkdir %ROOT%\phase4\tests
type nul > %ROOT%\phase4\tests\__init__.py
type nul > %ROOT%\phase4\tests\test_phase4.py

:: ── PHASE 5: Frontend UI ──────────────────────────────────────────────────────
mkdir %ROOT%\phase5
type nul > %ROOT%\phase5\__init__.py

mkdir %ROOT%\phase5\streamlit_app
type nul > %ROOT%\phase5\streamlit_app\app.py

mkdir %ROOT%\phase5\react_app\src\components
type nul > %ROOT%\phase5\react_app\src\components\ChatWindow.jsx
type nul > %ROOT%\phase5\react_app\src\components\AnimeCard.jsx
type nul > %ROOT%\phase5\react_app\src\components\PreferencePanel.jsx
type nul > %ROOT%\phase5\react_app\src\App.jsx
type nul > %ROOT%\phase5\react_app\package.json

:: ── PHASE 6: Deployment ───────────────────────────────────────────────────────
mkdir %ROOT%\phase6

mkdir %ROOT%\phase6\docker
type nul > %ROOT%\phase6\docker\Dockerfile.api
type nul > %ROOT%\phase6\docker\Dockerfile.frontend

type nul > %ROOT%\phase6\docker-compose.yml

mkdir %ROOT%\phase6\k8s
type nul > %ROOT%\phase6\k8s\deployment.yaml
type nul > %ROOT%\phase6\k8s\service.yaml

mkdir %ROOT%\phase6\.github\workflows
type nul > %ROOT%\phase6\.github\workflows\ci-cd.yml

:: ── DOCS ──────────────────────────────────────────────────────────────────────
mkdir %ROOT%\docs\phase_guides
type nul > %ROOT%\docs\architecture.md
type nul > %ROOT%\docs\phase_guides\phase1_guide.md
type nul > %ROOT%\docs\phase_guides\phase2_guide.md

:: ── Write .gitignore content ──────────────────────────────────────────────────
(
echo # Environment
echo .env
echo .env.*
echo !.env.example
echo.
echo # Python
echo __pycache__/
echo *.py[cod]
echo *.pyo
echo venv/
echo .venv/
echo *.egg-info/
echo dist/
echo build/
echo.
echo # Data - generated files, never commit
echo phase1/data/raw/
echo phase1/data/processed/
echo phase1/data/checkpoints/
echo phase2/data/faiss_index/
echo phase2/data/chroma_db/
echo !*/.gitkeep
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo.
echo # Node
echo node_modules/
echo phase5/react_app/build/
) > %ROOT%\.gitignore

:: ── Write .env.example content ────────────────────────────────────────────────
(
echo # Copy this file to .env and fill in your keys
echo # NEVER commit .env to git
echo.
echo # ── Phase 3+ LLM Keys ─────────────────────────
echo OPENAI_API_KEY=sk-...
echo ANTHROPIC_API_KEY=sk-ant-...
echo.
echo # ── Phase 1: No keys needed! ──────────────────
echo # Jikan and AniList are free with no auth.
echo.
echo # ── Phase 4: Redis ────────────────────────────
echo REDIS_URL=redis://localhost:6379
echo.
echo # ── Phase 6: GCP ──────────────────────────────
echo GCP_PROJECT_ID=your-project-id
echo GCP_REGION=us-central1
) > %ROOT%\.env.example

:: ── Write requirements.txt content ───────────────────────────────────────────
(
echo # Phase 1 - Data Collection
echo aiohttp==3.9.5
echo aiofiles==23.2.1
echo requests==2.31.0
echo tenacity==8.3.0
echo pandas==2.2.2
echo numpy==1.26.4
echo pydantic==2.7.1
echo python-dotenv==1.0.1
echo orjson==3.10.3
echo pyarrow==16.1.0
echo typer==0.12.3
echo rich==13.7.1
echo tqdm==4.66.4
echo.
echo # Phase 2 - Embeddings
echo sentence-transformers==3.0.1
echo chromadb==0.5.3
echo faiss-cpu==1.8.0
echo openai==1.35.0
echo.
echo # Phase 3 - RAG
echo langchain==0.2.6
echo langchain-openai==0.1.13
echo langchain-community==0.2.6
echo llama-index==0.10.51
echo.
echo # Phase 4 - API
echo fastapi==0.111.0
echo uvicorn==0.30.1
echo redis==5.0.6
echo slowapi==0.1.9
echo.
echo # Testing
echo pytest==8.2.0
echo pytest-asyncio==0.23.7
) > %ROOT%\requirements.txt

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ==========================================
echo   Structure created successfully!
echo  ==========================================
echo.
echo   Project folder : %ROOT%\
echo   Total phases   : 6
echo   Next step      : cd %ROOT%  and  python -m venv venv
echo.
echo   Quick start:
echo     cd %ROOT%
echo     python -m venv venv
echo     venv\Scripts\activate
echo     pip install -r requirements.txt
echo     python main.py run-all --test
echo.

pause
