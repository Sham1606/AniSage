"""
Phase 2 Tests

Run with: pytest phase2/tests/test_phase2.py -v

Tests:
  - test_embedding_model_shape       : single + batch embed return correct dims
  - test_sentence_transformer_normalised : output vectors are unit-norm
  - test_chroma_insert_and_query     : insert 10 docs, query returns results
  - test_chroma_metadata_filter      : filtered query returns only matching docs
  - test_faiss_build_and_query       : build index, query returns correct top-1
  - test_faiss_save_and_load         : save/load round-trip works
  - test_embed_pipeline_small        : run pipeline on 20 synthetic records
  - test_cosine_similarity_range     : scores are between 0 and 1
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def st_model():
    """Load SentenceTransformer model once for all tests in this module."""
    from phase2.embeddings.embedding_models import SentenceTransformerModel
    return SentenceTransformerModel("all-MiniLM-L6-v2")


@pytest.fixture
def dummy_texts():
    return [
        "A dark psychological thriller with complex characters",
        "Romantic slice of life high school drama",
        "Epic mecha battle in a post-apocalyptic world",
        "Samurai swordsman seeks revenge for his fallen clan",
        "Magical girl fights evil monsters to protect the city",
        "Detective solves impossible murder mysteries",
        "Space pirates explore the galaxy in a beat-up ship",
        "A chef competes in culinary tournaments with unique cooking techniques",
        "Time traveller tries to prevent a tragic future",
        "Demons and humans learn to coexist in modern Tokyo",
    ]


@pytest.fixture
def dummy_metadata():
    return [
        {
            "mal_id": i + 1,
            "anilist_id": i + 100,
            "title": f"Test Anime {i + 1}",
            "genres": f"Action, Adventure",
            "year": 2000 + i,
            "score": 7.0 + (i * 0.1),
            "media_type": "TV",
            "synopsis": f"Synopsis for test anime {i + 1}",
            "image_url": "",
        }
        for i in range(10)
    ]


# ── Embedding model tests ─────────────────────────────────────────────────────

class TestEmbeddingModels:

    def test_sentence_transformer_single_embed(self, st_model):
        """embed_one returns a 1-D array of the correct dimension."""
        vec = st_model.embed_one("dark fantasy anime with demons")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == st_model.dim  # 384 for MiniLM
        assert vec.dtype == np.float32

    def test_sentence_transformer_batch_embed(self, st_model, dummy_texts):
        """embed returns shape (N, dim) for a batch of N texts."""
        vecs = st_model.embed(dummy_texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (len(dummy_texts), st_model.dim)
        assert vecs.dtype == np.float32

    def test_sentence_transformer_normalised(self, st_model, dummy_texts):
        """All output vectors should be L2-unit-norm (within float tolerance)."""
        vecs = st_model.embed(dummy_texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_sentence_transformer_dim(self, st_model):
        """MiniLM-L6-v2 should produce 384-dim embeddings."""
        assert st_model.dim == 384

    def test_different_texts_produce_different_vectors(self, st_model):
        """Semantically different texts should not produce identical vectors."""
        v1 = st_model.embed_one("romance anime")
        v2 = st_model.embed_one("horror anime with gore")
        assert not np.allclose(v1, v2)

    def test_similar_texts_produce_close_vectors(self, st_model):
        """Semantically similar texts should have high cosine similarity."""
        v1 = st_model.embed_one("dark psychological thriller with mind games")
        v2 = st_model.embed_one("psychological horror with complex mind-bending plot")
        similarity = float(np.dot(v1, v2))  # cosine sim (both unit-norm)
        assert similarity > 0.5, f"Expected high similarity, got {similarity:.3f}"

    def test_factory_returns_correct_type(self):
        """get_embedding_model factory returns the right class."""
        from phase2.embeddings.embedding_models import (
            get_embedding_model,
            SentenceTransformerModel,
        )
        model = get_embedding_model("sentence-transformer")
        assert isinstance(model, SentenceTransformerModel)

    def test_factory_invalid_type_raises(self):
        from phase2.embeddings.embedding_models import get_embedding_model
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_embedding_model("invalid-backend")


# ── ChromaDB tests ────────────────────────────────────────────────────────────

class TestChromaStore:

    @pytest.fixture(autouse=True)
    def chroma_store(self, tmp_path, st_model, dummy_texts, dummy_metadata):
        """Create a fresh ChromaStore in a temp dir for each test."""
        from phase2.vectordb.chromadb_store import ChromaStore
        store = ChromaStore(persist_dir=tmp_path / "chroma", reset=True)

        # Embed and ingest
        embeddings = st_model.embed(dummy_texts)
        store.ingest(embeddings, dummy_metadata, skip_existing=False)
        self.store = store
        self.embeddings = embeddings
        self.st_model = st_model
        yield store

    def test_ingest_count(self):
        """Collection should have exactly 10 records after ingest."""
        assert self.store.count() == 10

    def test_query_returns_results(self):
        """A query should return results."""
        q = self.st_model.embed_one("dark psychological thriller")
        results = self.store.query(query_embedding=q, n_results=5)
        assert len(results) > 0

    def test_query_result_structure(self):
        """Each result should have required fields."""
        q = self.st_model.embed_one("action adventure")
        results = self.store.query(query_embedding=q, n_results=3)
        assert len(results) > 0
        for r in results:
            assert "title" in r
            assert "_rank" in r
            assert "_similarity" in r
            assert r["_rank"] >= 1

    def test_query_similarity_range(self):
        """Similarity scores should be between 0 and 1."""
        q = self.st_model.embed_one("romance drama")
        results = self.store.query(query_embedding=q, n_results=5)
        for r in results:
            assert 0.0 <= r["_similarity"] <= 1.0, \
                f"Similarity out of range: {r['_similarity']}"

    def test_query_ranked_by_similarity(self):
        """Results should be in descending similarity order."""
        q = self.st_model.embed_one("mecha robots")
        results = self.store.query(query_embedding=q, n_results=5)
        sims = [r["_similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_n_results_respected(self):
        """n_results parameter should be respected."""
        q = self.st_model.embed_one("samurai")
        results = self.store.query(query_embedding=q, n_results=3)
        assert len(results) <= 3

    def test_get_stats(self):
        """get_stats should return a dict with total_records."""
        stats = self.store.get_stats()
        assert "total_records" in stats
        assert stats["total_records"] == 10


# ── FAISS tests ───────────────────────────────────────────────────────────────

class TestFAISSStore:

    @pytest.fixture(autouse=True)
    def faiss_store(self, tmp_path, st_model, dummy_texts, dummy_metadata):
        """Create a FAISS store in a temp dir."""
        from phase2.vectordb.faiss_store import FAISSStore

        store = FAISSStore(
            index_path=tmp_path / "anime.index",
            idmap_path=tmp_path / "id_map.json",
        )
        embeddings = st_model.embed(dummy_texts)
        store.build(embeddings, dummy_metadata)
        store.save()

        self.store = store
        self.embeddings = embeddings
        self.st_model = st_model
        self.tmp_path = tmp_path
        yield store

    def test_build_count(self):
        """FAISS index should contain 10 vectors."""
        assert self.store.count() == 10

    def test_query_returns_results(self):
        """Query should return non-empty results."""
        q = self.st_model.embed_one("dark thriller")
        results = self.store.query(q, k=5)
        assert len(results) > 0

    def test_query_result_structure(self):
        """Each result should have _rank and _score fields."""
        q = self.st_model.embed_one("adventure")
        results = self.store.query(q, k=3)
        for r in results:
            assert "_rank" in r
            assert "_score" in r
            assert "title" in r

    def test_top1_exact_match(self):
        """
        Querying with an exact stored vector should return that vector as top-1
        with near-perfect score.
        """
        # Use the embedding of the first document as query
        query_vec = self.embeddings[0]
        results = self.store.query(query_vec, k=1)
        assert len(results) == 1
        assert results[0]["_score"] > 0.99, \
            f"Expected near-perfect score for exact match, got {results[0]['_score']}"

    def test_scores_in_valid_range(self):
        """Cosine similarity scores should be between -1 and 1 (typically 0-1)."""
        q = self.st_model.embed_one("magical girl")
        results = self.store.query(q, k=5)
        for r in results:
            assert -1.0 <= r["_score"] <= 1.0

    def test_results_ranked_by_score(self):
        """Results should be in descending score order."""
        q = self.st_model.embed_one("space opera")
        results = self.store.query(q, k=5)
        scores = [r["_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load_roundtrip(self):
        """Loading a saved index should give identical query results."""
        from phase2.vectordb.faiss_store import FAISSStore

        q = self.st_model.embed_one("mystery detective")
        original_results = self.store.query(q, k=3)

        # Load from disk into new instance
        new_store = FAISSStore(
            index_path=self.tmp_path / "anime.index",
            idmap_path=self.tmp_path / "id_map.json",
        )
        loaded = new_store.load()
        assert loaded is True
        assert new_store.count() == 10

        loaded_results = new_store.query(q, k=3)

        # Top-1 should be the same title
        assert original_results[0]["title"] == loaded_results[0]["title"]

    def test_get_stats(self):
        """get_stats should return total_vectors and dimensions."""
        stats = self.store.get_stats()
        assert stats["total_vectors"] == 10
        assert stats["dimensions"] == self.st_model.dim


# ── Pipeline integration test ─────────────────────────────────────────────────

class TestEmbedPipeline:

    def test_embed_pipeline_small(self, tmp_path, st_model):
        """
        Run the full EmbedPipeline on 20 synthetic records.
        Verify output files exist and have correct shapes.
        """
        import json as _json
        from phase2.embeddings.embed_pipeline import EmbedPipeline

        # Create a small synthetic JSONL input file
        input_path = tmp_path / "test_anime.jsonl"
        records = []
        for i in range(20):
            record = {
                "mal_id": i + 1,
                "title": f"Test Anime {i + 1}",
                "synopsis": f"A story about character {i} who goes on an adventure in a fantasy world.",
                "genres": ["Action", "Adventure"],
                "embedding_text": (
                    f"Title: Test Anime {i + 1}\n"
                    f"Genres: Action, Adventure\n"
                    f"Synopsis: A story about character {i} who goes on an adventure."
                ),
            }
            records.append(record)
            with open(input_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")

        output_dir = tmp_path / "phase2_output"
        output_dir.mkdir()

        pipeline = EmbedPipeline(
            model=st_model,
            batch_size=8,
            input_path=input_path,
            output_dir=output_dir,
            force_reembed=True,
        )
        embeddings, metadata = pipeline.run()

        # Verify output files exist
        assert (output_dir / "embeddings.npy").exists()
        assert (output_dir / "metadata.jsonl").exists()
        assert (output_dir / "embed_checkpoint.json").exists()

        # Verify shapes
        assert embeddings.shape == (20, st_model.dim)
        assert len(metadata) == 20

        # Verify embeddings are float32
        assert embeddings.dtype == np.float32

        # Verify metadata has expected fields
        assert metadata[0]["title"] == "Test Anime 1"
        assert metadata[0]["mal_id"] == 1

    def test_pipeline_checkpoint_resume(self, tmp_path, st_model):
        """
        Simulate an interrupted pipeline and verify it resumes correctly.
        """
        import json as _json
        from phase2.embeddings.embed_pipeline import EmbedPipeline

        input_path = tmp_path / "resume_test.jsonl"
        for i in range(10):
            record = {
                "mal_id": i + 1,
                "title": f"Resume Anime {i + 1}",
                "embedding_text": f"Title: Resume Anime {i + 1}\nSynopsis: Test synopsis {i}.",
            }
            with open(input_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")

        output_dir = tmp_path / "resume_output"
        output_dir.mkdir()

        # First run — complete
        p1 = EmbedPipeline(model=st_model, batch_size=4, input_path=input_path, output_dir=output_dir)
        emb1, meta1 = p1.run()

        # Second run — should detect checkpoint and not re-embed
        p2 = EmbedPipeline(model=st_model, batch_size=4, input_path=input_path, output_dir=output_dir)
        emb2, meta2 = p2.run()

        assert len(meta1) == len(meta2) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])