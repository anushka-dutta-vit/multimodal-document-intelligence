import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingestion.chunker import chunk_text, split_into_sentences


def test_split_into_sentences_basic():
    text = "This is one. This is two! Is this three?"
    sentences = split_into_sentences(text)
    assert sentences == ["This is one.", "This is two!", "Is this three?"]


def test_split_into_sentences_empty():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_chunk_text_respects_chunk_size():
    text = " ".join([f"Sentence number {i}." for i in range(50)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # allow slack since we never split mid-sentence
        assert len(c) <= 160


def test_chunk_text_single_short_text():
    text = "Just one short sentence."
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert chunks == ["Just one short sentence."]


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=800, overlap=150) == []


def test_chunk_overlap_preserves_tail_context():
    text = " ".join([f"Fact{i} is true." for i in range(30)])
    chunks = chunk_text(text, chunk_size=120, overlap=40)
    assert len(chunks) >= 2
    # the start of chunk 2 should share some characters with the tail of chunk 1
    tail_of_first = chunks[0][-40:]
    assert any(word in chunks[1] for word in tail_of_first.split())