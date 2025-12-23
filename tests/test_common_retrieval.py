from stack import retrieval
from stack.common import NOT_IN_DOCS, finish


def test_finish_plain_answer_and_refusal():
    assert finish("Use it [1].", ["a#b"]).sources == ["a#b"]
    r = finish(NOT_IN_DOCS, ["a#b"])
    assert r.refused and r.sources == []


def test_finish_drops_the_whole_think_block():
    # qwen3 can emit its reasoning inline; the reasoning must not count as the answer
    text = "<think>Maybe the answer is NOT_IN_DOCS? No, passage 1 covers it.</think>\nUse withoutGlobalScope [1]."
    r = finish(text, ["eloquent.md#Removing Global Scopes"])
    assert not r.refused and r.answer == "Use withoutGlobalScope [1]."
    assert finish("<think>\n\n</think>\n\nNOT_IN_DOCS", []).refused


def test_index_and_search(corpus_index):
    hits = retrieval.search("remove a global scope for one query", k=2)
    assert hits[0].cite() == "eloquent.md#Removing Global Scopes" and hits[0].score > hits[1].score
    text = retrieval.format_passages(hits)
    assert text.startswith("[1] (eloquent.md#Removing Global Scopes)")


def test_sections_skip_anchors_and_tiny_sections(tmp_path):
    p = tmp_path / "x.md"
    p.write_text('<a name="a"></a>\n## Short\n\ntoo short\n\n## Long\n\n' + "word " * 30)
    assert [h for h, _ in retrieval._sections(p)] == ["Long"]
