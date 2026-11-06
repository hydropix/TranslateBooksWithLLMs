"""Refinement: placeholder-corruption retry and chunk-based progress (issue #294)."""
import asyncio
from types import SimpleNamespace

from src.core.epub import xhtml_translator as xt
from src.core.refine import epub_refiner

FMT = ("[id", "]")
DRAFT = "[id0]Hello[id1]"
TAG_MAP = {"[id0]": "<b>", "[id1]": "</b>"}
CHUNK = {"local_tag_map": TAG_MAP, "global_indices": [0, 1]}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def make_request(self, user, model, system_prompt=None):
        self.prompts.append(user)
        content = self.replies.pop(0)
        return SimpleNamespace(content=content, prompt_tokens=1, completion_tokens=1,
                               context_used=2, context_limit=10)

    def extract_translation(self, content):
        return content


def _refine(client, retries):
    return asyncio.run(xt._refine_epub_chunks(
        [DRAFT], [CHUNK], "French", "m", client, None, FMT, None, None,
        max_refine_retries=retries))


def test_corrupted_then_valid_keeps_refined_text():
    client = FakeClient(["[id0]Broken", "[id0]Refined[id1]"])
    assert _refine(client, 1) == ["[id0]Refined[id1]"]
    assert len(client.prompts) == 2
    assert "IMPORTANT" in client.prompts[1] and "[id0]" in client.prompts[1]
    assert "IMPORTANT" not in client.prompts[0]


def test_corrupted_twice_falls_back_to_draft():
    client = FakeClient(["[id0]Broken", "Still broken"])
    assert _refine(client, 1) == [DRAFT]
    assert len(client.prompts) == 2


def test_retry_disabled_falls_back_immediately():
    client = FakeClient(["[id0]Broken"])
    assert _refine(client, 0) == [DRAFT]
    assert len(client.prompts) == 1


def test_refine_progress_never_exceeds_total(tmp_path, monkeypatch):
    """Multi-file refine reports chunk counts in one unit, never above 100%."""
    files = {"a.xhtml": 3, "b.xhtml": 2, "c.xhtml": 4}
    monkeypatch.setattr(epub_refiner, "_count_file_chunks",
                        lambda root, _m: int(root.get("n")))
    monkeypatch.setattr(epub_refiner, "_extract_epub", lambda *a, **k: None)
    monkeypatch.setattr(epub_refiner, "_parse_epub_manifest", lambda *a, **k: {
        "content_files": list(files), "opf_dir": str(tmp_path)})
    monkeypatch.setattr(epub_refiner, "_repackage_epub", lambda **k: None)
    monkeypatch.setattr(epub_refiner, "build_refine_client",
                        lambda **k: (SimpleNamespace(), None))
    for name, n in files.items():
        (tmp_path / name).write_text(f'<html xmlns="http://www.w3.org/1999/xhtml" n="{n}"><body/></html>')

    async def fake_refine_one(doc_root, stats_callback=None, **kw):
        n = int(doc_root.get("n"))
        for i in range(1, n + 1):
            stats_callback({"total_chunks": n, "completed_chunks": i, "failed_chunks": 0})
        return True

    monkeypatch.setattr(epub_refiner, "_refine_one_xhtml", fake_refine_one)
    seen = []
    epub = tmp_path / "in.epub"
    epub.write_bytes(b"")
    ok = asyncio.run(epub_refiner.refine_epub_file(
        str(epub), str(tmp_path / "out.epub"), "French", stats_callback=seen.append))
    assert ok
    assert seen[0]["total_chunks"] == 9
    assert all(s["total_chunks"] == 9 and s["completed_chunks"] <= 9 for s in seen)
    assert seen[-1]["completed_chunks"] == 9
    done = [s["completed_chunks"] for s in seen]
    assert done == sorted(done)



def test_reminder_lists_placeholders_in_numeric_order():
    tag_map = {f"[id{i}]": "<x>" for i in range(13)}
    reminder = xt._build_refine_placeholder_reminder(tag_map)
    expected = ", ".join(f"[id{i}]" for i in range(13))
    assert expected in reminder
