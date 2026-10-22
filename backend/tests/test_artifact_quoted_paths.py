"""Quoted sandbox paths keep their complete filenames in persistent handles."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from deerflow.agents.middlewares.artifact_capture_middleware import ArtifactCaptureMiddleware
from deerflow.agents.middlewares.artifact_resolution_middleware import ArtifactResolutionMiddleware
from deerflow.agents.thread_state import merge_tool_artifacts
from deerflow.tools.artifact_registry import extract_artifacts_from_result


def _entries(content, *, artifact=None):
    result = ToolMessage(content=content, artifact=artifact, tool_call_id="export-1", name="export_report")
    return extract_artifacts_from_result(result, thread_id="thread-1")


@pytest.mark.parametrize("quote", ['"', "'", "`"])
@pytest.mark.parametrize("text_block", [False, True])
def test_quoted_path_with_spaces_is_one_complete_reference(quote, text_block):
    path = "/mnt/user-data/outputs/quarterly report.pdf"
    text = f"Created {quote}{path}{quote} successfully."
    content = [{"type": "text", "text": text}] if text_block else text

    entries = _entries(content)

    assert [entry["real_ref"] for entry in entries] == [path]
    assert entries[0]["display_name"] == "quarterly report.pdf"


def test_multiple_quoted_and_bare_paths_preserve_order_without_prefix_artifacts():
    paths = [
        "/mnt/user-data/outputs/first report.pdf",
        "/mnt/user-data/outputs/second report.pdf",
        "/mnt/user-data/outputs/control.txt",
    ]
    for text, order in [
        (f'Saved "{paths[0]}", `{paths[1]}`, and {paths[2]}.', paths),
        (f'Saved {paths[2]}, "{paths[0]}", and `{paths[1]}`.', [paths[2], paths[0], paths[1]]),
        (f'Saved "{paths[0]}", {paths[2]}, and `{paths[1]}`.', [paths[0], paths[2], paths[1]]),
    ]:
        assert [entry["real_ref"] for entry in _entries(text)] == order


@pytest.mark.parametrize("quote", ['"', "'", "`"])
def test_quoted_text_deduplicates_with_exact_structured_reference(quote):
    path = "/mnt/user-data/outputs/quarterly report.pdf"
    entries = _entries(f"Saved {quote}{path}{quote}.", artifact={"structured_content": {"file_path": path}})

    assert [entry["real_ref"] for entry in entries] == [path]


@pytest.mark.parametrize("quote", ['"', "'", "`"])
@pytest.mark.parametrize("separator", ["", "\n", "\r\n"])
def test_unclosed_or_multiline_quoted_paths_do_not_create_partial_handles(quote, separator):
    text = f"Saved {quote}/mnt/user-data/outputs/quarterly report"
    if separator:
        text += f"{separator}continued.pdf{quote}"
    text += "\nControl /mnt/user-data/outputs/control.txt"

    assert [entry["real_ref"] for entry in _entries(text)] == ["/mnt/user-data/outputs/control.txt"]


def test_quoted_filename_preserves_literal_punctuation_and_other_quote_characters():
    path = "/mnt/user-data/outputs/O'Brien report (final).txt"
    assert [entry["real_ref"] for entry in _entries(f'Wrote "{path}".')] == [path]


@pytest.mark.parametrize("suffix", ["report .", "report,", "report)"])
def test_quoted_filename_keeps_literal_trailing_punctuation(suffix):
    path = f"/mnt/user-data/outputs/{suffix}"
    assert [entry["real_ref"] for entry in _entries(f'Wrote "{path}".')] == [path]


def test_quoted_path_does_not_create_handles_for_embedded_references():
    path = "/mnt/user-data/outputs/folder /mnt/user-data/x https://example.com/a.pdf"
    control_url = "https://example.com/control.pdf"
    entries = _entries(f'Wrote "{path}" and {control_url}.')
    assert [entry["real_ref"] for entry in entries] == [path, control_url]


@pytest.mark.parametrize("closed", [False, True])
def test_rejected_quoted_span_does_not_leak_embedded_references(closed):
    path = "/mnt/user-data/outputs/" + "a" * 4096 + " /mnt/user-data/x https://example.com/a.pdf"
    text = f'Wrote "{path}' + ('"' if closed else "") + "\nControl /mnt/user-data/outputs/control.txt"
    assert [entry["real_ref"] for entry in _entries(text)] == ["/mnt/user-data/outputs/control.txt"]


def test_bare_path_retains_conservative_whitespace_boundary():
    entries = _entries("Saved /mnt/user-data/outputs/report with a comment")
    assert [entry["real_ref"] for entry in entries] == ["/mnt/user-data/outputs/report"]


@pytest.mark.parametrize("quote", ['"', "'", "`"])
def test_persisted_quoted_handle_resolves_to_readable_file(tmp_path: Path, quote):
    filename = "quarterly report.txt"
    path = f"/mnt/user-data/outputs/{filename}"
    (tmp_path / filename).write_text("The complete quarterly report", encoding="utf-8")
    result = ToolMessage(content=f"Saved {quote}{path}{quote}.", tool_call_id="export-1", name="export_report")
    update = ArtifactCaptureMiddleware().before_model({"messages": [result]}, SimpleNamespace(context={"thread_id": "thread-1"}))
    assert update is not None
    registry = merge_tool_artifacts([], update["tool_artifacts"])
    # Drop the source message and round-trip only the persistent registry, as
    # after checkpoint reload / context compaction.
    state = {"tool_artifacts": json.loads(json.dumps(registry))}
    (entry,) = state["tool_artifacts"]
    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {"path": entry["handle"]}, "id": "read-1", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=None,
    )

    def read_file(resolved):
        resolved_path = resolved.tool_call["args"]["path"]
        relative = Path(resolved_path).relative_to("/mnt/user-data/outputs")
        return ToolMessage(content=(tmp_path / relative).read_text(encoding="utf-8"), tool_call_id="read-1")

    read_result = ArtifactResolutionMiddleware().wrap_tool_call(request, read_file)
    assert read_result.content == "The complete quarterly report"
