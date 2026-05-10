"""
Shared helpers for Gemini managed Batch API jobs.

The Batch API is asynchronous, so callers submit a job, poll until it reaches a
terminal state, and then read inline responses in the same order as requests.
"""
import os
import time
from typing import Any


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

DEFAULT_POLL_SECONDS = int(os.environ.get("GEMINI_BATCH_POLL_SECONDS", "30"))
DEFAULT_MAX_WAIT_SECONDS = int(os.environ.get("GEMINI_BATCH_MAX_WAIT_SECONDS", "86400"))


def state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    if hasattr(state, "name"):
        return state.name
    return str(state)


def wait_for_batch(
    client: Any,
    job: Any,
    *,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS,
) -> Any:
    """Poll a Gemini batch job until completion or timeout."""
    job_name = job.name
    deadline = time.time() + max_wait_seconds
    started_at = time.time()
    print(f"Created Gemini batch job: {job_name}", flush=True)

    while True:
        job = client.batches.get(name=job_name)
        current_state = state_name(job)
        elapsed = int(time.time() - started_at)
        print(f"  Batch state: {current_state} (elapsed={elapsed}s)", flush=True)
        if current_state in COMPLETED_STATES:
            break
        if time.time() >= deadline:
            raise TimeoutError(
                f"Gemini batch job {job_name} did not finish within "
                f"{max_wait_seconds} seconds"
            )
        time.sleep(poll_seconds)

    if state_name(job) != "JOB_STATE_SUCCEEDED":
        error = getattr(job, "error", None)
        raise RuntimeError(f"Gemini batch job {job_name} ended as {state_name(job)}: {error}")
    return job


def inline_generate_responses(job: Any) -> list[Any]:
    dest = getattr(job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) if dest else None
    if responses is None:
        raise RuntimeError("Gemini batch job completed without inline generate responses")
    return list(responses)


def inline_embedding_responses(job: Any) -> list[Any]:
    dest = getattr(job, "dest", None)
    responses = getattr(dest, "inlined_embed_content_responses", None) if dest else None
    if responses is None:
        raise RuntimeError("Gemini batch job completed without inline embedding responses")
    return list(responses)


def response_text(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", None)
    if text is not None:
        return text
    if isinstance(response, dict):
        if response.get("text"):
            return response["text"]
        candidates = response.get("candidates") or []
        if candidates:
            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )
            return "".join(part.get("text", "") for part in parts)
    return str(response)


def embedding_values(embedding: Any) -> list[float]:
    values = getattr(embedding, "values", None)
    if values is None and isinstance(embedding, dict):
        values = embedding.get("values")
    if values is None:
        raise RuntimeError(f"Unable to extract embedding values from {embedding!r}")
    return list(values)


def response_embeddings(response: Any) -> list[Any]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is None:
        embedding = getattr(response, "embedding", None)
        if embedding is None and isinstance(response, dict):
            embedding = response.get("embedding")
        if embedding is not None:
            return [embedding]
    if embeddings is None:
        raise RuntimeError(f"Unable to extract embeddings from {response!r}")
    return list(embeddings)
