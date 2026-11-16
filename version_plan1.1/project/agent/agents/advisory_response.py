"""Text extraction and bounded diagnostics for advisory JSON responses."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any


def extract_message_content(message: Any) -> str:
    if message is None:
        return ""
    content = (
        message.get("content", message)
        if isinstance(message, dict)
        else getattr(message, "content", message)
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {
                "text",
                "output_text",
            }:
                value = block.get("text", "")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    # Some local advisors return the decoded object directly.
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return ""


def decode_json_response(message: Any) -> dict:
    text = extract_message_content(message).strip()
    # Accept one explicitly fenced document, including explanatory prose.
    # Never guess from arbitrary braces or repair a truncated document.
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1).strip()
    elif "```" in text:
        fenced = re.search(
            r"^[ \t]*```(?:json)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
            text, re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        if text.count("```") != 2 or fenced is None:
            raise ValueError("advisory response must contain exactly one complete JSON fence")
        text = fenced.group(1).strip()
    if not text:
        raise ValueError("empty advisory text response")
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("advisory JSON response must be an object")
    return value


def response_diagnostics(message: Any, exc: Exception, *, stage: str) -> dict:
    metadata = (
        message.get("response_metadata", {})
        if isinstance(message, dict)
        else getattr(message, "response_metadata", {})
    )
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        **exception_diagnostics(exc),
        "failure_stage": stage,
        "response_type": type(message).__name__,
        "response_excerpt": extract_message_content(message)[:2000],
        "finish_reason": metadata.get("finish_reason"),
    }


def exception_diagnostics(exc: Exception) -> dict:
    """Retain network root causes without persisting credentials or headers."""
    def safe(value):
        text = str(value)
        for key, secret in os.environ.items():
            if any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")) and len(secret) >= 6:
                text = text.replace(secret, "[redacted]")
        text = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", text)
        text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", text)
        return text[:500]

    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": safe(current)})
        current = current.__cause__ or current.__context__
    return {
        "error": f"{type(exc).__name__}: {safe(exc)}",
        "exception_chain": chain,
        "http_status": getattr(exc, "status_code", None),
        "request_id": getattr(exc, "request_id", None),
    }


def retryable_request_error(exc: Exception) -> bool:
    import httpx
    from openai import APIConnectionError, APITimeoutError

    status = getattr(exc, "status_code", None)
    return (isinstance(exc, (APIConnectionError, APITimeoutError, httpx.TransportError,
                             ConnectionError, TimeoutError))
            or status in {408, 409, 429}
            or isinstance(status, int) and 500 <= status < 600)


def unsupported_feature(exc: Exception, feature: str) -> bool:
    # Do not reinterpret authentication, transport, or arbitrary schema errors as
    # a provider capability failure and silently discard the evidence/tool loop.
    text = str(exc).lower()
    return (getattr(exc, "status_code", None) in {400, 422}
            and feature in text
            and any(word in text for word in ("unsupported", "not support", "unknown", "unrecognized", "not allowed")))


def trial_ids_from_evidence(value):
    """Collect observed identity fields, never trial names from free-form prose."""
    found = set()
    if isinstance(value, dict):
        if isinstance(value.get("trial_id"), str):
            found.add(value["trial_id"])
        if isinstance(value.get("trial_ids"), list):
            found.update(item for item in value["trial_ids"] if isinstance(item, str))
        for item in value.values():
            found.update(trial_ids_from_evidence(item))
    elif isinstance(value, list):
        for item in value:
            found.update(trial_ids_from_evidence(item))
    return found


def correction_prompt(
    original_prompt: str, diagnostics: dict, *, observed_trial_ids=()
) -> str:
    """Ask for one complete regeneration without changing the phase or evidence."""
    try:
        payload = json.loads(original_prompt)
    except (TypeError, ValueError):
        payload = {"original_request": original_prompt}
    if not isinstance(payload, dict):
        payload = {"original_request": payload}
    payload["response_correction"] = {
        "instruction": (
            "Final correction attempt: return one complete JSON object, not a patch or explanation. "
            "Match schema and response_contract, including required keys, types and locked values. "
            "Use only supplied observations and permitted actions for the original phase. "
            "Do not invent evidence or copy synthetic example values. If a justified proposal cannot "
            "be formed, use the no-change/empty-suggestions schema with a brief truthful reason."
        ),
        "failure_stage": diagnostics["failure_stage"],
        "validation_error": diagnostics["error"],
    }
    if observed_trial_ids:
        payload["response_correction"]["observed_trial_ids"] = sorted(
            observed_trial_ids
        )
    if diagnostics.get("finish_reason") == "length":
        payload["response_correction"]["length_guidance"] = (
            "The previous response was truncated. Keep diagnostics and explanations short; "
            "omit copied history, but retain every required field and all necessary controls."
        )
    return json.dumps(payload, ensure_ascii=False)


def validate_advice_response(value: dict, prompt: str) -> None:
    """Validate the two advisory shapes consumed outside the HPO proposal parser."""
    try:
        payload = json.loads(prompt)
    except (TypeError, ValueError):
        return  # Legacy callers may send a free-form prompt without a contract.
    if not isinstance(payload, dict):
        return
    contract = payload.get("response_contract") or {}
    kind = contract.get("kind")
    if kind not in {"data_processing", "coordination"}:
        return
    required = {
        "diagnostics",
        "notes",
        "suggested_operations" if kind == "data_processing" else "risks",
    }
    if set(value) != required:
        raise ValueError(
            f"response keys must be exactly {sorted(required)}; got {sorted(value)}"
        )
    for key in required - {"suggested_operations"}:
        if not isinstance(value[key], list) or any(
            not isinstance(item, str) for item in value[key]
        ):
            raise ValueError(f"{key} must be an array of strings")
    if kind != "data_processing":
        return
    operations = value["suggested_operations"]
    if not isinstance(operations, list) or len(operations) > 5:
        raise ValueError(
            "suggested_operations must be an array of at most five objects"
        )
    context = payload.get("context") or {}
    available = context.get("available_operations") or {}
    seen = set()
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) != {
            "operation",
            "parameters",
            "reason",
        }:
            raise ValueError(
                "each operation requires exactly operation, parameters and reason"
            )
        name = operation["operation"]
        if not isinstance(name, str) or name not in available or name in seen:
            raise ValueError(
                "operation must be a distinct name from available_operations"
            )
        seen.add(name)
        if not isinstance(operation["parameters"], dict):
            raise ValueError("operation parameters must be an object")
        if (
            not isinstance(operation["reason"], str)
            or not operation["reason"].strip()
        ):
            raise ValueError(
                "operation reason must be a nonempty string tied to observed evidence"
            )
        descriptor = available[name]
        supported = descriptor.get("supported_data_types", [])
        if (
            "*" not in supported
            and context.get("dataset_type") not in supported
        ):
            raise ValueError(
                f"operation {name} does not support the dataset type"
            )
        parameter_schema = descriptor.get("parameter_schema") or {}
        for key in operation["parameters"]:
            if (
                key not in parameter_schema
                or parameter_schema[key].get("advisor_allowed", True) is False
            ):
                raise ValueError(
                    f"operation parameter is undeclared or protected: {key}"
                )
            parameter = operation["parameters"][key]
            declared_type = parameter_schema[key].get("type")
            matches = {
                "boolean": type(parameter) is bool,
                "integer": type(parameter) is int,
                "number": isinstance(parameter, (int, float))
                and not isinstance(parameter, bool),
                "string": isinstance(parameter, str),
                "array": isinstance(parameter, list),
                "object": isinstance(parameter, dict),
            }
            if declared_type in matches and not matches[declared_type]:
                raise ValueError(
                    f"operation parameter {key} must have JSON type {declared_type}"
                )
