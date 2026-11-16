"""Durable bounded consultations; response persistence precedes policy application."""
from datetime import datetime
from uuid import uuid4

from .resources import ResourceBudgetExhausted


def request_response(record, persist, request, *, encode=lambda x: x, max_attempts=3):
    if "response" in record:
        return record["response"]
    record.setdefault("request_id", uuid4().hex)
    record.setdefault("attempts", 0)
    record.setdefault("max_attempts", max_attempts)
    record.setdefault("errors", [])
    while record["attempts"] < record["max_attempts"]:
        record.update(status="requested", attempts=record["attempts"] + 1,
                      requested_at=datetime.now().isoformat())
        persist()
        try:
            response = encode(request())
            if response is None or (isinstance(response, dict) and response.get("advice_error")):
                raise ValueError(str(response.get("advice_error")) if isinstance(response, dict) else "advisor returned no valid response")
        except ResourceBudgetExhausted:
            record["status"] = "interrupted"
            persist()
            raise
        except Exception as exc:
            record["status"] = "retryable_error"
            record["errors"].append(f"{type(exc).__name__}: {exc}")
            persist()
            continue
        record.update(status="response_saved", response=response, responded_at=datetime.now().isoformat())
        persist()
        return response
    record["status"] = "exhausted"
    persist()
    return None
