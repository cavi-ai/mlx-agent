"""Versioned result contracts shared by every provider adapter."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError("{0} must be a string".format(name))
    if not value:
        raise ValueError("{0} must not be empty".format(name))


def _validate_warnings(warnings: Any) -> List[Dict[str, str]]:
    if warnings is None:
        return []
    if not isinstance(warnings, list):
        raise TypeError("warnings must be a list of objects of strings")

    validated: List[Dict[str, str]] = []
    for index, warning in enumerate(warnings):
        if not isinstance(warning, dict) or not all(
            isinstance(item, str) for item in warning.values()
        ):
            raise TypeError("warnings[{0}] must be an object of strings".format(index))
        validated.append(dict(warning))
    return validated


@dataclass(frozen=True)
class ErrorDetail:
    """A stable, actionable description of an unsuccessful operation."""

    code: str
    message: str
    remediation: str
    retryable: bool = False


@dataclass(frozen=True)
class ResultEnvelope:
    """The provider-neutral result returned by MLX agent operations."""

    operation: str
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[ErrorDetail] = None
    schema_version: str = SCHEMA_VERSION
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def ok(cls, operation: str, data: Dict[str, Any], warnings=None):
        """Create a successful versioned result."""
        _require_non_empty_string(operation, "operation")
        if not isinstance(data, dict):
            raise TypeError("data must be an object")
        return cls(
            operation=operation,
            status="ok",
            data=dict(data),
            warnings=_validate_warnings(warnings),
        )

    @classmethod
    def fail(
        cls,
        operation: str,
        code: str,
        message: str,
        remediation: str,
        retryable=False,
    ):
        """Create an error result with a machine-readable recovery path."""
        _require_non_empty_string(operation, "operation")
        _require_non_empty_string(code, "code")
        _require_non_empty_string(message, "message")
        _require_non_empty_string(remediation, "remediation")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean")
        return cls(
            operation=operation,
            status="error",
            error=ErrorDetail(code, message, remediation, retryable),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the envelope without an empty error object on success."""
        value = asdict(self)
        if self.error is None:
            value.pop("error")
        return value
