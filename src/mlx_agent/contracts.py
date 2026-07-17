"""Versioned result contracts shared by every provider adapter."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


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
        return cls(operation=operation, status="ok", data=data, warnings=list(warnings or []))

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
