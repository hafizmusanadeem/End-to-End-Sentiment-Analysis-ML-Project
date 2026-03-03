"""
exception.py — Centralized exception handling for sentiment-mlops
-----------------------------------------------------------------
Design decisions:
  - Single exception class (SentimentMLOpsException) with error codes
  - Full traceback preserved via `raise ... from original_exc`
  - Rich metadata captured automatically on raise:
        timestamp, hostname, python version, active MLflow run_id,
        pipeline stage (if passed), env (APP_ENV)
  - boto3 / AWS ClientError + NoCredentialsError wrapped automatically
    via the `aws_exception_handler` context manager
  - Manual logging only — callers decide when/how to log
"""

import os
import sys
import socket
import traceback
import platform
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# 1. Error codes
# ──────────────────────────────────────────────

class ErrorCode(str, Enum):
    """
    Categorical error codes across the full ML pipeline.
    Using str mixin so codes serialise cleanly in logs / API responses.
    """
    # Data layer
    DATA_INGESTION_FAILED       = "E100"
    DATA_VALIDATION_FAILED      = "E101"
    DATA_TRANSFORMATION_FAILED  = "E102"
    DATA_SCHEMA_MISMATCH        = "E103"
    DATA_EMPTY_DATASET          = "E104"

    # AWS / S3
    AWS_CREDENTIALS_MISSING     = "E200"
    AWS_S3_UPLOAD_FAILED        = "E201"
    AWS_S3_DOWNLOAD_FAILED      = "E202"
    AWS_S3_BUCKET_NOT_FOUND     = "E203"
    AWS_CLIENT_ERROR            = "E204"

    # Model layer
    MODEL_TRAINING_FAILED       = "E300"
    MODEL_EVALUATION_FAILED     = "E301"
    MODEL_REGISTRY_FAILED       = "E302"
    MODEL_LOAD_FAILED           = "E303"
    MODEL_PREDICTION_FAILED     = "E304"

    # MLflow
    MLFLOW_TRACKING_FAILED      = "E400"
    MLFLOW_RUN_NOT_ACTIVE       = "E401"

    # Config / IO
    CONFIG_LOAD_FAILED          = "E500"
    FILE_NOT_FOUND              = "E501"
    SERIALIZATION_FAILED        = "E502"

    # API
    API_INVALID_INPUT           = "E600"
    API_PREDICTION_TIMEOUT      = "E601"

    # Generic fallback
    UNKNOWN_ERROR               = "E999"


# ──────────────────────────────────────────────
# 2. Metadata helper
# ──────────────────────────────────────────────

def _collect_metadata(stage: Optional[str] = None) -> dict[str, Any]:
    """
    Snapshot runtime context at the moment the exception is constructed.
    MLflow run_id is captured only if mlflow is importable and a run is active,
    so there is zero hard dependency on MLflow from this module.
    """
    mlflow_run_id: Optional[str] = None
    try:
        import mlflow  # local import — avoids circular deps
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow_run_id = active_run.info.run_id
    except Exception:
        pass  # MLflow not available or no active run — silently skip

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "hostname":       socket.gethostname(),
        "python_version": platform.python_version(),
        "platform":       platform.system(),
        "env":            os.getenv("APP_ENV", "dev"),
        "stage":          stage,
        "mlflow_run_id":  mlflow_run_id,
    }


# ──────────────────────────────────────────────
# 3. Core exception class
# ──────────────────────────────────────────────

class SentimentMLOpsException(Exception):
    """
    Single, unified exception for the entire sentiment-mlops pipeline.

    Parameters
    ----------
    message : str
        Human-readable description of what went wrong.
    error_code : ErrorCode
        Categorical code for programmatic handling / alerting.
    stage : str, optional
        Pipeline stage label ("data_ingestion", "model_trainer", etc.).
        Logged in metadata for fast filtering.
    original_exception : Exception, optional
        The upstream exception being wrapped. Its traceback is captured
        and the raise chain is preserved.

    Usage
    -----
        # Basic raise
        raise SentimentMLOpsException(
            "Failed to load config",
            ErrorCode.CONFIG_LOAD_FAILED,
        )

        # Wrapping an upstream exception (preserves full traceback)
        except FileNotFoundError as e:
            raise SentimentMLOpsException(
                "Config file missing",
                ErrorCode.FILE_NOT_FOUND,
                stage="configuration",
                original_exception=e,
            ) from e

        # Accessing structured info after catching
        except SentimentMLOpsException as e:
            print(e.error_code)       # ErrorCode.FILE_NOT_FOUND  → "E501"
            print(e.metadata)         # dict with timestamp, hostname, etc.
            print(e.traceback_str)    # full formatted traceback string
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        stage: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        self.message            = message
        self.error_code         = error_code
        self.original_exception = original_exception
        self.metadata           = _collect_metadata(stage=stage)
        self.traceback_str      = self._extract_traceback(original_exception)

        super().__init__(self._format_message())

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_traceback(exc: Optional[Exception]) -> Optional[str]:
        """Return a formatted traceback string from an existing exception."""
        if exc is None:
            return None
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return "".join(tb_lines).strip()

    def _format_message(self) -> str:
        """
        Build the string representation shown in logs / stack traces.
        Rich enough to be self-contained without opening the log file.
        """
        parts = [
            f"[{self.error_code}] {self.message}",
            f"  env={self.metadata['env']}",
            f"  stage={self.metadata['stage'] or 'unspecified'}",
            f"  host={self.metadata['hostname']}",
            f"  ts={self.metadata['timestamp']}",
        ]
        if self.metadata["mlflow_run_id"]:
            parts.append(f"  mlflow_run_id={self.metadata['mlflow_run_id']}")
        if self.original_exception is not None:
            parts.append(
                f"  caused_by={type(self.original_exception).__name__}: "
                f"{self.original_exception}"
            )
        return "\n".join(parts)

    # ── public helpers ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the exception to a plain dict — useful for API error
        responses (FastAPI) and structured log entries.

        Example FastAPI usage
        ---------------------
            except SentimentMLOpsException as e:
                raise HTTPException(status_code=500, detail=e.to_dict())
        """
        return {
            "error_code":   self.error_code.value,
            "error_name":   self.error_code.name,
            "message":      self.message,
            "metadata":     self.metadata,
            "traceback":    self.traceback_str,
            "caused_by":    (
                f"{type(self.original_exception).__name__}: {self.original_exception}"
                if self.original_exception else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"SentimentMLOpsException("
            f"error_code={self.error_code!r}, "
            f"stage={self.metadata['stage']!r}, "
            f"message={self.message!r})"
        )


# ──────────────────────────────────────────────
# 4. AWS auto-wrapping context manager
# ──────────────────────────────────────────────

class aws_exception_handler:
    """
    Context manager that automatically wraps boto3 / botocore errors into
    SentimentMLOpsException with the appropriate ErrorCode.

    Usage
    -----
        with aws_exception_handler(stage="data_ingestion"):
            s3_client.download_file(bucket, key, local_path)

    Wrapped error types
    -------------------
        NoCredentialsError       → ErrorCode.AWS_CREDENTIALS_MISSING
        botocore.ClientError     → mapped by HTTP status code:
            403 / AccessDenied   → AWS_CREDENTIALS_MISSING
            404 / NoSuchBucket   → AWS_S3_BUCKET_NOT_FOUND
            other upload ops     → AWS_S3_UPLOAD_FAILED / AWS_S3_DOWNLOAD_FAILED
            fallback             → AWS_CLIENT_ERROR
    """

    # Maps (operation_name_fragment, http_status) → ErrorCode
    _CLIENT_ERROR_MAP: dict[tuple[str, int], ErrorCode] = {
        ("upload",   403): ErrorCode.AWS_CREDENTIALS_MISSING,
        ("download", 403): ErrorCode.AWS_CREDENTIALS_MISSING,
        ("upload",   404): ErrorCode.AWS_S3_BUCKET_NOT_FOUND,
        ("download", 404): ErrorCode.AWS_S3_BUCKET_NOT_FOUND,
        ("upload",   500): ErrorCode.AWS_S3_UPLOAD_FAILED,
        ("download", 500): ErrorCode.AWS_S3_DOWNLOAD_FAILED,
    }

    def __init__(self, stage: Optional[str] = None) -> None:
        self.stage = stage

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # no exception — pass through

        # Lazy imports so this module has zero hard AWS dependency
        try:
            from botocore.exceptions import NoCredentialsError, ClientError
        except ImportError:
            return False  # botocore not installed; let exception propagate

        if isinstance(exc_val, NoCredentialsError):
            raise SentimentMLOpsException(
                "AWS credentials not found. Ensure IAM role or env vars are configured.",
                ErrorCode.AWS_CREDENTIALS_MISSING,
                stage=self.stage,
                original_exception=exc_val,
            ) from exc_val

        if isinstance(exc_val, ClientError):
            http_status  = exc_val.response["ResponseMetadata"]["HTTPStatusCode"]
            operation    = exc_val.operation_name.lower()          # e.g. "putobject"
            error_code_str = exc_val.response["Error"].get("Code", "")

            # Derive operation family (upload vs download) from operation name
            op_family = (
                "upload"   if any(k in operation for k in ("put", "upload", "copy")) else
                "download" if any(k in operation for k in ("get", "download", "head")) else
                "unknown"
            )

            # Special-case AccessDenied regardless of HTTP status
            if error_code_str in ("AccessDenied", "InvalidClientTokenId"):
                mapped_code = ErrorCode.AWS_CREDENTIALS_MISSING
            elif error_code_str == "NoSuchBucket":
                mapped_code = ErrorCode.AWS_S3_BUCKET_NOT_FOUND
            else:
                mapped_code = self._CLIENT_ERROR_MAP.get(
                    (op_family, http_status),
                    ErrorCode.AWS_CLIENT_ERROR,
                )

            raise SentimentMLOpsException(
                f"AWS ClientError [{error_code_str}] during '{operation}': {exc_val}",
                mapped_code,
                stage=self.stage,
                original_exception=exc_val,
            ) from exc_val

        return False  # non-AWS exception — let it propagate unchanged


# ──────────────────────────────────────────────
# 5. Sys.excepthook — catch unhandled exceptions
# ──────────────────────────────────────────────

def _global_exception_hook(exc_type, exc_value, exc_tb):
    """
    Replaces sys.excepthook so any *unhandled* exception that reaches the
    top of the call stack is formatted consistently.

    Callers that want to log unhandled exceptions should call
    `install_global_exception_hook()` once at application entry point
    (training_pipeline.py, app.py, etc.).

    KeyboardInterrupt is intentionally left to Python's default handler.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    # Print to stderr so it's visible even if the logger sink is down
    print(
        f"\n[UNHANDLED EXCEPTION]\n"
        f"  type      : {exc_type.__name__}\n"
        f"  message   : {exc_value}\n"
        f"  env       : {os.getenv('APP_ENV', 'dev')}\n"
        f"  host      : {socket.gethostname()}\n"
        f"  timestamp : {datetime.now(timezone.utc).isoformat()}\n"
        f"\n{formatted}",
        file=sys.stderr,
    )


def install_global_exception_hook() -> None:
    """
    Call once at the entry point of training_pipeline.py and app.py to
    ensure all unhandled exceptions are printed in a consistent format.

    Usage
    -----
        from src.utils.exception import install_global_exception_hook
        install_global_exception_hook()
    """
    sys.excepthook = _global_exception_hook


# ──────────────────────────────────────────────
# Public surface
# ──────────────────────────────────────────────

__all__ = [
    "SentimentMLOpsException",
    "ErrorCode",
    "aws_exception_handler",
    "install_global_exception_hook",
]