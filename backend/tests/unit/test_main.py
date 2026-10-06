"""Tests for application-level setup in app.main."""

import logging
import time

from fastapi.testclient import TestClient

BOUNDARY = "XyZ"
JUNK_BYTES = 64 * 1024


def _multipart_with_trailing_junk() -> bytes:
    """A valid one-file form whose closing boundary is followed by junk.

    python-multipart 0.0.9 logged one WARNING per junk byte (CVE-2024-53981),
    and the form is parsed before authentication runs.
    """
    head = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
        "%PDF-1.4\r\n"
        f"--{BOUNDARY}--\r\n"
    ).encode()
    return head + b"A" * JUNK_BYTES


def _is_multipart_record(record: logging.LogRecord) -> bool:
    return record.name.split(".")[0] in ("multipart", "python_multipart")


class TestMultipartLogFlood:
    def test_multipart_loggers_only_report_errors(self):
        import app.main  # noqa: F401  (configures the loggers on import)

        for name in ("multipart", "python_multipart"):
            assert logging.getLogger(name).getEffectiveLevel() == logging.ERROR
        for name in ("multipart.multipart", "python_multipart.multipart"):
            assert not logging.getLogger(name).isEnabledFor(logging.WARNING)

    def test_trailing_junk_rejected_quickly_without_log_flood(
        self, client: TestClient, api_prefix: str, caplog
    ):
        caplog.set_level(logging.DEBUG)
        # caplog.set_level only lowers the root logger; the multipart
        # loggers keep the level app.main gave them.
        start = time.monotonic()
        response = client.post(
            f"{api_prefix}/documents/upload",
            content=_multipart_with_trailing_junk(),
            headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        )
        elapsed = time.monotonic() - start

        # Rejected by auth (no token) once the body is parsed.
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "MISSING_TOKEN"
        multipart_records = [r for r in caplog.records if _is_multipart_record(r)]
        assert multipart_records == []
        assert elapsed < 5
