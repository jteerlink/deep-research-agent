"""Small stdlib HTTP helpers with secret-safe errors."""

from __future__ import annotations

import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .errors import ProviderError

SECRET_MARKERS = ("api_key", "x-api-key", "authorization", "bearer")


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    ca_bundle: str = "",
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    context = _ssl_context(ca_bundle)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:4000]
        raise ProviderError(f"HTTP {exc.code} from {url}: {_redact(body)}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"connection failed for {url}: {_redact(str(exc.reason))}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response from {url}: {_redact(body[:4000])}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError(f"unexpected response shape from {url}")
    return parsed


def _ssl_context(ca_bundle: str = "") -> ssl.SSLContext:
    bundle = ca_bundle or _macos_keychain_bundle()
    if bundle:
        return ssl.create_default_context(cafile=bundle)
    return ssl.create_default_context()


def _macos_keychain_bundle() -> str:
    if sys.platform != "darwin":
        return ""
    cache_dir = Path(".leadgen_cache")
    bundle = cache_dir / "macos-system-ca-bundle.pem"
    if bundle.exists() and bundle.stat().st_size > 0:
        return str(bundle)
    keychains = [
        "/System/Library/Keychains/SystemRootCertificates.keychain",
        "/Library/Keychains/System.keychain",
    ]
    try:
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p", *keychains],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if "BEGIN CERTIFICATE" not in result.stdout:
        return ""
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle.write_text(result.stdout, encoding="utf-8")
    return str(bundle)


def _redact(value: str) -> str:
    text = value
    for marker in SECRET_MARKERS:
        text = text.replace(marker, "<redacted>")
    return text
