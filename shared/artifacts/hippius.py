"""Hippius S3 connection requirements shared by both storage-owning roles."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

ENDPOINT = "https://s3.hippius.com"
REGION = "decentralized"


def hippius_environment(environment: Mapping[str, str]) -> dict[str, str]:
    values = dict(environment)
    endpoint = values.get("S3_ENDPOINT_URL") or ENDPOINT
    try:
        parsed = urlsplit(endpoint)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname == "s3.hippius.com"
            and parsed.port in {None, 443}
            and parsed.path in {"", "/"}
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Hippius storage requires S3_ENDPOINT_URL=https://s3.hippius.com")
    if (values.get("S3_REGION") or REGION) != REGION:
        raise ValueError("Hippius storage requires S3_REGION=decentralized")
    if not values.get("S3_ACCESS_KEY_ID") or not values.get("S3_SECRET_ACCESS_KEY"):
        raise ValueError("Hippius storage requires S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY")
    values["S3_ENDPOINT_URL"] = ENDPOINT
    values["S3_REGION"] = REGION
    return values
