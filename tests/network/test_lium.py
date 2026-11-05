import os

import httpx
import pytest


@pytest.mark.network
def test_provider_advertises_required_rental_capabilities():
    if os.environ.get("PREXON_NETWORK_TESTS") != "1":
        pytest.skip("Set PREXON_NETWORK_TESTS=1 for read-only external integration checks")
    response = httpx.get("https://lium.io/api/version", timeout=30)
    response.raise_for_status()
    features = response.json()["features"]
    assert "rent_by_spec" in features
    assert "rent_idempotency" in features
