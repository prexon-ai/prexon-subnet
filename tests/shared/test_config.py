import json
import re
from pathlib import Path, PurePosixPath

import yaml

from validator.config import ROOT, UPLOAD_POLICY_PATH, UPLOAD_STATE_PATH


def profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "engine_version": "0.5.23",
                "worker_image": "example/worker@sha256:" + "a" * 64,
                "hardware": {"gpu_type": "H100", "gpu_count": 1, "min_vram_gb": 80},
                "workloads": [
                    {
                        "name": "low",
                        "concurrency": 1,
                        "requests": 2,
                        "output_tokens": 8,
                        "weight": 1,
                        "prompts": ["Hello"],
                    }
                ],
            }
        )
    )
    return path


def test_compose_api_mounts_config_and_policy_read_only_and_gets_only_s3_secrets():
    # The validator runs on the host under pm2; only the upload API is a container.
    root = Path(__file__).resolve().parents[2]
    compose = yaml.safe_load((root / "docker/compose.yaml").read_text())
    assert set(compose["services"]) == {"api"}
    api = compose["services"]["api"]
    mounts = {}
    for volume in api["volumes"]:
        # Missing host paths must stop Compose, not appear as root-owned directories.
        assert volume["type"] == "bind" and volume["bind"]["create_host_path"] is False
        source = (root / "docker" / volume["source"]).resolve()
        mounts[volume["target"]] = (source, volume.get("read_only", False))
    assert mounts["/app/validator/config.py"] == (root / "validator/config.py", True)
    # config.py anchors paths to the checkout: the host checkout, or /app in the container.
    # A read-only directory mount keeps the validator's atomic replacement visible.
    policy = UPLOAD_POLICY_PATH.relative_to(ROOT)
    state = UPLOAD_STATE_PATH.relative_to(ROOT)
    assert mounts[str(PurePosixPath("/app") / policy.parent)] == (root / policy.parent, True)
    assert mounts[str(PurePosixPath("/app") / state.parent)] == (root / state.parent, False)
    assert set(api["environment"]) == {"S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"}
    assert api["ports"] == ["127.0.0.1:8080:8080"]
    # Docker reports the container unhealthy when /api/health stops answering.
    assert "http://127.0.0.1:8080/api/health" in " ".join(api["healthcheck"]["test"])
    # The validator writes the policy with mode 0640, so the API runs as its user.
    assert api["user"].startswith("${API_CONTAINER_USER:?")
    assert "PREXON_" not in (root / "docker/Dockerfile.api").read_text()


def test_no_leftover_references_to_removed_settings():
    # Names that no longer exist: the short-lived prexon.toml, the file chain, and the
    # settings machinery replaced by plain config modules.
    root = Path(__file__).resolve().parents[2]
    files = [root / ".env.example", root / "README.md", *root.glob("docker/*")]
    files += [*root.glob(".github/workflows/*")]
    for folder in ("api", "evaluation", "miner", "shared", "validator", "docs"):
        files += [p for p in (root / folder).rglob("*") if p.suffix in {".py", ".md"}]
    stale = re.compile(
        r"prexon(\.example)?\.toml|\[(chain|schedule|storage|rental|api)\] [a-z_]+"
        r'|\bCHAIN_FILE\b|\bchain_file\b|FileChain|chain\.json|CHAIN = "file"'
        r"|load_settings|settings_loader|load_miner_settings|read_environment"
        r"|S3_ENDPOINT_URL|S3_REGION|S3_SESSION_TOKEN|session_token"
        r"|\.env\.miner|moved to validator/config\.py|miner\.config\b|miner/config\.py"
        r"|shared\.config\b|shared/config\.py"
        r"|\bslots?\b|\bTARGETS\b|build_targets|--model-id|--competition-id|--slot\b"
        r"|examples/profile\.json|profiles/model-|prexon-miner targets"
        r"|competitions\.json|COMPETITIONS_FILE|examples/datasets\.json|competitions/datasets|category_weights"
    )
    leftovers = [
        str(path.relative_to(root))
        for path in files
        if "superpowers" not in path.parts and stale.search(path.read_text())
    ]
    assert leftovers == []
