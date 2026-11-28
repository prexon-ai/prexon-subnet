"""Lium REST requests for GPU rentals; job orchestration stays with each caller."""

from __future__ import annotations

import math
import re
from typing import Any

import httpx

from evaluation.config import (
    HF_TOKEN,
    LIUM_BASE_URL,
    LIUM_TEMPLATE_ID,
    LIUM_VOLUMES,
    SSH_PUBLIC_KEY,
)
from shared.targets import TargetConfig

# Pod states that cannot become a usable pod again (lower-cased Lium PodStatus values).
FAILED_POD_STATUSES = frozenset(
    {
        "deleted",
        "terminated",
        "failed",
        "stopped",
        "error",
        "creation_failed",
        "broken",
        "reboot_failed",
        "deleting",
    }
)


# Where a pod keeps its copy of the target on the competition's volume (mounted at /mnt).
TARGET_VOLUME = "/mnt/prexon-targets"


def worker_settings(volume: bool) -> dict[str, str]:
    """Environment for the pod's evaluation, sent on stdin: SSH sessions do not see the
    image's environment, and a command line would show the token."""
    settings = {"HF_XET_HIGH_PERFORMANCE": "1"}
    if HF_TOKEN:
        settings["HF_TOKEN"] = HF_TOKEN
    if volume:
        settings["PREXON_TARGET_VOLUME"] = TARGET_VOLUME
    return settings


class RentalRefused(RuntimeError):
    """The provider answered a rent request with a definite refusal: nothing was rented.

    Lium replays a refusal for a repeated Idempotency-Key and body, so a later attempt
    needs a new rental identity. Unknown outcomes are reported as ordinary errors.
    """

    def __init__(self, status: int, code: str = "") -> None:
        super().__init__(f"Lium refused the rental (HTTP {status}{', ' + code if code else ''})")
        self.status, self.code = status, code


class LiumClient:
    def __init__(
        self, api_key: str, max_cost: float, *, client: httpx.Client | None = None
    ) -> None:
        # The spend cap differs: the validator's sessions and prexon-evaluate rounds.
        self.max_cost = max_cost
        if not api_key:
            raise ValueError("LIUM_API_KEY is required for Lium provisioning")
        self.client = client or httpx.Client(
            base_url=LIUM_BASE_URL.rstrip("/") + "/",
            headers={"X-API-Key": api_key},
            timeout=30,
        )

    def build_request(self, session: dict[str, Any], target: TargetConfig) -> dict[str, Any]:
        template = target.lium_template_id or LIUM_TEMPLATE_ID
        if not template:
            raise ValueError("Configure a Lium template for the pinned worker image")
        response = self.client.get(f"templates/{template}")
        response.raise_for_status()
        data = response.json()
        reference, digest = target.worker_image.rsplit("@", 1)
        # Lium keeps the repository and the tag in separate fields.
        image, tag = reference, ""
        if ":" in reference.rsplit("/", 1)[-1]:
            image, tag = reference.rsplit(":", 1)
        template_image = (
            data.get("docker_image"),
            data.get("docker_image_tag") or "",
            data.get("docker_image_digest"),
        )
        if template_image != (image, tag, digest):
            raise ValueError("Lium template image does not match the immutable worker image")
        hours = max(1, math.ceil((session["deadline"] - session["started_at"]) / 3600))
        if hours > 720:
            raise ValueError("Lium termination_hours exceeds 720")
        hardware = target.hardware
        request: dict[str, Any] = {
            "pod_name": f"prexon-{session['id']}",
            "template_id": template,
            "user_public_key": SSH_PUBLIC_KEY.expanduser().read_text().strip(),
            "gpu_type": hardware.gpu_type,
            "gpu_count": hardware.gpu_count,
            "min_vram_gb": hardware.min_vram_gb,
            "min_cpus": hardware.min_cpus,
            "min_ram_gb": hardware.min_ram_gb,
            "min_disk_gb": hardware.min_disk_gb,
            "max_price_per_gpu_hour": min(
                hardware.max_price_per_gpu_hour,
                self.max_cost / (hardware.gpu_count * hours),
            ),
            "termination_hours": hours,
            "enable_jupyter": False,
            "enable_backups": False,
            "skip_add_agent_ssh_key": True,
        }
        if target.model_id in LIUM_VOLUMES:
            request["volume_id"] = LIUM_VOLUMES[target.model_id]
        return request

    def quote(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post("executors/rent-by-spec", json={**request, "dry_run": True})
        response.raise_for_status()
        data = dict(response.json())
        if data.get("success") is not True or data.get("dry_run") is not True:
            raise RuntimeError("Lium did not confirm a dry-run quote")
        return data

    def rent(self, session_id: str, request: dict[str, Any]) -> str:
        response = self.client.post(
            "executors/rent-by-spec",
            json=request,
            headers={"Idempotency-Key": f"prexon-{session_id}"},
        )
        # Idempotency conflicts mean an earlier call may still own or be creating a pod.
        if (
            400 <= response.status_code < 500
            and response.status_code != 408
            and "idempotency" not in response.text.lower()
        ):
            try:
                code = response.json().get("code")
            except (ValueError, AttributeError):
                code = None
            known = isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,64}", code)
            raise RentalRefused(response.status_code, code if known else "")
        response.raise_for_status()
        data = response.json()
        if data.get("success") is not True or not data.get("pod_id"):
            raise RuntimeError("Lium did not confirm a pod identity")
        return str(data["pod_id"])

    def status(self, pod_id: str) -> dict[str, Any]:
        response = self.client.get(f"pods/{pod_id}")
        if response.status_code == 404:
            return {"status": "deleted"}
        response.raise_for_status()
        return dict(response.json())

    def recover(self, session_id: str) -> str | None:
        response = self.client.get("pods")
        response.raise_for_status()
        data = response.json()
        pods = data if isinstance(data, list) else data.get("pods", [])
        matches = [p for p in pods if p.get("pod_name") == f"prexon-{session_id}"]
        if len(matches) > 1:
            raise RuntimeError("Multiple pods found for one rental identity")
        return str(matches[0]["id"]) if matches else None

    def delete(self, pod_id: str) -> None:
        response = self.client.delete(f"pods/{pod_id}")
        if response.status_code != 404:
            response.raise_for_status()
