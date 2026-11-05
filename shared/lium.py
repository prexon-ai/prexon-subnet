"""Shared Lium REST requests; job orchestration stays in each role."""

from __future__ import annotations

import math
from typing import Any

import httpx

from shared.config import Settings, TargetConfig


class LiumClient:
    def __init__(
        self, settings: Settings, api_key: str, *, client: httpx.Client | None = None
    ) -> None:
        self.settings = settings
        if not api_key:
            raise ValueError("LIUM_API_KEY is required for Lium provisioning")
        self.client = client or httpx.Client(
            base_url=settings.lium_base_url.rstrip("/") + "/",
            headers={"X-API-Key": api_key},
            timeout=30,
        )

    def build_request(self, session: dict[str, Any], target: TargetConfig) -> dict[str, Any]:
        template = target.lium_template_id or self.settings.lium_template_id
        if not template:
            raise ValueError("Configure a Lium template for the pinned worker image")
        response = self.client.get(f"templates/{template}")
        response.raise_for_status()
        data = response.json()
        image, digest = target.worker_image.rsplit("@", 1)
        if data.get("docker_image") != image or data.get("docker_image_digest") != digest:
            raise ValueError("Lium template image does not match the immutable worker image")
        hours = max(1, math.ceil((session["deadline"] - session["started_at"]) / 3600))
        if hours > 720:
            raise ValueError("Lium termination_hours exceeds 720")
        hardware = target.hardware
        request: dict[str, Any] = {
            "pod_name": f"prexon-{session['id']}",
            "template_id": template,
            "user_public_key": self.settings.ssh_public_key.expanduser().read_text().strip(),
            "gpu_type": hardware.gpu_type,
            "gpu_count": hardware.gpu_count,
            "min_vram_gb": hardware.min_vram_gb,
            "min_cpus": hardware.min_cpus,
            "min_ram_gb": hardware.min_ram_gb,
            "min_disk_gb": hardware.min_disk_gb,
            "max_price_per_gpu_hour": min(
                hardware.max_price_per_gpu_hour,
                self.settings.session_max_cost / (hardware.gpu_count * hours),
            ),
            "termination_hours": hours,
            "enable_jupyter": False,
            "enable_backups": False,
            "skip_add_agent_ssh_key": True,
        }
        if self.settings.lium_volume_id:
            request["volume_id"] = self.settings.lium_volume_id
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
