#!/usr/bin/env bash
# Repack a Docker image into one layer, keeping its environment, command, working directory and
# labels. Lium re-packs any image of 30 or more layers into one on the machine before starting
# it (Sysbox cannot start deeper images), which needs room for a full extra copy that a one-GPU
# pod's disk leaves none of; SGLang images have about 70 layers.
#
#   docker/flatten-image.sh SOURCE_IMAGE TARGET_IMAGE
set -euo pipefail
source_image=${1:?usage: flatten-image.sh SOURCE_IMAGE TARGET_IMAGE}
target_image=${2:?usage: flatten-image.sh SOURCE_IMAGE TARGET_IMAGE}

# `docker import` keeps only the files, so the settings are restated as Dockerfile lines.
changes=()
while IFS= read -r line; do
  changes+=(--change "$line")
done < <(docker image inspect --format '{{json .Config}}' "$source_image" | python3 -c '
import json, sys

config = json.load(sys.stdin)


def word(value):
    return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"").replace("$", "\\$") + "\""


for entry in config.get("Env") or []:
    key, _, value = entry.partition("=")
    print(f"ENV {key}={word(value)}")
if config.get("Entrypoint"):
    print("ENTRYPOINT " + json.dumps(config["Entrypoint"]))
if config.get("Cmd"):
    print("CMD " + json.dumps(config["Cmd"]))
if config.get("WorkingDir"):
    print("WORKDIR " + word(config["WorkingDir"]))
if config.get("User"):
    print("USER " + word(config["User"]))
for port in sorted(config.get("ExposedPorts") or {}):
    print(f"EXPOSE {port}")
for key, value in sorted((config.get("Labels") or {}).items()):
    print(f"LABEL {key}={word(value)}")
')

container=$(docker create "$source_image")
trap 'docker rm -f "$container" >/dev/null' EXIT
docker export "$container" | docker import "${changes[@]}" - "$target_image"
