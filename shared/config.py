"""Values more than one program uses."""

# Largest package upload: the API's limit, the miner's default and the validator's
# download cap (4 GiB of checkpoint plus 2 MiB of archive overhead).
UPLOAD_MAX_BYTES = 4 * 1024**3 + 2 * 1024**2
