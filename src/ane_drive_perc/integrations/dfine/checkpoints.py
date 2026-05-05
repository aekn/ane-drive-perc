import urllib.request
from pathlib import Path


def ensure_local_checkpoint(
    *,
    url: str,
    local_path: str | Path,
) -> Path:
    output = Path(local_path)

    if output.exists() and output.stat().st_size > 0:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading checkpoint: {output}")
    urllib.request.urlretrieve(url, output)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(
            f"Checkpoint download failed or produced an empty file: {output}"
        )

    return output
