"""Read-only proof that Kaggle can authenticate to the target HF dataset."""
import json
from pathlib import Path

from kaggle_secrets import UserSecretsClient

REPO_ID = "TryDotAtwo/multigpubfs-bfs-results"


def main():
    output = Path("/kaggle/working/hf-auth-probe")
    output.mkdir()
    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_SECRET_HF_TOKEN_EMPTY")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("HUGGINGFACE_HUB_MISSING") from error
    api = HfApi(token=token)
    identity = api.whoami()
    info = api.dataset_info(REPO_ID)
    result = {
        "schema": 1,
        "status": "PASS",
        "dataset": REPO_ID,
        "dataset_id": info.id,
        "authenticated_name": identity.get("name"),
        "token_was_not_emitted": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
