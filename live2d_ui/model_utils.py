import json
import traceback
from pathlib import Path

def get_fixed_model_path(original_path: Path) -> Path:
    try:
        fixed_path = original_path.with_stem(original_path.stem + "_fixed")
        if fixed_path.exists():
            return fixed_path

        with open(original_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.pop("DefaultExpression", None)
        if "FileReferences" in data:
            data["FileReferences"].pop("DefaultExpression", None)

        with open(fixed_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return fixed_path
    except Exception:
        traceback.print_exc()
        return original_path

