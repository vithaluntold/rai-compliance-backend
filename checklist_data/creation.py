import json
import os
from pathlib import Path

# Define the base directory containing the frameworks
base_dir = Path(__file__).parent / "frameworks"

frameworks = []

for framework_folder in sorted(os.listdir(base_dir)):
    framework_path = base_dir / framework_folder
    if not framework_path.is_dir():
        continue
    standards = []
    for filename in sorted(os.listdir(framework_path)):
        if filename.endswith(".json"):
            std_id = filename.replace(
                ".json", ""
            )  # Use the filename as ID (with spaces)
            std_name = filename.replace(".json", "")
            standards.append(
                {
                    "id": std_id,
                    "name": std_name,
                    "description": f"{std_name} checklist",
                    "available": True,
                }
            )
    frameworks.append(
        {
            "id": framework_folder,  # Use the folder name as ID (with spaces if any)
            "name": framework_folder,
            "description": f"{framework_folder} standards.",
            "standards": standards,
        }
    )

frameworks_json = {"version": "1.0", "frameworks": frameworks}

# Write to frameworks.json
frameworks_json_path = Path(__file__).parent / "frameworks.json"
with open(frameworks_json_path, "w", encoding="utf-8") as f:
    json.dump(frameworks_json, f, indent=2, ensure_ascii=False)

print(f"frameworks.json updated with {len(frameworks)} frameworks.")
