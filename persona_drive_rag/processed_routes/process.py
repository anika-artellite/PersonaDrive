import json
import glob
import os

# Place this script in the same folder as the JSON files
INPUT_DIR = "."

json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

for file_path in json_files:
    print(f"Processing: {file_path}")

    with open(file_path, "r") as f:
        records = json.load(f)

    for record in records:
        sentence = record.get("sentence", "")
        tp = record.get("target_point", [])
        tp_next = record.get("target_point_next", [])

        tp_str = f"[{tp[0]:.4f}, {tp[1]:.4f}]"
        tp_next_str = f"[{tp_next[0]:.4f}, {tp_next[1]:.4f}]"

        lines = sentence.split("\n")
        updated_lines = []
        already_added = False  # avoid duplicating if script is run twice

        for line in lines:
            # Remove old injection from Next Aim Waypoint line if present
            if line.startswith("Next Aim Waypoint:") and " with next two target points:" in line:
                line = line.split(" with next two target points:")[0]

            # Skip previously added Target Point lines to avoid duplicates
            if line.startswith("Target Point:") or line.startswith("Target Point Next:"):
                continue

            updated_lines.append(line)

            # Add target points right after Current Command
            if line.startswith("Current Command:") and not already_added:
                updated_lines.append(f"Target Point: {tp_str}")
                updated_lines.append(f"Target Point Next: {tp_next_str}")
                already_added = True

        record["sentence"] = "\n".join(updated_lines)

    with open(file_path, "w") as f:
        json.dump(records, f, indent=4)

    print(f"  Updated {len(records)} records.\n")

print("Done.")