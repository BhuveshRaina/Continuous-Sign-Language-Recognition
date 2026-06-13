import csv
import os
import re

def normalize(s):
    if not s:
        return ""
    # Lowercase
    s = s.lower().strip()
    # Replace common variations
    s = s.replace("dont", "do not")
    s = s.replace("don't", "do not")
    s = s.replace("cannot", "can not")
    s = s.replace("iam", "i am")
    s = s.replace("alot", "a lot")
    s = s.replace("_", " ")
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra spaces
    s = " ".join(s.split())
    return s

# Read CSV manually
original_labels = set()
try:
    with open('metadata/video_metadata.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['sentence_label']:
                original_labels.add(row['sentence_label'])
except Exception as e:
    print(f"Error reading CSV: {e}")

# Read Processed Dirs
processed_path = 'data/processed_keypoints'
processed_dirs = [d for d in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, d))]

# Create comparison
mapping = {}
for original in original_labels:
    norm_orig = normalize(original)
    found = False
    for processed in processed_dirs:
        norm_proc = normalize(processed)
        if norm_orig == norm_proc:
            if processed not in mapping:
                mapping[processed] = []
            mapping[processed].append(original)
            found = True
            break
    if not found:
        if "UNMATCHED" not in mapping:
            mapping["UNMATCHED"] = []
        mapping["UNMATCHED"].append(original)

# Print results in a readable format
print("# Label Preprocessing & Merging Report")
print(f"\nTotal Unique Labels in Original Data (CSV): {len(original_labels)}")
print(f"Total Unique Classes in Processed Data: {len(processed_dirs)}")
print("\nThis report shows how multiple variations of the same sentence (differing by case, punctuation, or contractions) were merged into single canonical classes.")

print("\n## Examples of Merged Variations")

for processed, originals in sorted(mapping.items()):
    if processed == "UNMATCHED":
        continue
    # Show only interesting cases where merging actually happened
    if len(originals) > 1 or (len(originals) == 1 and processed != originals[0]):
        print(f"\n- **Canonical Class:** `{processed}`")
        print(f"  - Original variations found: " + ", ".join([f"`{o}`" for o in sorted(originals)]))

if "UNMATCHED" in mapping:
    print("\n## Unmatched Original Labels (Potential Issues)")
    print("These labels in the CSV did not match any processed directory after normalization.")
    for o in sorted(mapping["UNMATCHED"]):
        print(f"- `{o}`")
