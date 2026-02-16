
import datasets
import json

print("Loading dataset in streaming mode...")
ds = datasets.load_dataset("macwiatrak/bacbench-antibiotic-resistance-protein-sequences", split="train", streaming=True)

item = next(iter(ds))

with open("dataset_sample.json", "w") as f:
    # Convert to string or use json.dump if serializable, but some fields might be large lists
    # Let's just dump keys and a summary of values to avoid huge files if protein sequences are long
    summary = {}
    for k, v in item.items():
        if isinstance(v, list) and len(v) > 10:
             summary[k] = f"List of length {len(v)} (first item: {v[0]})"
        else:
             summary[k] = v
    json.dump(summary, f, indent=2)

print("Saved sample to dataset_sample.json")
