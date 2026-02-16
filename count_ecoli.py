
import datasets
import tqdm

print("Loading dataset from local cache...")
# Load without streaming to use the cached version
ds = datasets.load_dataset("macwiatrak/bacbench-antibiotic-resistance-protein-sequences", split="train")

target_taxid = "562"
count = 0
total_scanned = 0

print(f"Counting genomes with taxid='{target_taxid}'...")
print(f"Total dataset size: {len(ds)}")

# Since it's loaded locally, we can iterate faster or filter directly
filtered_ds = ds.filter(lambda x: str(x["taxid"]) == target_taxid, num_proc=4)
count = len(filtered_ds)

print(f"\nFinished.")
print(f"Total genomes: {len(ds)}")
print(f"Total E. coli (taxid {target_taxid}): {count}")
print(f"Percentage: {count/len(ds)*100:.2f}%")
