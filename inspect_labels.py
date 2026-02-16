
import datasets
from collections import Counter
from config import CONFIG
from dataset import _flatten_strings
from tqdm import tqdm

def inspect_labels():
    print("Loading dataset...")
    ds = datasets.load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])
    
    species_col = CONFIG.get("species_column", "taxid")
    target = CONFIG["species_filter"]
    
    if target:
        print(f"Filtering for {species_col} == {target}")
        ds = ds.filter(lambda x: str(x[species_col]) == target, num_proc=4)
    
    # Take a small subset to be fast and avoid huge logs
    ds = ds.select(range(min(len(ds), 50)))
    print(f"Subset dataset size: {len(ds)}")
    
    label_col = CONFIG["label_column"]
    product_counter = Counter()
    
    print("Scanning labels...")
    for row in tqdm(ds, desc="Scanning"):
        p_list = row[label_col]
        
        for p in _flatten_strings(p_list):
            if p:
                product_counter[p.strip()] += 1
                        
    print("\nTop 20 labels found:")
    for label, count in product_counter.most_common(20):
        print(f"'{label}': {count}")
        
    top_n = CONFIG.get("top_n_classes", 15)
    most_common = product_counter.most_common(top_n)
    label_map = {label: i for i, (label, _) in enumerate(most_common)}
    
    print(f"\nLabel Map (size {len(label_map)}):")
    print(label_map)
    
    # Check coverage
    total_labels = sum(product_counter.values())
    covered_labels = sum(count for label, count in product_counter.items() if label in label_map)
    
    print(f"\nTotal labels: {total_labels}")
    print(f"Covered labels: {covered_labels}")
    print(f"Coverage: {covered_labels/total_labels:.2%}")

if __name__ == "__main__":
    inspect_labels()
