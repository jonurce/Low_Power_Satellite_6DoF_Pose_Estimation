# analyze_dataset.py

from pathlib import Path
from collections import Counter

# === CONFIG ===
DATASET_ROOT = "_dataset/animals"  # adjust if needed
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = [
    'cat', 'chicken', 'cow', 'dog', 'fox',
    'goat', 'horse', 'person', 'racoon', 'skunk'
]  # 10 classes

def count_classes_in_split(split_dir: Path):
    """Count number of images per class in one split (by reading label txt files)"""
    label_dir = split_dir / "labels"
    if not label_dir.exists():
        print(f"  → No labels folder found in {split_dir}")
        return Counter()

    class_counts = Counter()
    txt_files = list(label_dir.glob("*.txt"))

    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue  # empty label → background image (class 0 or none)
            # Read first class (or count all if multi-object, but usually 1)
            for line in lines:
                if line.strip():
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
                    break

    return class_counts


def main():
    print(f"\nAnalyzing dataset: {DATASET_ROOT}\n")
    print(f"{'Split':<10} {'Total images':<15} {'Class distribution':<60}")
    print("-" * 85)

    for split in SPLITS:
        split_dir = Path(DATASET_ROOT) / split
        if not split_dir.exists():
            print(f"{split:<10} Not found")
            continue

        img_dir = split_dir / "images"
        img_count = len(list(img_dir.glob("*.[jp][pn]g"))) if img_dir.exists() else 0

        class_counts = count_classes_in_split(split_dir)

        # Print summary
        print(f"{split:<10} {img_count:<15}", end="")
        if not class_counts:
            print("No labels or empty labels")
        else:
            dist_str = []
            for cls_id in sorted(class_counts.keys()):
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                count = class_counts[cls_id]
                dist_str.append(f"{name}: {count}")
            print(", ".join(dist_str))
        print()


if __name__ == "__main__":
    main()