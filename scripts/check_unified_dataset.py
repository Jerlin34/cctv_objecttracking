import argparse
import os
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_data_yaml(yaml_path: Path):
    names = {}
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key.isdigit():
                names[int(key)] = value
    return names


def yolo_row_is_valid(parts):
    if len(parts) != 5:
        return False
    try:
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        return False
    if class_id < 0:
        return False
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return False
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return False
    return True


def collect_stems(directory: Path, suffixes):
    if not directory.exists():
        return set()
    stems = set()
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in suffixes:
            stems.add(item.stem)
    return stems


def check_split(root: Path, split: str, class_count: int):
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    image_stems = collect_stems(images_dir, IMAGE_EXTS)
    label_stems = collect_stems(labels_dir, {".txt"})

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    empty_labels = []
    invalid_rows = []
    invalid_class_ids = []

    for label_file in sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []:
        text = label_file.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels.append(label_file.name)
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            parts = line.strip().split()
            if not yolo_row_is_valid(parts):
                invalid_rows.append(f"{label_file.name}:{line_no}")
                continue
            class_id = int(parts[0])
            if class_id >= class_count:
                invalid_class_ids.append(f"{label_file.name}:{line_no} -> {class_id}")

    return {
        "split": split,
        "images_count": len(image_stems),
        "labels_count": len(label_stems),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "empty_labels": empty_labels,
        "invalid_rows": invalid_rows,
        "invalid_class_ids": invalid_class_ids,
    }


def print_report(report):
    split = report["split"]
    print(f"\n[{split}]")
    print(f"images: {report['images_count']} | labels: {report['labels_count']}")
    print(f"missing labels: {len(report['missing_labels'])}")
    print(f"missing images: {len(report['missing_images'])}")
    print(f"empty labels: {len(report['empty_labels'])}")
    print(f"invalid rows: {len(report['invalid_rows'])}")
    print(f"invalid class ids: {len(report['invalid_class_ids'])}")

    def preview(title, values, n=8):
        if values:
            print(f"  {title}: {', '.join(values[:n])}" + (" ..." if len(values) > n else ""))

    preview("missing label files for image stems", report["missing_labels"])
    preview("missing image files for label stems", report["missing_images"])
    preview("empty labels", report["empty_labels"])
    preview("bad rows", report["invalid_rows"])
    preview("out-of-range class ids", report["invalid_class_ids"])


def main():
    parser = argparse.ArgumentParser(description="Validate unified YOLO dataset consistency.")
    parser.add_argument(
        "--data-root",
        default=os.path.join("datasets", "unified"),
        help="Dataset root containing images/ and labels/ folders.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing data.yaml: {yaml_path}")

    names = read_data_yaml(yaml_path)
    class_count = len(names)
    if class_count == 0:
        raise ValueError("No classes found in data.yaml names section.")

    print(f"Checking dataset: {root}")
    print(f"Class count: {class_count}")

    splits = ["train", "val", "test"]
    reports = [check_split(root, split, class_count) for split in splits]
    for report in reports:
        print_report(report)

    total_issues = sum(
        len(r["missing_labels"])
        + len(r["missing_images"])
        + len(r["empty_labels"])
        + len(r["invalid_rows"])
        + len(r["invalid_class_ids"])
        for r in reports
    )
    if total_issues == 0:
        print("\nDataset check passed. No consistency issues found.")
    else:
        print(f"\nDataset check found {total_issues} issue(s). Fix these before training.")


if __name__ == "__main__":
    main()
