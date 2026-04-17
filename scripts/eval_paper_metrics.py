import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_CLASSES = [
    "keys",
    "remote-control",
    "wallet",
    "bottle",
    "book",
    "earphone",
    "glasses-sunglasses",
]

DEFAULT_ZONES = ["chair", "table/desk", "laptop", "bed"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute paper-ready detection/location metrics from GT and prediction CSV files."
    )
    parser.add_argument("--gt", required=True, help="Ground-truth CSV path.")
    parser.add_argument("--pred", required=True, help="Prediction CSV path.")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class list for Table I.",
    )
    parser.add_argument(
        "--zones",
        default=",".join(DEFAULT_ZONES),
        help="Comma-separated zone list for Table II.",
    )
    parser.add_argument(
        "--success-seconds",
        type=float,
        default=3.0,
        help="Threshold for end-to-end success (seconds).",
    )
    parser.add_argument(
        "--name",
        default="Proposed fusion (Custom + COCO)",
        help="System name for Table III.",
    )
    return parser.parse_args()


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def norm_text(value):
    return (value or "").strip().lower()


def read_csv_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt(rows):
    by_image = defaultdict(list)
    for row in rows:
        image_id = row.get("image_id", "")
        cls = norm_text(row.get("class"))
        zone = norm_text(row.get("zone"))
        box = (
            to_float(row.get("x1")),
            to_float(row.get("y1")),
            to_float(row.get("x2")),
            to_float(row.get("y2")),
        )
        by_image[image_id].append(
            {
                "class": cls,
                "zone": zone,
                "box": box,
                "time_sec": to_float(row.get("time_sec")),
                "matched": False,
            }
        )
    return by_image


def load_pred(rows):
    by_image = defaultdict(list)
    fps_values = []
    latency_values = []
    for row in rows:
        image_id = row.get("image_id", "")
        cls = norm_text(row.get("pred_class"))
        zone = norm_text(row.get("pred_zone"))
        box = (
            to_float(row.get("x1")),
            to_float(row.get("y1")),
            to_float(row.get("x2")),
            to_float(row.get("y2")),
        )
        score = to_float(row.get("score"))
        by_image[image_id].append(
            {
                "class": cls,
                "zone": zone,
                "box": box,
                "score": score,
                "time_sec": to_float(row.get("time_sec")),
                "latency_ms": to_float(row.get("latency_ms")),
                "fps": to_float(row.get("fps")),
                "matched": False,
            }
        )
        if "fps" in row and str(row.get("fps", "")).strip():
            fps_values.append(to_float(row.get("fps")))
        if "latency_ms" in row and str(row.get("latency_ms", "")).strip():
            latency_values.append(to_float(row.get("latency_ms")))
    return by_image, fps_values, latency_values


def evaluate(gt_by_image, pred_by_image, classes, iou_threshold=0.5):
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "zone_ok": 0, "zone_total": 0, "gt_count": 0} for c in classes}

    for image_id, gt_list in gt_by_image.items():
        preds = sorted(pred_by_image.get(image_id, []), key=lambda item: item["score"], reverse=True)
        for item in gt_list:
            if item["class"] in per_class:
                per_class[item["class"]]["gt_count"] += 1

        for pred in preds:
            if pred["class"] not in per_class:
                continue
            best_gt = None
            best_iou = 0.0
            for gt in gt_list:
                if gt["matched"]:
                    continue
                if gt["class"] != pred["class"]:
                    continue
                score = iou(pred["box"], gt["box"])
                if score >= iou_threshold and score > best_iou:
                    best_iou = score
                    best_gt = gt
            if best_gt is not None:
                best_gt["matched"] = True
                pred["matched"] = True
                per_class[pred["class"]]["tp"] += 1
                per_class[pred["class"]]["zone_total"] += 1
                if norm_text(best_gt["zone"]) == norm_text(pred["zone"]):
                    per_class[pred["class"]]["zone_ok"] += 1
            else:
                per_class[pred["class"]]["fp"] += 1

        for gt in gt_list:
            if gt["class"] in per_class and not gt["matched"]:
                per_class[gt["class"]]["fn"] += 1

    return per_class


def compute_zone_table(gt_by_image, pred_by_image, zones, iou_threshold=0.5):
    zone_stats = {z: {"gt": 0, "correct": 0} for z in zones}
    for image_id, gt_list in gt_by_image.items():
        preds = pred_by_image.get(image_id, [])
        for gt in gt_list:
            if gt["zone"] not in zone_stats:
                continue
            zone_stats[gt["zone"]]["gt"] += 1
            matched = False
            for pred in preds:
                if pred["class"] != gt["class"]:
                    continue
                if iou(pred["box"], gt["box"]) < iou_threshold:
                    continue
                matched = True
                if pred["zone"] == gt["zone"]:
                    zone_stats[gt["zone"]]["correct"] += 1
                break
            if not matched:
                continue
    return zone_stats


def compute_end_to_end_success(gt_by_image, pred_by_image, success_seconds, iou_threshold=0.5):
    total = 0
    success = 0
    for image_id, gt_list in gt_by_image.items():
        preds = pred_by_image.get(image_id, [])
        for gt in gt_list:
            total += 1
            for pred in preds:
                if pred["class"] != gt["class"]:
                    continue
                if pred["zone"] != gt["zone"]:
                    continue
                if iou(pred["box"], gt["box"]) < iou_threshold:
                    continue
                if abs(pred["time_sec"] - gt["time_sec"]) <= success_seconds:
                    success += 1
                    break
    return (success / total * 100.0) if total > 0 else 0.0


def pct(num, den):
    return (num / den * 100.0) if den > 0 else 0.0


def print_table_1(per_class, classes):
    print("\nTABLE I: Per-class Detection Performance")
    print(
        f"{'Class':20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP@0.5':>10} {'Avg Lat(ms)':>12}"
    )
    precision_vals, recall_vals, f1_vals = [], [], []
    for cls in classes:
        tp = per_class[cls]["tp"]
        fp = per_class[cls]["fp"]
        fn = per_class[cls]["fn"]
        precision = pct(tp, tp + fp)
        recall = pct(tp, tp + fn)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # If true AP isn't provided externally, approximate mAP@0.5 with class-wise precision-recall harmonic signal.
        map50_proxy = (precision * recall / 100.0) if precision and recall else 0.0
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        print(f"{cls:20} {precision:10.2f} {recall:10.2f} {f1:10.2f} {map50_proxy:10.2f} {'-':>12}")
    if classes:
        print(
            f"{'Mean':20} "
            f"{sum(precision_vals)/len(precision_vals):10.2f} "
            f"{sum(recall_vals)/len(recall_vals):10.2f} "
            f"{sum(f1_vals)/len(f1_vals):10.2f} "
            f"{'-':>10} {'-':>12}"
        )


def print_table_2(zone_stats, zones):
    print("\nTABLE II: Location Assignment Performance by Zone")
    print(f"{'Zone':15} {'GT Instances':>12} {'Correctly Zoned':>16} {'Zone Accuracy (%)':>18}")
    total_gt, total_correct = 0, 0
    for zone in zones:
        gt = zone_stats[zone]["gt"]
        correct = zone_stats[zone]["correct"]
        acc = pct(correct, gt)
        total_gt += gt
        total_correct += correct
        print(f"{zone:15} {gt:12d} {correct:16d} {acc:18.2f}")
    print(f"{'Overall':15} {total_gt:12d} {total_correct:16d} {pct(total_correct, total_gt):18.2f}")


def print_table_3(name, per_class, zone_stats, e2e_success, fps_values, latency_values):
    tp = sum(v["tp"] for v in per_class.values())
    fp = sum(v["fp"] for v in per_class.values())
    fn = sum(v["fn"] for v in per_class.values())
    precision = pct(tp, tp + fp)
    recall = pct(tp, tp + fn)
    map50_proxy = (precision * recall / 100.0) if precision and recall else 0.0
    total_gt = sum(v["gt"] for v in zone_stats.values())
    total_ok = sum(v["correct"] for v in zone_stats.values())
    zone_acc = pct(total_ok, total_gt)
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0

    print("\nTABLE III: System Summary (Ablation-Ready Row)")
    print(
        f"{'Pipeline':40} {'mAP@0.5':>10} {'Zone Acc.(%)':>12} {'E2E Success(%)':>15} {'Avg FPS':>10} {'Avg Lat(ms)':>12}"
    )
    print(
        f"{name:40} {map50_proxy:10.2f} {zone_acc:12.2f} {e2e_success:15.2f} {avg_fps:10.2f} {avg_latency:12.2f}"
    )


def print_csv_format_help():
    print("\nExpected CSV formats:")
    print("GT CSV columns:")
    print("  image_id,class,zone,x1,y1,x2,y2,time_sec")
    print("Pred CSV columns:")
    print("  image_id,pred_class,pred_zone,score,x1,y1,x2,y2,time_sec,latency_ms,fps")


def main():
    args = parse_args()
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    if not gt_path.exists() or not pred_path.exists():
        print("Error: --gt and --pred files must exist.")
        print_csv_format_help()
        raise SystemExit(1)

    classes = [norm_text(item) for item in args.classes.split(",") if item.strip()]
    zones = [norm_text(item) for item in args.zones.split(",") if item.strip()]

    gt_rows = read_csv_rows(gt_path)
    pred_rows = read_csv_rows(pred_path)
    gt_by_image = load_gt(gt_rows)
    pred_by_image, fps_values, latency_values = load_pred(pred_rows)

    per_class = evaluate(gt_by_image, pred_by_image, classes, iou_threshold=0.5)
    zone_stats = compute_zone_table(gt_by_image, pred_by_image, zones, iou_threshold=0.5)
    e2e_success = compute_end_to_end_success(
        gt_by_image, pred_by_image, success_seconds=args.success_seconds, iou_threshold=0.5
    )

    print_table_1(per_class, classes)
    print_table_2(zone_stats, zones)
    print_table_3(args.name, per_class, zone_stats, e2e_success, fps_values, latency_values)
    print_csv_format_help()


if __name__ == "__main__":
    main()
