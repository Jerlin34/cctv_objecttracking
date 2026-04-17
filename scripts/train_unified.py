import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune one unified YOLO model for objects + location classes."
    )
    parser.add_argument(
        "--data",
        default=os.path.join("datasets", "unified", "data.yaml"),
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Base model for fresh training OR checkpoint for fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device id (e.g., 0) or cpu",
    )
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument(
        "--project",
        default=os.path.join("runs", "unified"),
        help="Output project directory",
    )
    parser.add_argument("--name", default="exp1", help="Run name")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Use checkpoint in --model and run a shorter fine-tune cycle.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose trainer logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"data.yaml not found: {args.data}\n"
            f"Create it first (for example: datasets/unified/data.yaml)."
        )

    model = YOLO(args.model)

    # Fine-tune runs are typically shorter than fresh training.
    epochs = 40 if args.fine_tune and args.epochs == 150 else args.epochs
    run_name = f"{args.name}_ft" if args.fine_tune and not args.name.endswith("_ft") else args.name

    print("Starting training with:")
    print(f"  data     : {args.data}")
    print(f"  model    : {args.model}")
    print(f"  epochs   : {epochs}")
    print(f"  imgsz    : {args.imgsz}")
    print(f"  batch    : {args.batch}")
    print(f"  device   : {args.device}")
    print(f"  workers  : {args.workers}")
    print(f"  project  : {args.project}")
    print(f"  name     : {run_name}")
    print(f"  fine_tune: {args.fine_tune}")

    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        patience=args.patience,
        verbose=args.verbose,
    )

    print("\nTraining complete.")
    print(f"Check weights at: {os.path.join(args.project, run_name, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
