import argparse

from ane_drive_perc.engine.overfit import run_tiny_overfit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="runs/overfit_tiny")
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--no-save-best", action="store_true")
    p.add_argument("--show-download-progress", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_tiny_overfit(
        config_path=args.config,
        split=args.split,
        num_images=args.num_images,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_best=not args.no_save_best,
        show_download_progress=args.show_download_progress,
    )
