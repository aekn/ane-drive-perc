import argparse

from ane_drive_perc.integrations.dfine.prepare import prepare_dfine_teacher_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument("--install-requirements", action="store_true")
    p.add_argument("--skip-materialize", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--summary-dir", default=None)
    p.add_argument("--resume-from", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dfine_teacher_run(
        config_path=args.config,
        run=args.run,
        install_requirements=True if args.install_requirements else None,
        output_dir_override=args.output_dir,
        summary_dir_override=args.summary_dir,
        resume_from_override=args.resume_from,
        skip_materialize=args.skip_materialize,
    )
