from __future__ import annotations

import argparse

from ane_drive_perc.integrations.dfine.prepare import prepare_dfine_teacher_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--install-requirements", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dfine_teacher_run(
        config_path=args.config,
        run=args.run,
        install_requirements=True if args.install_requirements else None,
    )
