from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TextIO

MetricKind = Literal["float", "int", "str"]
EventKind = Literal["note", "warn", "error", "debug"]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class JsonlLike(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class MetricColumn:
    key: str
    title: str
    precision: int = 4
    kind: MetricKind = "float"
    width: int = 8


class MeanMeter:
    def __init__(self) -> None:
        self.count = 0
        self.totals: defaultdict[str, float] = defaultdict(float)

    def update(self, metrics: dict[str, float], *, n: int = 1) -> None:
        self.count += n
        for key, value in metrics.items():
            self.totals[key] += float(value) * n

    def means(self) -> dict[str, float]:
        if self.count == 0:
            return {key: 0.0 for key in self.totals}
        return {key: value / self.count for key, value in self.totals.items()}


class Ansi:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    RED = "\x1b[31m"


class TrainingReporter:
    """Terminal renderer and JSONL event writer for training loops."""

    def __init__(
        self,
        *,
        writer: JsonlLike,
        columns: list[MetricColumn],
        progress_width: int = 12,
        stream: TextIO = sys.stdout,
        color: bool | None = None,
    ) -> None:
        self.writer = writer
        self.columns = columns
        self.progress_width = progress_width
        self.stream = stream
        self.use_color = should_use_color(stream) if color is None else color
        self.epoch_width = 7
        self.split_width = 6
        self.step_width = 9
        self.time_width = 6
        self._last_visible_line_len = 0

    def start_run(self, name: str, fields: dict[str, object]) -> None:
        field_text = " | ".join(f"{key}={value}" for key, value in fields.items())
        print(self._style(f"{name} | {field_text}", Ansi.BOLD), file=self.stream)
        self.print_header()
        self.writer.write(
            {
                "type": "run_start",
                "name": name,
                **{key: str(value) for key, value in fields.items()},
            }
        )

    def print_header(self) -> None:
        parts = [
            f"{'epoch':>{self.epoch_width}}",
            f"{'split':<{self.split_width}}",
            f"{'progress':>{self.progress_width + 2}}",
            f"{'step':>{self.step_width}}",
        ]
        parts.extend(f"{column.title:>{column.width}}" for column in self.columns)
        parts.append(f"{'time':>{self.time_width}}")
        print(file=self.stream)
        print(self._style(" ".join(parts), Ansi.DIM), file=self.stream)

    def begin_train_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        steps_in_epoch: int,
    ) -> TrainEpochReporter:
        return TrainEpochReporter(
            reporter=self,
            epoch=epoch,
            total_epochs=total_epochs,
            steps_in_epoch=steps_in_epoch,
        )

    def log_val_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        step: int,
        metrics: dict[str, float],
        elapsed_sec: float,
    ) -> None:
        self._end_live_line_if_needed()
        line = self._format_metric_row(
            epoch=epoch,
            total_epochs=total_epochs,
            split="val",
            progress_current=1,
            progress_total=1,
            step_text="-",
            metrics=metrics,
            elapsed_sec=elapsed_sec,
        )
        print(self._style(line, Ansi.GREEN), file=self.stream)
        self.writer.write(
            {
                "type": "epoch",
                "split": "val",
                "epoch": epoch,
                "step": step,
                "elapsed_sec": elapsed_sec,
                **metrics,
            }
        )

    def event(
        self,
        kind: EventKind,
        message: str,
        *,
        epoch: int | None = None,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._end_live_line_if_needed()
        epoch_text = "" if epoch is None else str(epoch)
        line = f"{epoch_text:>{self.epoch_width}} {kind:<{self.split_width}} {message}"
        print(self._style_event(kind, line), file=self.stream)
        record: dict[str, Any] = {"type": "event", "kind": kind, "message": message}
        if epoch is not None:
            record["epoch"] = epoch
        if step is not None:
            record["step"] = step
        if extra is not None:
            record["extra"] = extra
        self.writer.write(record)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        self.writer.write({"type": "summary", **metrics})

    def finish_run(
        self,
        *,
        first_loss: float,
        last_loss: float,
        best_loss: float,
        best_epoch: int,
        best_step: int,
        run_dir: object,
        checkpoint_path: object,
    ) -> None:
        self._end_live_line_if_needed()
        print(file=self.stream)
        print(
            self._style(
                f"done | loss {first_loss:.4f} -> {last_loss:.4f} | "
                f"best {best_loss:.4f} @ epoch {best_epoch}, step {best_step}",
                Ansi.BOLD,
            ),
            file=self.stream,
        )
        print(f"run: {run_dir}", file=self.stream)
        print(f"ckpt: {checkpoint_path}", file=self.stream)

    def _write_step_record(
        self,
        *,
        split: str,
        epoch: int,
        step: int,
        total_steps: int,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "type": "step",
            "split": split,
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            **metrics,
        }
        if extra is not None:
            record.update(extra)
        self.writer.write(record)

    def _finish_train_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        global_step: int,
        steps_in_epoch: int,
        metrics: dict[str, float],
        elapsed_sec: float,
    ) -> None:
        line = self._format_metric_row(
            epoch=epoch,
            total_epochs=total_epochs,
            split="train",
            progress_current=steps_in_epoch,
            progress_total=steps_in_epoch,
            step_text=f"{steps_in_epoch}/{steps_in_epoch}",
            metrics=metrics,
            elapsed_sec=elapsed_sec,
        )
        self._write_live_line(line)
        print(file=self.stream)
        self._last_visible_line_len = 0
        self.writer.write(
            {
                "type": "epoch",
                "split": "train",
                "epoch": epoch,
                "step": global_step,
                "elapsed_sec": elapsed_sec,
                **metrics,
            }
        )

    def _update_train_progress(
        self,
        *,
        epoch: int,
        total_epochs: int,
        step_in_epoch: int,
        steps_in_epoch: int,
        metrics: dict[str, float],
        elapsed_sec: float,
    ) -> None:
        line = self._format_metric_row(
            epoch=epoch,
            total_epochs=total_epochs,
            split="train",
            progress_current=step_in_epoch,
            progress_total=steps_in_epoch,
            step_text=f"{step_in_epoch}/{steps_in_epoch}",
            metrics=metrics,
            elapsed_sec=elapsed_sec,
        )
        self._write_live_line(line)

    def _format_metric_row(
        self,
        *,
        epoch: int,
        total_epochs: int,
        split: str,
        progress_current: int,
        progress_total: int,
        step_text: str,
        metrics: dict[str, float],
        elapsed_sec: float,
    ) -> str:
        epoch_text = f"{epoch}/{total_epochs}"
        progress_plain = make_progress_bar(
            progress_current, progress_total, width=self.progress_width
        )
        progress_text = self._style(
            f"{progress_plain:>{self.progress_width + 2}}", Ansi.CYAN
        )
        parts = [
            f"{epoch_text:>{self.epoch_width}}",
            f"{split:<{self.split_width}}",
            progress_text,
            f"{step_text:>{self.step_width}}",
        ]
        parts.extend(
            format_metric(column, metrics.get(column.key)) for column in self.columns
        )
        parts.append(f"{elapsed_sec:>{self.time_width - 1}.1f}s")
        return " ".join(parts)

    def _write_live_line(self, line: str) -> None:
        visible_len = visible_length(line)
        padding = " " * max(0, self._last_visible_line_len - visible_len)
        print(f"\r{line}{padding}", end="", file=self.stream, flush=True)
        self._last_visible_line_len = visible_len

    def _end_live_line_if_needed(self) -> None:
        if self._last_visible_line_len > 0:
            print(file=self.stream)
            self._last_visible_line_len = 0

    def _style(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"{code}{text}{Ansi.RESET}"

    def _style_event(self, kind: EventKind, line: str) -> str:
        if kind == "note":
            return self._style(line, Ansi.BLUE)
        if kind == "warn":
            return self._style(line, Ansi.YELLOW)
        if kind == "error":
            return self._style(line, Ansi.RED)
        if kind == "debug":
            return self._style(line, Ansi.DIM)
        return line


class TrainEpochReporter:
    def __init__(
        self,
        *,
        reporter: TrainingReporter,
        epoch: int,
        total_epochs: int,
        steps_in_epoch: int,
    ) -> None:
        if steps_in_epoch <= 0:
            raise ValueError(f"steps_in_epoch must be positive, got {steps_in_epoch}.")
        self.reporter = reporter
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.steps_in_epoch = steps_in_epoch
        self.step_in_epoch = 0
        self.start_time = time.perf_counter()
        self.meter = MeanMeter()

    def step(
        self,
        *,
        global_step: int,
        total_steps: int,
        metrics: dict[str, float],
        write_json: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.step_in_epoch += 1
        if self.step_in_epoch > self.steps_in_epoch:
            raise RuntimeError(
                f"Epoch {self.epoch} received too many steps: "
                f"{self.step_in_epoch}>{self.steps_in_epoch}."
            )

        self.meter.update(metrics)
        elapsed = time.perf_counter() - self.start_time
        self.reporter._update_train_progress(
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            step_in_epoch=self.step_in_epoch,
            steps_in_epoch=self.steps_in_epoch,
            metrics=metrics,
            elapsed_sec=elapsed,
        )
        if write_json:
            self.reporter._write_step_record(
                split="train",
                epoch=self.epoch,
                step=global_step,
                total_steps=total_steps,
                metrics=metrics,
                extra=extra,
            )

    def mean_metrics(self) -> dict[str, float]:
        return self.meter.means()

    def finish(
        self,
        *,
        global_step: int,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        elapsed = time.perf_counter() - self.start_time
        final_metrics = self.mean_metrics() if metrics is None else metrics
        self.reporter._finish_train_epoch(
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            global_step=global_step,
            steps_in_epoch=self.steps_in_epoch,
            metrics=final_metrics,
            elapsed_sec=elapsed,
        )
        return final_metrics


def make_progress_bar(current: int, total: int, *, width: int = 12) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = round(width * current / total)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def format_metric(column: MetricColumn, value: float | None) -> str:
    if value is None:
        return f"{'-':>{column.width}}"
    if column.kind == "int":
        return f"{int(round(value)):>{column.width}d}"
    if column.kind == "str":
        return f"{str(value):>{column.width}}"
    return f"{value:>{column.width}.{column.precision}f}"


def should_use_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    is_tty = getattr(stream, "isatty", lambda: False)
    return bool(is_tty())


def visible_length(text: str) -> int:
    return len(ANSI_RE.sub("", text))
