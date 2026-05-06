ARGS ?=

.PHONY: inspect-raw materialize validate visualize package stage export-coco train-dfine

inspect-raw:
	uv run python -m adp.data.inspect_raw $(ARGS)

materialize:
	uv run python -m adp.data.materialize $(ARGS)

validate:
	uv run python -m adp.data.validate $(ARGS)

visualize:
	uv run python -m adp.data.visualize $(ARGS)

package:
	uv run python -m adp.data.package $(ARGS)

stage:
	uv run python -m adp.data.stage $(ARGS)

export-coco:
	uv run python -m adp.data.export_coco $(ARGS)

train-dfine:
	uv run python -m adp.train.train_dfine $(ARGS)
