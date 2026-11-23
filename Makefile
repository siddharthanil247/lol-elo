.PHONY: test lint report

test:
	uv run pytest

lint:
	uv run ruff check .

report:
	uv run python -m lol_elo report --season $(SEASON)
	open reports/$(SEASON).html
