"""Typer CLI: db init | bootstrap | trade [--paper] | feeds | report [daily|status|leadlag] | status | kill | canary."""

from __future__ import annotations

import typer

app = typer.Typer(help="sigcup: SIG Predictions Cup trading system", no_args_is_help=True)


@app.callback()
def _root() -> None:
    pass


if __name__ == "__main__":
    app()
