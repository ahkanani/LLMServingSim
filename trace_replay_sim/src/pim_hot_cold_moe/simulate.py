"""Compatibility entry point for ``python -m pim_hot_cold_moe.simulate``."""

from __future__ import annotations

from .simulation import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
