"""Verification: scripted scenarios on an independent plant, with pass / fail criteria."""

from .run import run_scenario, verify
from .scenarios import SCENARIOS

__all__ = ["SCENARIOS", "run_scenario", "verify"]
