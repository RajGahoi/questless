# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adrian Diez Cuadrado
"""Tests for M13.12: Town treasury reserve target to prevent late-run wage deficits."""

import dataclasses

from questless.config import load_scenario
from questless.sim.ledger import Asset
from questless.sim.world import build_world
from tests.conftest import SCENARIOS


def test_reserve_days_config_defaults_to_zero(baseline):
    assert baseline.town.reserve_days == 0
    assert baseline.town.reserve_rate == 0.04


def test_reserve_target_keeps_treasury_above_floor_and_eliminates_unpaid_wages():
    base = load_scenario(SCENARIOS / "baseline.toml")
    # Enable reserve target for 5 days of wages
    tuned_town = dataclasses.replace(base.town, reserve_days=5, reserve_rate=0.04)
    cfg = dataclasses.replace(base, seed=3, days=2000, town=tuned_town)

    world = build_world(cfg)
    for _ in range(2000):
        world.step()

    unpaid_days = sum(1 for s in world.stats if s.wages_unpaid > 0)
    assert unpaid_days == 0, f"Expected 0 unpaid guard days, got {unpaid_days}"

    # Ensure town coin at end is at least above the configured floor
    guards = world.npcs_by_profession("guard")
    floor = 5 * len(guards) * cfg.town.guard_wage
    town_coin = world.ledger.balance("town", Asset.COIN)
    assert town_coin >= floor, f"Town coin {town_coin} below reserve floor {floor}"


def test_shifted_mine_disaster_with_reserve_target_has_no_unpaid_guard_days():
    mine = load_scenario(SCENARIOS / "mine_disaster.toml")
    tuned_town = dataclasses.replace(mine.town, reserve_days=5, reserve_rate=0.04)
    # Shift shock to day 1015
    shocks = [dataclasses.replace(s, start=1015) for s in mine.shocks]
    mine_shifted = dataclasses.replace(
        mine, seed=3, days=1200, shocks=tuple(shocks), town=tuned_town
    )

    world = build_world(mine_shifted)
    for _ in range(1200):
        world.step()

    unpaid_days = sum(1 for s in world.stats if s.wages_unpaid > 0)
    assert unpaid_days == 0, f"Expected 0 unpaid guard days, got {unpaid_days}"
