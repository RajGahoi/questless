# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adrian Diez Cuadrado
"""World state, its construction from a scenario, and the tick loop."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

from questless.config import ScenarioConfig
from questless.contracts.engine import ContractBoard, ContractTickReport
from questless.contracts.needs import NeedTracker
from questless.contracts.validation import check_board
from questless.sim.entities import (
    ESCROW,
    IN_TRANSIT,
    MARKET,
    NPC,
    OUTSIDE,
    PLAYER,
    TOWN,
    Guild,
    Profession,
    Site,
    Traits,
)
from questless.sim.guilds import GuildReport, collect_dues, provide_relief
from questless.sim.guilds import close_reports as close_guild_reports
from questless.sim.guilds import reports as guild_reports
from questless.sim.ledger import Asset, Ledger
from questless.sim.market import MarketReport
from questless.sim.market import clear as clear_market
from questless.sim.names import make_name
from questless.sim.rng import RngStreams
from questless.sim.shocks import Modifiers, Shock
from questless.sim.transport import TransportReport, ship


@dataclass
class TickStats:
    """World-level totals for one tick; the raw material for metrics and provenance."""

    tick: int
    produced: Counter[Asset] = field(default_factory=Counter)
    consumed: Counter[Asset] = field(default_factory=Counter)
    unmet: Counter[Asset] = field(default_factory=Counter)
    produced_by_site: dict[Site, Counter[Asset]] = field(default_factory=dict)
    idle_npcs: int = 0
    hungry_npcs: int = 0
    # Per profession: "count", "idle", "hungry", "broken" (worked without usable equipment).
    by_profession: dict[Profession, Counter[str]] = field(default_factory=dict)
    upkeep: int = 0
    wages_paid: int = 0
    wages_unpaid: int = 0
    market: dict[Asset, MarketReport] = field(default_factory=dict)
    transport: TransportReport | None = None
    prices: dict[Asset, float] = field(default_factory=dict)  # after today's adjustment
    guilds: dict[str, GuildReport] = field(default_factory=dict)
    contracts: ContractTickReport | None = None


@dataclass
class World:
    config: ScenarioConfig
    ledger: Ledger
    rng: RngStreams
    npcs: dict[str, NPC]
    guilds: dict[str, Guild] = field(default_factory=dict)
    prices: dict[Asset, float] = field(default_factory=dict)  # settle at unit_price()
    modifiers: Modifiers = field(default_factory=Modifiers)
    stats: list[TickStats] = field(default_factory=list)
    needs: NeedTracker = field(default_factory=NeedTracker)
    contracts: ContractBoard = field(default_factory=ContractBoard)
    actions: list[dict] = field(default_factory=list)  # player action log, for replay
    pending_caravans: list[tuple[int, str, int]] = field(default_factory=list)
    tick: int = 0
    check_every_tick: bool = True  # conservation check per tick; cheap at prototype scale

    def step(self, days: int = 1) -> None:
        for _ in range(days):
            self._tick()

    def _tick(self) -> None:
        from questless.player import receive_caravans  # player imports World for typing
        from questless.sim.behaviour import run_npc_day

        stats = TickStats(tick=self.tick)
        start_coin = {
            n.id: self.ledger.balance(n.account, Asset.COIN) for n in self.npcs_in_order()
        }
        for npc in self.npcs_in_order():
            rec = run_npc_day(self, npc)
            stats.produced.update(rec.produced)
            stats.consumed.update(rec.consumed)
            stats.unmet.update(rec.unmet)
            stats.produced_by_site.setdefault(npc.site, Counter()).update(rec.produced)
            hungry = npc.hunger > self.config.needs.hungry_after
            stats.idle_npcs += rec.idle
            stats.hungry_npcs += hungry
            prof = stats.by_profession.setdefault(npc.profession, Counter())
            prof["count"] += 1
            prof["idle"] += rec.idle
            prof["hungry"] += hungry
            equipment = self.config.professions[npc.profession].equipment
            prof["broken"] += equipment is not None and rec.unmet.get(equipment, 0) > 0
        stats.transport = ship(self)
        guild_upkeep = self._collect_upkeep(stats)
        self._pay_wages(stats)
        relief = provide_relief(self)  # before the market, so relief buys food today
        stats.market = clear_market(self, stats.produced, stats.consumed)
        stats.prices = dict(self.prices)
        dues = collect_dues(self, start_coin, relief)
        stats.guilds = guild_reports(self, dues, relief, guild_upkeep)
        self.stats.append(stats)
        self.needs.update(self)
        stats.contracts = self.contracts.step(self)
        stats.guilds = close_guild_reports(self, stats.guilds)  # escrow moved after dues
        self.tick += 1
        # Caravans due on the new day unload now, so the player can use them that day.
        receive_caravans(self)
        if self.check_every_tick:
            self.check_invariants()

    def _collect_upkeep(self, stats: TickStats) -> dict[str, int]:
        """Town levy on idle coin. Closes the money loop: coin that pools with one profession
        flows back to the town, which pays guards, who buy food and weapons. Guild treasuries
        above their ``max_treasury`` pay the same rate on the excess, so dues can't pool there
        forever. Returns the levy paid by each guild."""
        town = self.config.town
        rate = town.upkeep_rate
        # Emergency reserve target: if town coin falls below reserve_days * daily_wages,
        # apply emergency reserve_rate (M13.12)
        if town.reserve_days > 0:
            guards = self.npcs_by_profession(Profession.GUARD)
            daily_wages = len(guards) * town.guard_wage
            if self.ledger.balance(TOWN, Asset.COIN) < town.reserve_days * daily_wages:
                rate = town.reserve_rate

        for npc in self.npcs_in_order():
            levy = town.levy(self.ledger.balance(npc.account, Asset.COIN), rate_override=rate)
            self.ledger.transfer(
                npc.account, TOWN, Asset.COIN, levy, tick=self.tick, reason="upkeep"
            )
            stats.upkeep += levy
        paid: dict[str, int] = {}
        for gid, guild in sorted(self.guilds.items()):
            cap = guild.policy.max_treasury
            if cap is None:
                continue
            levy = int(max(self.ledger.balance(guild.account, Asset.COIN) - cap, 0) * rate)
            self.ledger.transfer(
                guild.account, TOWN, Asset.COIN, levy, tick=self.tick, reason="guild upkeep"
            )
            paid[gid] = levy
            stats.upkeep += levy
        return paid

    def _pay_wages(self, stats: TickStats) -> None:
        wage = self.config.town.guard_wage
        guards = self.npcs_by_profession(Profession.GUARD)
        if self.ledger.balance(TOWN, Asset.COIN) < wage * len(guards):
            self.rng.stream("wages").shuffle(guards)
        for guard in guards:
            if self.ledger.balance(TOWN, Asset.COIN) >= wage:
                self.ledger.transfer(
                    TOWN, guard.account, Asset.COIN, wage, tick=self.tick, reason="guard wage"
                )
                stats.wages_paid += wage
            else:
                stats.wages_unpaid += wage

    def unit_price(self, good: Asset) -> int:
        """Integer coin price per unit at which trades settle today."""
        return max(1, round(self.prices[good]))

    def npcs_in_order(self) -> list[NPC]:
        return [self.npcs[k] for k in sorted(self.npcs)]

    def npcs_by_profession(self, profession: Profession) -> list[NPC]:
        return [n for n in self.npcs_in_order() if n.profession is profession]

    def check_invariants(self) -> None:
        self.ledger.check_conservation()
        check_board(self, self.contracts)
        if self.ledger.holdings(MARKET):
            raise AssertionError(
                f"market clearing account not empty: {self.ledger.holdings(MARKET)}"
            )

    def state_dict(self) -> dict:
        """Everything that defines the simulation state, in a deterministic JSON-friendly form."""
        return {
            "tick": self.tick,
            "prices": {str(a): round(p, 9) for a, p in sorted(self.prices.items())},
            "ledger": self.ledger.snapshot(),
            "npcs": {
                n.id: {
                    "guild": n.guild_id,
                    "tool_condition": round(n.tool_condition, 9),
                    "hunger": n.hunger,
                    "state": str(n.state),
                    "carry": dict(sorted(n.carry.items())),
                    "unmet_streak": {str(a): d for a, d in sorted(n.unmet_streak.items())},
                }
                for n in self.npcs_in_order()
            },
            "shocks": self.modifiers.state(),
            "contracts": {
                c.id: [str(c.status), c.reward, c.qty, sorted(c.beneficiaries.items()), c.funder]
                for c in self.contracts.contracts.values()
            },
            "cooldowns": sorted([list(k), v] for k, v in self.contracts.cooldowns.items()),
            "needs": sorted(
                [s.subject, str(s.good), s.days, s.calm_days] for s in self.needs.states.values()
            ),
            "reputation": {g.id: dict(sorted(g.reputation.items())) for g in self.guilds.values()},
            "caravans": sorted(self.pending_caravans),
            "actions": len(self.actions),
            "rng": repr(self.rng.getstate()),
        }

    def state_hash(self) -> str:
        blob = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def build_world(config: ScenarioConfig) -> World:
    """Create the genesis world. Same config (incl. seed) -> identical world, every time."""
    rng = RngStreams(config.seed)
    ledger = Ledger()
    npcs: dict[str, NPC] = {}
    taken_names: set[str] = set()
    trait_rng = rng.stream("genesis.traits")
    name_rng = rng.stream("genesis.names")
    t = config.traits

    index = 0
    for profession in Profession:  # enum order is fixed, so ids are stable
        pc = config.professions[profession]
        for _ in range(pc.count):
            npc_id = f"npc_{index:03d}"
            index += 1
            traits = Traits(
                skill=round(trait_rng.uniform(*t.skill), 3),
                thrift=round(trait_rng.uniform(*t.thrift), 3),
                stock_days=trait_rng.randint(*t.stock_days),
                patience=trait_rng.randint(*t.patience),
            )
            npc = NPC(
                id=npc_id,
                name=make_name(name_rng, str(profession), taken_names),
                profession=profession,
                site=pc.site,
                traits=traits,
                tool_condition=round(trait_rng.uniform(*pc.tool_condition), 3),
            )
            wallet = trait_rng.randint(*pc.wallet)
            ledger.open_account(npc.account, {Asset.COIN: wallet, **pc.inventory})
            if npc.site_account != npc.account:
                ledger.open_account(npc.site_account)
            npcs[npc_id] = npc

    guilds: dict[str, Guild] = {}
    if config.guilds_enabled:
        for gid, gc in config.guilds.items():
            members = sorted(n.id for n in npcs.values() if n.profession in gc.professions)
            guild = Guild(
                id=gid, name=gc.name, policy=gc.policy, dues_rate=gc.dues_rate, members=members
            )
            for member in members:
                npcs[member].guild_id = gid
            ledger.open_account(guild.account, {Asset.COIN: gc.treasury})
            guilds[gid] = guild

    ledger.open_account(TOWN, {Asset.COIN: config.town.treasury})
    ledger.open_account(PLAYER, {Asset.COIN: config.player_wallet})
    ledger.open_account(ESCROW)
    ledger.open_account(MARKET)
    ledger.open_account(IN_TRANSIT)
    ledger.open_account(OUTSIDE, dict(config.outside))
    ledger.seal()

    world = World(
        config=config,
        ledger=ledger,
        rng=rng,
        npcs=npcs,
        guilds=guilds,
        prices={asset: float(gc.base_price) for asset, gc in config.goods.items()},
        modifiers=Modifiers([Shock.from_config(s) for s in config.shocks]),
    )
    world.check_invariants()
    return world
