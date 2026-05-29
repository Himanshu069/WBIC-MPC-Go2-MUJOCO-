# gait_scheduler.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

# ─────────────────────────────────────────────
# Leg ordering (used everywhere as the contract)
# ─────────────────────────────────────────────
# Index:  0    1    2    3
LEGS = ["FL", "FR", "RL", "RR"]


# ─────────────────────────────────────────────
# Gait pattern definition
# ─────────────────────────────────────────────
@dataclass
class GaitPattern:
    """
    Fully describes a periodic gait.
    Adding a new gait = subclass this (or just instantiate it directly).

    phase_offsets : (4,) float in [0, 1)
        When each leg's stance phase begins within the cycle.
        0.0 = starts at the very beginning of the cycle.
        0.5 = starts halfway through.
        Leg order follows LEGS = [FL, FR, RL, RR].

    stance_ratio : float in (0, 1)
        Fraction of one cycle each leg spends on the ground.

    period : float  [seconds]
        Duration of one full gait cycle.

    name : str
        Human-readable label (for logging / debugging).
    """
    name:          str
    phase_offsets: np.ndarray   # shape (4,)
    stance_ratio:  float        # 0 < φ_s < 1
    period:        float        # seconds

    def __post_init__(self):
        self.phase_offsets = np.asarray(self.phase_offsets, dtype=float)
        assert self.phase_offsets.shape == (4,), "Need one offset per leg"
        assert 0.0 < self.stance_ratio < 1.0
        assert self.period > 0.0


# ─────────────────────────────────────────────
# Built-in gaits
# ─────────────────────────────────────────────
class Gaits:
    """
    Factory for the standard quadruped gaits.
    All phase offsets follow the [FL, FR, RL, RR] convention.
    """

    @staticmethod
    def trot(period: float = 0.5, stance_ratio: float = 0.5) -> GaitPattern:
        """
        Diagonal pairs move together: (FL+RR) vs (FR+RL).
        Most common dynamic gait for quadrupeds.
        """
        return GaitPattern(
            name          = "trot",
            phase_offsets = np.array([0.0, 0.5, 0.5, 0.0]),  # FL, FR, RL, RR
            stance_ratio  = stance_ratio,
            period        = period,
        )

    @staticmethod
    def walk(period: float = 1.0, stance_ratio: float = 0.75) -> GaitPattern:
        """
        Slow static gait: one leg lifted at a time, 3-foot contact always.
        Sequence: FL → RR → FR → RL (each offset by T/4).
        """
        return GaitPattern(
            name          = "walk",
            phase_offsets = np.array([0.0, 0.5, 0.25, 0.75]),
            stance_ratio  = stance_ratio,
            period        = period,
        )

    @staticmethod
    def pace(period: float = 0.5, stance_ratio: float = 0.5) -> GaitPattern:
        """
        Lateral pairs move together: (FL+RL) vs (FR+RR).
        Like a camel. Less stable than trot on flat ground.
        """
        return GaitPattern(
            name          = "pace",
            phase_offsets = np.array([0.0, 0.5, 0.0, 0.5]),
            stance_ratio  = stance_ratio,
            period        = period,
        )

    @staticmethod
    def bound(period: float = 0.4, stance_ratio: float = 0.4) -> GaitPattern:
        """
        Front pair then rear pair: (FL+FR) vs (RL+RR).
        High-speed gait, requires more aggressive control.
        """
        return GaitPattern(
            name          = "bound",
            phase_offsets = np.array([0.0, 0.0, 0.5, 0.5]),
            stance_ratio  = stance_ratio,
            period        = period,
        )

    @staticmethod
    def pronk(period: float = 0.4, stance_ratio: float = 0.3) -> GaitPattern:
        """
        All four legs in phase — a jump/hop. Rarely used for locomotion
        but useful for testing symmetric force distribution.
        """
        return GaitPattern(
            name          = "pronk",
            phase_offsets = np.array([0.0, 0.0, 0.0, 0.0]),
            stance_ratio  = stance_ratio,
            period        = period,
        )

    @staticmethod
    def stand(period: float = 1.0) -> GaitPattern:
        """
        All feet in stance permanently. Stance ratio = 1 is not legal,
        so we use 0.999 — every leg will always read as 'stance'.
        """
        return GaitPattern(
            name          = "stand",
            phase_offsets = np.array([0.0, 0.0, 0.0, 0.0]),
            stance_ratio  = 0.999,
            period        = period,
        )

    @staticmethod
    def custom(
        phase_offsets: list[float],
        stance_ratio:  float,
        period:        float,
        name:          str = "custom",
    ) -> GaitPattern:
        """Escape hatch — define any gait by hand."""
        return GaitPattern(
            name          = name,
            phase_offsets = np.array(phase_offsets),
            stance_ratio  = stance_ratio,
            period        = period,
        )


# ─────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────
class GaitScheduler:
    """
    Tracks time and queries a GaitPattern for contact information.

    The scheduler is completely gait-agnostic — it never checks
    which gait is active. All logic lives in GaitPattern.

    Usage
    -----
    scheduler = GaitScheduler(Gaits.trot(), dt=0.02)

    # inside control loop:
    scheduler.step()                        # advance time by dt
    c  = scheduler.contact_state()          # (4,) int  → WBIC
    cs = scheduler.contact_schedule(N)      # (N, 4) int → MPC
    sp = scheduler.swing_phase("FL")        # float [0,1] → foot planner

    # switch gait mid-run (resets phase to zero):
    scheduler.switch_gait(Gaits.walk())
    """

    def __init__(self, gait: GaitPattern, dt: float = 0.02):
        self.gait = gait
        self.dt   = dt
        self._t   = 0.0          # continuous time [s]

    # ── time ──────────────────────────────────
    def step(self) -> None:
        """Advance the scheduler by one control timestep."""
        self._t += self.dt

    def reset(self) -> None:
        """Reset phase to zero (e.g. on robot initialisation)."""
        self._t = 0.0

    # ── gait switching ─────────────────────────
    def switch_gait(self, new_gait: GaitPattern, reset_phase: bool = True) -> None:
        """
        Hot-swap the active gait.
        reset_phase=True  → start new gait from phase 0 (cleaner transitions).
        reset_phase=False → keep current time (smoother if gaits share structure).
        """
        self.gait = new_gait
        if reset_phase:
            self._t = 0.0

    # ── internal helpers ──────────────────────
    def _leg_phase(self, leg_idx: int, t: float | None = None) -> float:
        """
        Normalised phase φ ∈ [0, 1) for leg `leg_idx` at time `t`.
        φ < stance_ratio  → stance
        φ >= stance_ratio → swing
        """
        t = self._t if t is None else t
        return (t / self.gait.period + self.gait.phase_offsets[leg_idx]) % 1.0

    def _is_stance(self, leg_idx: int, t: float | None = None) -> bool:
        return self._leg_phase(leg_idx, t) < self.gait.stance_ratio

    # ── public API ────────────────────────────
    def phase(self, leg: str) -> float:
        """Raw normalised phase ∈ [0, 1) for a leg. Useful for debugging."""
        return self._leg_phase(LEGS.index(leg))

    def contact_state(self) -> np.ndarray:
        """
        Current binary contact: 1 = stance, 0 = swing.
        Shape: (4,)  — ordered [FL, FR, RL, RR].
        Feed this to WBIC and the footstep planner.
        """
        return np.array(
            [int(self._is_stance(i)) for i in range(4)], dtype=int
        )

    def contact_schedule(self, N: int) -> np.ndarray:
        """
        Predicted contact over MPC horizon.
        Shape: (N, 4)  — rows = timesteps, cols = legs [FL, FR, RL, RR].
        Feed this to centroidal_mpc().
        """
        schedule = np.zeros((N, 4), dtype=int)
        for k in range(N):
            t_k = self._t + k * self.dt
            for i in range(4):
                schedule[k, i] = int(self._is_stance(i, t_k))
        return schedule

    def swing_phase(self, leg: str) -> float:
        """
        Normalised swing progress ∈ [0, 1] for a leg.
        0.0 when in stance. Reaches 1.0 at the end of swing.
        Feed this to the foot swing trajectory generator.
        """
        i   = LEGS.index(leg)
        phi = self._leg_phase(i)
        if phi < self.gait.stance_ratio:
            return 0.0
        return (phi - self.gait.stance_ratio) / (1.0 - self.gait.stance_ratio)

    def all_swing_phases(self) -> dict[str, float]:
        """Convenience: swing phase for all legs at once."""
        return {leg: self.swing_phase(leg) for leg in LEGS}

    # ── diagnostics ───────────────────────────
    def status(self) -> str:
        cs = self.contact_state()
        phases = [f"{l}:{self._leg_phase(i):.2f}" for i, l in enumerate(LEGS)]
        return (
            f"[{self.gait.name}] t={self._t:.3f}s  "
            f"contact={dict(zip(LEGS, cs))}  "
            f"phases=[{', '.join(phases)}]"
        )