"""ctypes wrapper for the vectorized Gen 1 Random Battle env (env/ Zig library).

Buffer layout must match env/src/main.zig (obs layout v1). All arrays are
zero-copy views into env-owned memory: read obs/masks after step(), write
actions before step().

Smoke test / benchmark:
    python3 python/gen1env.py teams/gen1-pool.bin [n_envs=64] [episodes=2000]
"""

import ctypes
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
_SUFFIX = "dylib" if sys.platform == "darwin" else "so"
LIB = ROOT / "env" / "zig-out" / "lib" / f"libgen1env.{_SUFFIX}"

N_ACTIONS = 10
ACTION_NAMES = ["move 1", "move 2", "move 3", "move 4",
                "switch 2", "switch 3", "switch 4", "switch 5", "switch 6", "move 0"]


class _Buffers(ctypes.Structure):
    _fields_ = [
        ("actions", ctypes.POINTER(ctypes.c_int32)),
        ("masks", ctypes.POINTER(ctypes.c_uint8)),
        ("needs", ctypes.POINTER(ctypes.c_uint8)),
        ("ints", ctypes.POINTER(ctypes.c_int32)),
        ("floats", ctypes.POINTER(ctypes.c_float)),
        ("m_ints", ctypes.POINTER(ctypes.c_int32)),
        ("m_floats", ctypes.POINTER(ctypes.c_float)),
        ("rewards", ctypes.POINTER(ctypes.c_float)),
        ("dones", ctypes.POINTER(ctypes.c_uint8)),
        ("ep_turns", ctypes.POINTER(ctypes.c_uint16)),
        ("ep_idx", ctypes.POINTER(ctypes.c_uint64)),
        ("n", ctypes.c_uint32),
        ("n_actions", ctypes.c_uint32),
        ("ints_per_player", ctypes.c_uint32),
        ("floats_per_player", ctypes.c_uint32),
    ]


def _load():
    lib = ctypes.CDLL(str(LIB))
    lib.g1_version.restype = ctypes.c_uint32
    lib.g1_create.restype = ctypes.c_void_p
    lib.g1_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint64]
    lib.g1_buffers.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Buffers)]
    lib.g1_step.argtypes = [ctypes.c_void_p]
    lib.g1_episodes.restype = ctypes.c_uint64
    lib.g1_episodes.argtypes = [ctypes.c_void_p]
    lib.g1_debug_stats.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint16)]
    lib.g1_destroy.argtypes = [ctypes.c_void_p]
    return lib


class Gen1Env:
    """n parallel gen1randombattle battles; 2 players per battle (self-play view)."""

    def __init__(self, pool: str, n: int = 64, seed: int = 0):
        self._lib = _load()
        assert self._lib.g1_version() == 2, "env version mismatch"
        self._h = self._lib.g1_create(str(pool).encode(), n, seed)
        if not self._h:
            raise RuntimeError(f"g1_create failed (pool={pool})")
        b = _Buffers()
        self._lib.g1_buffers(self._h, ctypes.byref(b))
        self.n = b.n
        self.n_actions = b.n_actions
        view = np.ctypeslib.as_array
        self.actions = view(b.actions, (n, 2))
        self.masks = view(b.masks, (n, 2, b.n_actions))
        self.needs = view(b.needs, (n, 2))
        self.ints = view(b.ints, (n, 2, b.ints_per_player))
        self.floats = view(b.floats, (n, 2, b.floats_per_player))
        self.m_ints = view(b.m_ints, (n, 2, b.ints_per_player))
        self.m_floats = view(b.m_floats, (n, 2, b.floats_per_player))
        self.rewards = view(b.rewards, (n, 2))
        self.dones = view(b.dones, (n,))
        self.ep_turns = view(b.ep_turns, (n,))
        self.ep_idx = view(b.ep_idx, (n,))

    def step(self):
        self._lib.g1_step(self._h)

    @property
    def episodes(self) -> int:
        return self._lib.g1_episodes(self._h)

    def debug_stats(self, i: int, side: int, slot: int):
        out = (ctypes.c_uint16 * 5)()
        self._lib.g1_debug_stats(self._h, i, side, slot, out)
        return dict(zip(["hp", "atk", "def", "spe", "spc"], out))

    def close(self):
        if self._h:
            self._lib.g1_destroy(self._h)
            self._h = None


def random_legal(env: Gen1Env, rng: np.random.Generator):
    """Writes uniform random legal actions for all players that need one."""
    masks = env.masks.reshape(-1, env.n_actions)
    needs = env.needs.reshape(-1)
    logits = np.where(masks > 0, rng.random(masks.shape), -1.0)
    env.actions.reshape(-1)[:] = np.where(needs > 0, logits.argmax(1), 0)


def parity_check(env: Gen1Env, fixture_path: str) -> int:
    """Engine-side stats (computed in Zig from the binary pool) must equal the
    PS-computed fixture. Battle k uses pool teams 2k/2k+1 = fixture[2k]/[2k+1]."""
    import json
    fixture = json.loads(pathlib.Path(fixture_path).read_text())
    checked = 0
    for i in range(min(env.n, len(fixture) // 2)):
        for side in (0, 1):
            expect = fixture[2 * i + side]
            for slot, mon in enumerate(expect):
                got = env.debug_stats(i, side, slot)
                for k in ("hp", "atk", "def", "spe", "spc"):
                    assert got[k] == mon[k], (
                        f"stat parity: battle {i} side {side} slot {slot} "
                        f"{k}: engine {got[k]} != PS {mon[k]} ({mon})")
                checked += 1
    return checked


if __name__ == "__main__":
    import sys
    import time

    pool = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "teams" / "gen1-pool.bin")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    env = Gen1Env(pool, n=n, seed=0)
    fixture = ROOT / "teams" / "parity-fixture.json"
    if fixture.exists():
        print(f"stat parity vs PS fixture: {parity_check(env, fixture)} mons OK")

    rng = np.random.default_rng(0)
    steps = 0
    t0 = time.perf_counter()
    wins = ties = 0
    while env.episodes < target:
        random_legal(env, rng)
        env.step()
        steps += 1
        done = env.dones > 0
        wins += int((env.rewards[done, 0] == 1).sum())
        ties += int((env.rewards[done, 0] == 0).sum())
    dt = time.perf_counter() - t0
    eps = env.episodes
    print(f"{eps} battles, {steps} vector steps in {dt:.1f}s "
          f"({eps / dt:.0f} battles/s, {steps * n / dt:.0f} decisions/s)")
    print(f"p1 wins {wins / eps:.3f}, ties {ties / eps:.4f}")
    env.close()
