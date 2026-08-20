# pokemon-showdown-rl

SOTA-targeting RL agent for **Gen 1 Random Battles** on Pokémon Showdown.
Plan: [PLAN.md](PLAN.md) · Pins: [VERSIONS.md](VERSIONS.md)

## Layout

```
engine/            vendored pkmn/engine (libpkmn) — fast training sim   [gitignored, pinned]
pokemon-showdown/  vendored official sim — team generator + eval truth  [gitignored, pinned]
scripts/           team pool generation, data tooling
bench/             throughput benchmarks
teams/             generated team pools                                 [gitignored]
```

## Quickstart

```sh
# setup (see VERSIONS.md for pinned SHAs)
node engine/src/bin/install-pkmn-engine --zig

# single-core engine benchmark (showdown-compat, random players)
cd engine && build/bin/zig/zig build benchmark -Dshowdown -- 1 1000/10000 0x12345

# generate a team pool (packed format, one team per line)
node scripts/generate_teams.mjs 10000 teams/gen1-pool.txt
```
