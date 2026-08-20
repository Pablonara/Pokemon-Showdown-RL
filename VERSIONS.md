# Pinned dependencies (vendored, gitignored — re-clone at these SHAs)

| Dep | Source | Pin |
|---|---|---|
| `engine/` | https://github.com/pkmn/engine | `78dc891c49788e6ec9007d0f02247d2e04a03d29` |
| `pokemon-showdown/` | https://github.com/smogon/pokemon-showdown | `d43fb79a0` |

## Setup

```sh
git clone https://github.com/pkmn/engine.git engine
git -C engine checkout 78dc891c49788e6ec9007d0f02247d2e04a03d29
node engine/src/bin/install-pkmn-engine --zig   # installs Zig 0.16.0 → engine/build/bin/zig/zig

git clone https://github.com/smogon/pokemon-showdown.git
git -C pokemon-showdown checkout d43fb79a0
```

Zig toolchain: 0.16.0 (matches engine CI "local" matrix entry).
