#!/usr/bin/env node
// Build a binary team pool for the Zig env from Showdown's own gen1randombattle
// generator, plus (optionally) a stat-parity fixture computed by Pokémon Showdown
// itself (ground truth for the engine-side stat calculation).
//
// Pool format: 12-byte header ["G1PK", u8 version=2, 3 reserved, u32le count],
// then `count` 108-byte team records: 6 mons x 18 bytes:
//   [species, level, move1..move4, ev hp/atk/def/spa/spd/spe, iv hp/atk/def/spa/spd/spe]
// (PS dex numbers; 0 = empty move slot). Gen 1 randbats sets vary EVs/IVs
// (no-Attack sets, HP tweaks), so full spreads are stored.
//
// Usage: node scripts/make_pool.mjs <count> <out.bin> [--fixture <out.json> <k>]

import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const requirePS = createRequire(path.join(here, '../pokemon-showdown/'));
const {Teams} = requirePS('./dist/sim/teams');
const {Battle} = requirePS('./dist/sim/battle');
const {Dex} = requirePS('./dist/sim/dex');

const FORMAT = 'gen1randombattle';
const BATTLE_FORMAT = 'gen1customgame'; // no validation; stats don't depend on rules
const dex = Dex.mod('gen1');

const count = Number(process.argv[2]);
const out = process.argv[3];
if (!count || !out) throw new Error('usage: make_pool.mjs <count> <out.bin> [--fixture <out.json> <k>]');
const fixtureAt = process.argv.indexOf('--fixture');
const [fixtureOut, fixtureK] = fixtureAt > 0
  ? [process.argv[fixtureAt + 1], Number(process.argv[fixtureAt + 2] ?? 64)] : [null, 0];

function seedFor(i) {
  let x = (i + 0x9e3779b9) >>> 0;
  const h = () => {
    x ^= x >>> 16; x = Math.imul(x, 0x21f0aaad) >>> 0; x ^= x >>> 15;
    return x & 0xffff;
  };
  return `${i & 0xffff},${(i >>> 16) & 0xffff},${h()},${h()}`;
}

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const MON_BYTES = 18;

const teamRecord = team => {
  const rec = Buffer.alloc(6 * MON_BYTES);
  if (team.length !== 6) throw new Error(`expected 6 mons, got ${team.length}`);
  team.forEach((set, j) => {
    const o = j * MON_BYTES;
    const species = dex.species.get(set.species);
    if (!(species.num >= 1 && species.num <= 151)) throw new Error(`bad species ${set.species}`);
    if (!(set.level >= 1 && set.level <= 100)) throw new Error(`bad level ${set.level}`);
    if (set.moves.length < 1 || set.moves.length > 4) throw new Error(`bad moves ${set.moves}`);
    rec[o] = species.num;
    rec[o + 1] = set.level;
    set.moves.forEach((m, k) => {
      const move = dex.moves.get(m);
      if (!(move.num >= 1 && move.num <= 165)) throw new Error(`bad move ${m}`);
      rec[o + 2 + k] = move.num;
    });
    STATS.forEach((st, k) => {
      const ev = set.evs?.[st] ?? 255;
      const iv = set.ivs?.[st] ?? 30;
      if (!(ev >= 0 && ev <= 255)) throw new Error(`bad ev ${st}=${ev}`);
      // odd ivs would break the exact dv = iv >> 1 equivalence in the env
      if (!(iv >= 0 && iv <= 30) || iv % 2) throw new Error(`bad iv ${st}=${iv}`);
      rec[o + 6 + k] = ev;
      rec[o + 12 + k] = iv;
    });
  });
  return rec;
};

fs.mkdirSync(path.dirname(out), {recursive: true});
const fd = fs.openSync(out, 'w');
const header = Buffer.alloc(12);
header.write('G1PK', 0, 'ascii');
header[4] = 2;
header.writeUInt32LE(count, 8);
fs.writeSync(fd, header);

const t0 = performance.now();
const teams = [];
for (let i = 0; i < count; i++) {
  const team = Teams.generate(FORMAT, {seed: seedFor(i)});
  fs.writeSync(fd, teamRecord(team));
  if (fixtureOut && i < fixtureK) teams.push(team);
}
fs.closeSync(fd);
const dt = (performance.now() - t0) / 1000;
console.log(`${count} teams -> ${out} in ${dt.toFixed(1)}s (${Math.round(count / dt)}/s)`);

if (fixtureOut) {
  // PS computes the ground-truth stats: pair up fixture teams in battles and
  // read maxhp + storedStats off the instantiated Pokemon
  const fixture = [];
  for (let i = 0; i < teams.length; i += 2) {
    const b = new Battle({formatid: BATTLE_FORMAT, seed: '1,2,3,4'});
    b.setPlayer('p1', {name: 'A', team: teams[i]});
    b.setPlayer('p2', {name: 'B', team: teams[i + 1] ?? teams[i]});
    for (const [ti, side] of [[i, b.p1], [i + 1, b.p2]]) {
      if (ti >= teams.length) break;
      fixture[ti] = side.pokemon.map(p => ({
        species: p.species.num,
        level: p.level,
        hp: p.maxhp,
        atk: p.storedStats.atk,
        def: p.storedStats.def,
        spe: p.storedStats.spe,
        spc: p.storedStats.spa,
      }));
    }
  }
  fs.writeFileSync(fixtureOut, JSON.stringify(fixture));
  console.log(`${fixture.length} team stat fixtures -> ${fixtureOut}`);
}
