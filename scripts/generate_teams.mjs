#!/usr/bin/env node
// Generate a pool of Pokémon Showdown random-battle teams, one packed team per line.
// Uses Showdown's own generator => exact ladder team distribution.
//
// Usage: node scripts/generate_teams.mjs <count> <outfile> [format=gen1randombattle]

import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const sim = require(path.join(here, '../pokemon-showdown/dist/sim/index.js'));
const Teams = sim.Teams ?? sim.default?.Teams;
if (!Teams) throw new Error('could not load Teams from pokemon-showdown build');

const count = Number(process.argv[2] ?? 10_000);
const out = process.argv[3] ?? 'teams/gen1-pool.txt';
const format = process.argv[4] ?? 'gen1randombattle';

// Deterministic, collision-free 4x16-bit PRNG seed per team index:
// two words carry i verbatim, two words are a hash of i.
function seedFor(i) {
  let x = (i + 0x9e3779b9) >>> 0;
  const h = () => {
    x ^= x >>> 16; x = Math.imul(x, 0x21f0aaad) >>> 0; x ^= x >>> 15;
    return x & 0xffff;
  };
  return `${i & 0xffff},${(i >>> 16) & 0xffff},${h()},${h()}`;
}

fs.mkdirSync(path.dirname(out), {recursive: true});
const stream = fs.createWriteStream(out);
const t0 = performance.now();
for (let i = 0; i < count; i++) {
  const team = Teams.generate(format, {seed: seedFor(i)});
  if (team.length !== 6) throw new Error(`team ${i}: expected 6 mons, got ${team.length}`);
  stream.write(Teams.pack(team) + '\n');
}
stream.end();
const dt = (performance.now() - t0) / 1000;
console.log(`${count} ${format} teams -> ${out} in ${dt.toFixed(1)}s (${Math.round(count / dt)}/s)`);
