#!/usr/bin/env node
// Export static gen 1 data (types, species, moves, type chart) from Pokémon
// Showdown's dex to JSON for Python-side policies/models.
//
// Usage: node scripts/export_gen1_data.mjs [out=data/gen1.json]

import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const {Dex} = createRequire(path.join(here, '../pokemon-showdown/'))('./dist/sim/dex');
const dex = Dex.mod('gen1');

const out = process.argv[2] ?? 'data/gen1.json';

const types = dex.types.all().map(t => t.name); // 15 in gen 1
const typeIndex = Object.fromEntries(types.map((t, i) => [t, i]));

// chart[atk][def] = damage multiplier
const chart = types.map(atk =>
  types.map(def => !dex.getImmunity(atk, [def]) ? 0 : 2 ** dex.getEffectiveness(atk, [def])));

const species = {}; // by dex num (1-151)
for (const s of dex.species.all()) {
  if (s.num < 1 || s.num > 151 || s.isNonstandard) continue;
  species[s.num] = {name: s.name, types: s.types.map(t => typeIndex[t])};
}

const moves = {}; // by move num (1-165)
for (const m of dex.moves.all()) {
  if (m.num < 1 || m.num > 165 || m.isNonstandard) continue;
  moves[m.num] = {
    name: m.name,
    type: typeIndex[m.type],
    power: m.basePower,
    status: m.category === 'Status',
  };
}

fs.mkdirSync(path.dirname(out), {recursive: true});
fs.writeFileSync(out, JSON.stringify({types, chart, species, moves}));
console.log(`gen1 data (${Object.keys(species).length} species, ${Object.keys(moves).length} moves) -> ${out}`);
