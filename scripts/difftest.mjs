#!/usr/bin/env node
// Differential fidelity test (M0 gate): play identical gen1randombattle battles
// in Pokémon Showdown (ladder ground truth) and pkmn/engine (training sim),
// with the same teams, RNG seeds, and choices; assert identical state evolution.
//
// Per battle: PS generates both teams (its own ladder generator, seeded), the
// engine battle is initialized from those exact sets, and every decision step
// picks uniformly among the engine's legal choices (seeded picker), mirrored
// into PS. After each engine update we compare: turn, winner, and each side's
// party as a multiset of species:hp:status (HP is the canary — any RNG
// misalignment cascades into HP differences within a turn).
//
// Known-benign classifications (not failures): 'turnlimit' (PS ties at turn
// 1000; engine has no such rule), 'ebc' (PS Endless Battle Clause; out of
// scope for the engine), 'truncated' (step cap hit, states equal throughout),
// 'softlock' (documented gen1 Transform+Mirror Move/Metronome PP edge case).
//
// Modes (measured 2026-01, engine 78dc891, PS d43fb79a0):
//   --sim  GATE: vs @pkmn/sim 0.9.31, the snapshot the engine is developed
//          against. Must be 100% ok (measured: 1000/1000). Any failure here is
//          a harness/converter/engine bug.
//   (none) DRIFT MONITOR: vs our pinned-HEAD pokemon-showdown. Measured: 97.5%
//          ok; all divergence root-caused to PS's 2025 gen1 rework, post-dating
//          the engine snapshot: (a) Counter now counters stale last-selected-
//          move damage (engine fails it), (b) sleep wake-up moved to turn start
//          (engine wakes on the mon's action; affects same-turn status moves),
//          (c) immobile (slp/frz) mons get a reduced move menu (handled here by
//          remapping to 'move 1'; mechanically a no-op).
//   --ladder-rules  adds Desync Clause Mod etc. (ladder gen1randombattle rule
//          set) to the PS side; measured identical to default mode.
//
// Usage: node scripts/difftest.mjs [count=100] [start=0] [--verbose|--sim|--ladder-rules]
//        node scripts/difftest.mjs --debug <i> [--sim]   # aligned protocol/RNG dump

import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const requirePS = createRequire(path.join(here, '../pokemon-showdown/'));
const requireEng = createRequire(path.join(here, '../engine/'));

// --sim: differential-test against @pkmn/sim (the PS snapshot the engine is
// developed against) instead of our pinned-HEAD pokemon-showdown. Teams always
// come from HEAD's ladder generator either way.
const useSim = process.argv.includes('--sim');
const {Teams} = requirePS('./dist/sim/teams');
const {Battle: PSBattle} = useSim ? requireEng('@pkmn/sim') : requirePS('./dist/sim/battle');
const {Battle: EngBattle, Log, Lookup} = requireEng('./build/pkg/index.js');
const {Generations} = requireEng('@pkmn/data');
const {Dex} = requireEng('@pkmn/dex');

// Tie-free clause-handler priorities (port of engine patch.generation): must
// be injected at module init, *before* any format/battle instantiation caches
// rule effect objects. Both onSetStatus handlers otherwise speed-tie whenever
// any status is applied, consuming a shuffle RNG frame the engine never rolls.
for (const dex of [requirePS('./dist/sim/dex').Dex, useSim ? requireEng('@pkmn/sim').Dex : null]) {
  if (!dex) continue;
  for (const d of [dex, dex.mod('gen1')]) {
    d.data.Rulesets['sleepclausemod'].onSetStatusPriority = -999;
    d.data.Rulesets['freezeclausemod'].onSetStatusPriority = -998;
  }
}

const gen = new Generations(Dex).get(1);
const FORMAT = 'gen1randombattle'; // team generation (ladder distribution)
// Battle rules: the engine implements exactly these mods (see engine
// src/test/showdown.ts formatFor); ladder gen1randombattle additionally has
// Desync Clause Mod (unimplemented by the engine) - measured separately.
const BATTLE_FORMAT = process.argv.includes('--ladder-rules')
  ? 'gen1randombattle'
  : 'gen1customgame@@@Endless Battle Clause,Sleep Clause Mod,Freeze Clause Mod';
const MAX_STEPS = 1500;

const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
const verbose = process.argv.includes('--verbose');
const debug = process.argv.includes('--debug'); // dump aligned protocol logs for one battle
const count = debug ? 1 : Number(args[0] ?? 100);
const start = Number(args[debug ? 0 : 1] ?? 0);

// --- deterministic seeding ---------------------------------------------------
const mulberry32 = a => () => {
  a |= 0; a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const seed4 = rng => Array.from({length: 4}, () => Math.floor(rng() * 0x10000));

// --- engine-compat PS patches -----------------------------------------------
// Port of engine/src/test/showdown.ts `patch`: the engine deliberately does not
// replicate PS's *artificial* speed-tie RNG advances (an architectural artifact,
// not game behavior), and instead matches a minimally patched PS that uses the
// cartridge's "host" ordering in Gens 1-2 and tie-free handler priorities.
// Divergences that remain after these patches indicate real mechanics drift.
function insertChoice(choices, midTurn = false) {
  if (Array.isArray(choices)) {
    for (const choice of choices) this.insertChoice(choice);
    return;
  }
  const choice = choices;
  if (choice.pokemon) choice.pokemon.updateSpeed();
  const actions = this.resolveAction(choice, midTurn);

  let firstIndex = null;
  let lastIndex = null;
  for (const [i, curAction] of this.list.entries()) {
    const compared = this.battle.comparePriority(actions[0], curAction);
    if (compared <= 0 && firstIndex === null) firstIndex = i;
    if (compared < 0) {
      lastIndex = i;
      break;
    }
  }
  if (firstIndex === null) {
    this.list.push(...actions);
  } else {
    if (lastIndex === null) lastIndex = this.list.length;
    // FIX: "host" ordering before gen 3 (no RNG tie roll)
    const index = firstIndex === lastIndex
      ? firstIndex : (this.battle.gen > 3)
        ? this.battle.random(firstIndex, lastIndex + 1)
        : lastIndex;
    this.list.splice(index, 0, ...actions);
  }
}

function eachEvent(eventid, effect, relayVar) {
  const actives = this.getAllActive();
  if (!effect && this.effect) effect = this.effect;
  // FIX: no speed sort before gen 3 = "host" ordering
  if (this.gen >= 3) this.speedSort(actives, (a, b) => b.speed - a.speed);
  for (const pokemon of actives) this.runEvent(eventid, pokemon, null, effect, relayVar);
  if (eventid === 'Weather' && this.gen >= 7) this.eachEvent('Update');
}

function fieldEvent(eventid, targets) {
  const callbackName = `on${eventid}`;
  let getKey;
  if (eventid === 'Residual') getKey = 'duration';
  let handlers = this.findFieldEventHandlers(this.field, `onField${eventid}`, getKey);
  for (const side of this.sides) {
    if (side.n < 2 || !side.allySide) {
      handlers = handlers.concat(this.findSideEventHandlers(side, `onSide${eventid}`, getKey));
    }
    for (const active of side.active) {
      if (!active) continue;
      if (eventid === 'SwitchIn') {
        handlers = handlers.concat(this.findPokemonEventHandlers(active, `onAny${eventid}`));
      }
      if (targets && !targets.includes(active)) continue;
      handlers = handlers.concat(this.findPokemonEventHandlers(active, callbackName, getKey));
      handlers = handlers.concat(this.findSideEventHandlers(side, callbackName, undefined, active));
      handlers = handlers.concat(
        this.findFieldEventHandlers(this.field, callbackName, undefined, active));
      handlers = handlers.concat(this.findBattleEventHandlers(callbackName, getKey, active));
    }
  }
  // FIX: no speed sort before gen 3 = "host" ordering
  if (this.gen >= 3) this.speedSort(handlers);
  while (handlers.length) {
    const handler = handlers[0];
    handlers.shift();
    const effect = handler.effect;
    if (handler.effectHolder.fainted) {
      if (!handler.state?.isSlotCondition) continue;
    }
    if (eventid === 'Residual' && handler.end && handler.state?.duration) {
      handler.state.duration--;
      if (!handler.state.duration) {
        const endCallArgs = handler.endCallArgs || [handler.effectHolder, effect.id];
        handler.end.call(...endCallArgs);
        if (this.ended) return;
        continue;
      }
    }
    let handlerEventid = eventid;
    if (handler.effectHolder.sideConditions) handlerEventid = `Side${eventid}`;
    if (handler.effectHolder.pseudoWeather) handlerEventid = `Field${eventid}`;
    if (handler.callback) {
      this.singleEvent(handlerEventid, effect, handler.state, handler.effectHolder,
        null, null, undefined, handler.callback);
    }
    this.faintMessages();
    if (this.ended) return;
  }
}

function patchBattle(battle) {
  try {
    // Bide/Disable handler tie; PS HEAD freezes conditions so this may fail —
    // safe for gen1randombattle since Bide is not in any randbats movepool.
    battle.dex.conditions.get('disable').onDisableMovePriority = 7;
  } catch { /* frozen dex: acceptable, see above */ }
  battle.queue.insertChoice = insertChoice.bind(battle.queue);
  battle.eachEvent = eachEvent.bind(battle);
  battle.fieldEvent = fieldEvent.bind(battle);
  return battle;
}

// --- state comparison ----------------------------------------------------------
const normStatus = s => !s ? 'ok' : s === 'tox' ? 'psn' : s;
// fainted mons collapse to 'fnt': a fainted-but-unreplaced transformed Pokémon
// transiently reports its copied species in the driver vs its base species in
// PS (pure representation; faint counts + alive identities constrain state)
const key = (species, hp, status) => hp === 0 ? 'fnt' : `${species}:${hp}:${normStatus(status)}`;

const psParty = b =>
  b.sides.map(s => s.pokemon.map(p => key(p.species.id, p.hp, p.status)).sort().join(' '));
const engParty = b =>
  [...b.sides].map(s => [...s.pokemon].map(p => key(p.species, p.hp, p.status)).sort().join(' '));

// --- single battle -------------------------------------------------------------
const KEEP = /^\|(move|switch|cant|faint|win|tie|turn|-\w+)\|/;
function runBattle(i) {
  const rng = mulberry32(0x5eed ^ Math.imul(i + 1, 0x9e3779b9));
  const battleSeed = seed4(rng);
  const t1 = Teams.generate(FORMAT, {seed: seed4(rng).join(',')});
  const t2 = Teams.generate(FORMAT, {seed: seed4(rng).join(',')});
  // patch *before* players join: setting both players starts the battle
  const ps = patchBattle(new PSBattle(
    {formatid: BATTLE_FORMAT, seed: battleSeed.join(','), strictChoices: true}));
  ps.setPlayer('p1', {name: 'P1', team: t1});
  ps.setPlayer('p2', {name: 'P2', team: t2});
  const engOptions = {
    p1: {name: 'P1', team: t1},
    p2: {name: 'P2', team: t2},
    seed: battleSeed,
    showdown: true,
    log: debug,
  };
  const eng = EngBattle.create(gen, engOptions);
  const engLog = debug ? new Log(gen, Lookup.get(gen), engOptions) : null;
  let psLogLen = ps.log.length;

  const dump = (step, picks) => {
    const fmt = c => (c ? `${c.type} ${c.data}` : 'start');
    const psSeed = ps.prng.getSeed().toString();
    const engSeed = [...eng.prng].join(',');
    const drift = psSeed !== engSeed ? '  << RNG DRIFT' : '';
    console.log(`--- step ${step}: p1={${fmt(picks?.[0])}} p2={${fmt(picks?.[1])}}`);
    console.log(`    rng ps=[${psSeed}] eng=[${engSeed}]${drift}`);
    for (const l of ps.log.slice(psLogLen)) if (KEEP.test(l)) console.log('  PS ', l);
    try {
      for (const {args} of engLog.parse(eng.log)) {
        const l = '|' + args.join('|');
        if (KEEP.test(l)) console.log('  ENG', l);
      }
    } catch (err) {
      console.log('  ENG <log parse error:', err.message + '>');
    }
  };

  let result = eng.update(); // pass/pass: engine switches in leads (PS did at construction)
  if (debug) dump(-1, null);
  for (let step = 0; ; step++) {
    // compare state
    const [psP, engP] = [psParty(ps), engParty(eng)];
    if (eng.turn !== ps.turn || psP[0] !== engP[0] || psP[1] !== engP[1]) {
      return {status: 'mismatch', detail: {step, engTurn: eng.turn, psTurn: ps.turn, psP, engP}};
    }
    // termination
    if (result.type) {
      const winner = {win: 'P1', lose: 'P2', tie: ''}[result.type];
      if (result.type === 'error') return {status: 'mismatch', detail: {step, error: 'engine error'}};
      if (!ps.ended || (ps.winner ?? '') !== winner) {
        return {status: 'mismatch', detail: {step, engWinner: winner, psEnded: ps.ended, psWinner: ps.winner}};
      }
      return {status: 'ok', turns: eng.turn};
    }
    if (ps.ended) {
      const tail = ps.log.slice(-25).join('\n');
      if (ps.turn >= 1000) return {status: 'turnlimit', turns: ps.turn};
      if (tail.includes('Endless Battle Clause')) return {status: 'ebc', turns: ps.turn};
      return {status: 'mismatch', detail: {step, error: 'PS ended, engine did not', psWinner: ps.winner}};
    }
    if (step >= MAX_STEPS) return {status: 'truncated', turns: eng.turn};
    psLogLen = ps.log.length;

    // choices: engine legal set is authoritative; mirror into PS
    const picks = [];
    for (const id of ['p1', 'p2']) {
      const choices = eng.choices(id, result);
      if (!choices.length) return {status: 'softlock', turns: eng.turn};
      picks.push(choices[Math.floor(rng() * choices.length)]);
    }
    for (const [j, id] of ['p1', 'p2'].entries()) {
      const c = picks[j];
      if (c.type === 'pass') continue;
      const str = c.type === 'move' ? `move ${c.data || 1}` : `switch ${c.data}`;
      try {
        if (!ps.choose(id, str)) throw new Error(`PS rejected '${str}'`);
      } catch (err) {
        // HEAD PS presents immobile (slp/frz) gen1 mons a reduced move menu;
        // the engine (matching the older snapshot) offers all slots. The choice
        // is a formality (the mon cannot act) - remap to the canonical slot.
        if (c.type === 'move' && c.data > 1 && /doesn't have a move/.test(err.message)) {
          if (ps.choose(id, 'move 1')) continue;
        }
        return {status: 'mismatch', detail: {step, error: err.message, [`${id}Choice`]: c}};
      }
    }
    result = eng.update(picks[0], picks[1]);
    if (debug) dump(step, picks);
  }
}

// --- main ------------------------------------------------------------------------
const t0 = performance.now();
const tally = {ok: 0, mismatch: 0, turnlimit: 0, ebc: 0, truncated: 0, softlock: 0};
const failures = [];
for (let i = start; i < start + count; i++) {
  let r;
  try {
    r = runBattle(i);
  } catch (err) {
    r = {status: 'mismatch', detail: {error: `exception: ${err.message}`}};
  }
  tally[r.status]++;
  if (r.status === 'mismatch') failures.push({battle: i, ...r.detail});
  if (verbose) console.log(`battle ${i}: ${r.status}${r.turns ? ` (${r.turns} turns)` : ''}`);
}
const dt = (performance.now() - t0) / 1000;

console.log(`\n${count} battles in ${dt.toFixed(1)}s (${(count / dt).toFixed(0)}/s)`);
console.log(Object.entries(tally).filter(([, n]) => n).map(([k, n]) => `${k}: ${n}`).join(', '));
for (const f of failures.slice(0, 5)) console.log('FAIL', JSON.stringify(f, null, 1));
if (failures.length > 5) console.log(`... and ${failures.length - 5} more failures`);
process.exit(failures.length ? 1 : 0);
