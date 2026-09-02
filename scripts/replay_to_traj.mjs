#!/usr/bin/env node
// Replay gen1randombattle inputlogs through Pokémon Showdown and emit training
// trajectories in the env's exact observation layout (v1, masked variant).
//
// Each replay JSON contains the battle seed + per-player team-generation seeds
// (sodium) + the full input stream, so battles re-simulate deterministically.
// We verify the re-simulation against the recorded protocol log turn by turn
// and DISCARD any battle that diverges (PS version drift since recording).
//
// Output: binary shards, 1048-byte records (see FIELD LAYOUT below), one
// record per human decision (>1 legal choice, matching env auto-resolve).
//
// Usage: node scripts/replay_to_traj.mjs <out_dir> <shard.jsonl> [...more]

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
// PS_PATH: point at an era-matched checkout (worktree); old builds ship
// .sim-dist instead of dist/sim
const psRoot = process.env.PS_PATH || path.join(here, '../pokemon-showdown');
const requirePS = createRequire(psRoot + path.sep);
function loadPS() {
  try {
    return {
      battle: requirePS('./dist/sim/battle'),
      dex: requirePS('./dist/sim/dex'),
    };
  } catch {
    return {
      battle: requirePS('./.sim-dist/battle'),
      dex: requirePS('./.sim-dist/dex'),
    };
  }
}
const _ps = loadPS();
const Battle = _ps.battle.Battle ?? _ps.battle;
const Dex = _ps.dex.Dex ?? _ps.dex; // old eras: dex module IS the Dex
// extractChannelMessages appeared ~2022; the 2018-2020 sim logs use the
// legacy convention: |split followed by FOUR channel lines in the order
// [spectator, p1, p2, omniscient] - keep the omniscient one (exact HP),
// matching how these replays were recorded
const extractChannelMessages = _ps.battle.extractChannelMessages ?? ((log) => {
  const out = [];
  const lines = log.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('|split')) {
      out.push(lines[i + 4] ?? '');
      i += 4;
    } else {
      out.push(lines[i]);
    }
  }
  return {'-1': out};
});
const dex = Dex.mod('gen1');
// era-tolerant move lookup: dex.moves.get is the 2021+ API
const getMove = dex.moves?.get ? (n) => getMove(n) : (n) => dex.getMove(n);

const [outDir, ...shards] = process.argv.slice(2);
if (!outDir || !shards.length) throw new Error('usage: replay_to_traj.mjs <out_dir> <shards...>');
fs.mkdirSync(outDir, {recursive: true});

// --- engine obs layout v1 (must match env/src/main.zig encode/maskObs) -------
const REC = 1048;
// header u32 battle_id | u8 player | u8 done | i8 ret | u8 pad          (8)
// u8 act | u8 mask[10] | u8 pad                                        (12)
// u8 rev[6] | u8 pad[2]                                                 (8)
// i16 sp[6]                                                            (12)
// i16 mv[24]                                                           (48)
// i32 mi[80]                                                          (320)
// f32 mf[160]                                                         (640)

const VOLATILE_ORDER = [
  'bide', 'lockedmove', null /* MultiHit: intra-turn */, 'flinch', 'twoturnmove',
  'partialtrappinglock', null /* Invulnerable: fly|dig */, 'confusion', 'mist',
  'focusenergy', 'substitute', 'mustrecharge', 'rage', 'leechseed', 'residualdmg',
  'lightscreen', 'reflect', 'transform',
];

const statusByte = p => {
  const t = p.statusState?.time ?? 0;
  switch (p.status) {
    case 'slp': return Math.min(7, Math.max(1, t));
    case 'psn': return 1 << 3;
    case 'brn': return 1 << 4;
    case 'frz': return 1 << 5;
    case 'par': return 1 << 6;
    case 'tox': return 0x88;
    default: return 0;
  }
};

const volatileBits = p => {
  let bits = 0;
  VOLATILE_ORDER.forEach((name, i) => {
    if (name && p.volatiles[name]) bits |= 1 << i;
  });
  if (p.volatiles['fly'] || p.volatiles['dig']) bits |= 1 << 6;
  return bits;
};

const boostsRaw = p => {
  const order = ['atk', 'def', 'spe', 'spa', 'accuracy', 'evasion'];
  let raw = 0;
  order.forEach((b, i) => {
    raw |= (p.boosts[b] & 0xf) << (i * 4);
  });
  return raw;
};

const moveNum = id => {
  const m = getMove(id);
  return m?.num >= 1 && m.num <= 165 ? m.num : 0;
};

class Encoder {
  constructor() {
    this.revealedMon = new Set(); // Pokemon object refs
    this.usedMoves = new Map(); // Pokemon -> Set(move id)
  }

  markSwitch(pokemon) {
    this.revealedMon.add(pokemon);
  }

  markMove(pokemon, moveId) {
    if (!this.usedMoves.has(pokemon)) this.usedMoves.set(pokemon, new Set());
    this.usedMoves.get(pokemon).add(moveId);
  }

  /** masked obs for `side`'s player; also returns belief targets. */
  encode(battle, side) {
    const mi = new Int32Array(80);
    const mf = new Float32Array(160);
    const sp = new Int16Array(6);
    const mv = new Int16Array(24);
    const rev = new Uint8Array(6);
    const foe = side.foe;

    for (const [s, mine] of [[side, true], [foe, false]]) {
      const off = mine ? 0 : 6;
      s.pokemon.forEach((p, j) => {
        if (j >= 6) return;
        const ii = (off + j) * 6;
        const fi = (off + j) * 8;
        const revealed = mine || this.revealedMon.has(p);
        const active = s.active[0] === p;
        if (!mine) {
          sp[j] = p.species.num;
          p.moveSlots.forEach((ms, k) => {
            if (k < 4) mv[j * 4 + k] = moveNum(ms.id);
          });
          rev[j] = revealed ? 1 : 0;
        }
        if (!revealed) {
          mf[fi] = 1.0; // unrevealed bench mons are provably undamaged
          return;
        }
        mi[ii] = p.species.num;
        mi[ii + 1] = statusByte(p);
        const used = this.usedMoves.get(p);
        p.moveSlots.forEach((ms, k) => {
          if (k >= 4) return;
          if (mine || used?.has(ms.id)) mi[ii + 2 + k] = moveNum(ms.id);
          if (mine) mf[fi + 4 + k] = ms.pp / 63.0;
        });
        mf[fi] = p.hp / p.maxhp;
        mf[fi + 1] = p.level / 100.0;
        mf[fi + 2] = active ? 1 : 0;
        mf[fi + 3] = 1;
      });
    }

    // extras (layout landmarks match env maskObs; opponent privates zeroed)
    for (const [k, s] of [[0, side], [1, foe]]) {
      const a = s.active[0];
      mi[72 + k] = a ? volatileBits(a) : 0;
      const bf = 96 + k * 24;
      if (a) {
        ['atk', 'def', 'spe', 'spa', 'accuracy', 'evasion'].forEach((b, i) => {
          mf[bf + i] = a.boosts[b] / 6.0;
        });
        const bits = volatileBits(a);
        for (let i = 0; i < 18; i++) mf[bf + 6 + i] = (bits >> i) & 1;
      }
      mi[74 + k * 3] = a ? boostsRaw(a) : 0;
      mi[75 + k * 3] = k === 0 ? moveNum(s.lastSelectedMove ?? '') : 0; // opp private
      mi[76 + k * 3] = moveNum(s.lastMove?.id ?? '');
      const sf = 144 + k * 5;
      if (k === 0 && a) { // own sleep counter + modified stats; opp stays zero
        mf[sf] = (a.status === 'slp' ? Math.min(7, a.statusState?.time ?? 0) : 0) / 7.0;
        const ms = a.modifiedStats ?? a.storedStats;
        ['atk', 'def', 'spe', 'spa'].forEach((st, i) => {
          mf[sf + 1 + i] = (ms[st] ?? 0) / 1000.0;
        });
      }
    }
    mf[154] = battle.turn / 500.0;
    return {mi, mf, sp, mv, rev};
  }
}

// --- legal choice enumeration from PS requests (mirrors env action space) -----
function legalActions(side) {
  const req = side.activeRequest;
  if (!req || req.wait) return [];
  const acts = [];
  if (req.forceSwitch) {
    for (let slot = 2; slot <= 6; slot++) {
      const p = side.pokemon[slot - 1];
      if (p && p.hp > 0) acts.push(2 + slot); // switch -> action 4..8
    }
    return acts;
  }
  if (!req.active) return [];
  const active = side.active[0];
  if (!active.trapped) {
    for (let slot = 2; slot <= 6; slot++) {
      const p = side.pokemon[slot - 1];
      if (p && p.hp > 0) acts.push(2 + slot);
    }
  } else if (req.active[0].moves.length === 1) {
    return [9]; // locked/struggle -> special action
  }
  const before = acts.length;
  req.active[0].moves.forEach((m, k) => {
    if (!m.disabled && (m.pp === undefined || m.pp > 0)) acts.push(k);
  });
  if (acts.length === before) acts.push(9);
  return acts;
}

const FILTER = /^\|(move|switch|faint|win|tie|turn|cant|-damage|-heal|-status|-curestatus)\|/;
// channel -1 = omniscient view: collapses |split| secret/public dual lines,
// matching how replays are recorded
const filtered = log => extractChannelMessages(log.join('\n'), [-1])[-1]
  .filter(l => FILTER.test(l))
  .map(l => l.split('|').slice(1, 4).join('|')); // trim trailing kwargs

// --- one replay ----------------------------------------------------------------
function processReplay(replay, battleId, out) {
  const input = (replay.inputlog || '').split('\n');
  const start = input.find(l => l.startsWith('>start'));
  const p1l = input.find(l => l.startsWith('>player p1'));
  const p2l = input.find(l => l.startsWith('>player p2'));
  if (!start || !p1l || !p2l) return 'no-input';

  const spec = JSON.parse(start.slice(7));
  if (!/^gen1randombattle/.test(spec.formatid)) return 'wrong-format';
  const battle = new Battle({formatid: spec.formatid, seed: spec.seed, strictChoices: false});
  battle.setPlayer('p1', JSON.parse(p1l.slice(11)));
  battle.setPlayer('p2', JSON.parse(p2l.slice(11)));

  const expect = filtered(replay.log.split('\n'));
  const enc = new Encoder();
  const rows = []; // {player, act, mask, obs...}
  let logPos = 0;
  let verified = 0;

  const consume = () => {
    // verify newly produced protocol against the recording; track reveals
    for (const line of filtered(battle.log.slice(logPos))) {
      if (line !== expect[verified]) return false;
      verified++;
    }
    for (const raw of battle.log.slice(logPos)) {
      const parts = raw.split('|');
      if (parts[1] === 'switch') {
        const sideId = parts[2].slice(0, 2);
        const side = battle[sideId];
        enc.markSwitch(side.active[0]);
      } else if (parts[1] === 'move') {
        const sideId = parts[2].slice(0, 2);
        const side = battle[sideId];
        const move = getMove(parts[3]);
        if (side.active[0] && move) enc.markMove(side.active[0], move.id);
      }
    }
    logPos = battle.log.length;
    return true;
  };
  if (!consume()) return 'diverged-at-start';

  for (const line of input) {
    const m = line.match(/^>(p[12]) (.+)$/);
    if (!m) {
      if (line.startsWith('>forcewin') || line.startsWith('>forcetie')) break;
      continue;
    }
    const [_, sideId, cmd] = m;
    const side = battle[sideId];
    if (battle.ended) break;

    // action label + emission (only real decisions, matching env semantics)
    const acts = legalActions(side);
    let label = -1;
    const mvMatch = cmd.match(/^move (.+?)( |$)/);
    const swMatch = cmd.match(/^switch (\d)/);
    if (mvMatch && side.active[0]) {
      const arg = mvMatch[1];
      const slot = /^[1-4]$/.test(arg)
        ? +arg - 1
        : side.active[0].moveSlots.findIndex(ms => ms.id === getMove(arg).id);
      label = acts.includes(9) && acts.length === 1 ? 9 : slot;
    } else if (swMatch) {
      label = 2 + +swMatch[1];
    } else if (cmd === 'default' || cmd.startsWith('undo')) {
      label = -2; // timer/undo: apply but do not emit
    }
    if (label >= 0 && acts.length > 1 && acts.includes(label)) {
      const o = enc.encode(battle, side);
      const mask = new Uint8Array(10);
      for (const a of acts) mask[a] = 1;
      rows.push({player: sideId === 'p1' ? 0 : 1, act: label, mask, ...o});
    }
    battle.choose(sideId, cmd);
    if (!consume()) return 'diverged';
  }

  // outcome from the recorded log (covers forfeits/timer)
  const winLine = replay.log.split('\n').find(l => l.startsWith('|win|') || l === '|tie');
  if (!winLine) return 'no-outcome';
  const p1name = JSON.parse(p1l.slice(11)).name;
  const r1 = winLine === '|tie' ? 0 : (winLine.slice(5) === p1name ? 1 : -1);
  if (!rows.length) return 'no-decisions';

  const buf = Buffer.alloc(REC);
  rows.forEach((r, i) => {
    buf.fill(0);
    const last = i === rows.length - 1
      || !rows.slice(i + 1).some(x => x.player === r.player);
    buf.writeUInt32LE(battleId, 0);
    buf.writeUInt8(r.player, 4);
    buf.writeUInt8(last ? 1 : 0, 5);
    buf.writeInt8(r.player === 0 ? r1 : -r1, 6);
    buf.writeUInt8(r.act, 8);
    Buffer.from(r.mask).copy(buf, 9);
    Buffer.from(r.rev).copy(buf, 20);
    Buffer.from(r.sp.buffer, r.sp.byteOffset, 12).copy(buf, 28);
    Buffer.from(r.mv.buffer, r.mv.byteOffset, 48).copy(buf, 40);
    Buffer.from(r.mi.buffer, r.mi.byteOffset, 320).copy(buf, 88);
    Buffer.from(r.mf.buffer, r.mf.byteOffset, 640).copy(buf, 408);
    out.write(Buffer.from(buf));
  });
  return 'ok';
}

// --- main ------------------------------------------------------------------------
let battleId = 0;
const stats = {};
for (const shard of shards) {
  const outPath = path.join(outDir, path.basename(shard).replace('.jsonl', '.traj'));
  const out = fs.createWriteStream(outPath);
  const lines = fs.readFileSync(shard, 'utf8').split('\n').filter(Boolean);
  for (const line of lines) {
    let status;
    try {
      status = processReplay(JSON.parse(line), battleId++, out);
    } catch (err) {
      status = `error:${err.message.slice(0, 60)}`;
    }
    stats[status] = (stats[status] ?? 0) + 1;
  }
  out.end();
  console.log(`${shard} -> ${outPath}`);
}
console.log(JSON.stringify(stats, null, 1));
