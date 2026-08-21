#!/usr/bin/env node
// Pokémon Showdown websocket bot: plays gen1randombattle using the model
// server (python/serve_model.py). Works against a local server (challenge it
// from your browser) or the official ladder (registered account + --pass).
//
// State tracking: own side comes from |request| JSON (exact); opponent side is
// tracked from public protocol events (reveals, hp%, status, boosts,
// volatiles). Known approximations vs training obs: sleep counters are not
// client-visible (encoded mid-range), opponent party order is
// active-first/reveal-order, own modified stats come from boost tables.
//
// Usage:
//   node scripts/showdown_bot.mjs --server ws://localhost:8000/showdown/websocket \
//        --name RanchuBot [--pass secret] [--model http://127.0.0.1:8765] \
//        [--accept] [--search N]

import path from 'node:path';
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {WebSocket} from 'ws';

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = JSON.parse(fs.readFileSync(path.join(here, '../data/gen1.json'), 'utf8'));

const norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
const SPECIES = new Map(Object.entries(DATA.species).map(([n, s]) => [norm(s.name), +n]));
const MOVES = new Map(Object.entries(DATA.moves).map(([n, m]) => [norm(m.name), +n]));

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a.startsWith('--')) args[a.slice(2)] = process.argv[i + 1]?.startsWith('--') || i + 1 >= process.argv.length ? true : process.argv[++i];
}
const SERVER = args.server ?? 'ws://localhost:8000/showdown/websocket';
const NAME = args.name ?? 'RanchuBot';
const MODEL = args.model ?? 'http://127.0.0.1:8765';
const FORMAT = args.format ?? 'gen1randombattle';
let searches = +(args.search ?? 0);

// --- battle state tracking ----------------------------------------------------
const STATUS_BITS = {psn: 1 << 3, brn: 1 << 4, frz: 1 << 5, par: 1 << 6, tox: 0x88, slp: 2};
const VOLS = ['bide', 'lockedmove', null, 'flinch', 'twoturnmove', 'partialtrappinglock',
  null, 'confusion', 'mist', 'focusenergy', 'substitute', 'mustrecharge', 'rage',
  'leechseed', 'residualdmg', 'lightscreen', 'reflect', 'transform'];
const VOLMAP = { // protocol effect name -> volatile key
  confusion: 'confusion', substitute: 'substitute', leechseed: 'leechseed',
  mist: 'mist', focusenergy: 'focusenergy', reflect: 'reflect',
  lightscreen: 'lightscreen', bide: 'bide', mustrecharge: 'mustrecharge',
  transform: 'transform',
};

class Battle {
  constructor(room, mySide) {
    this.room = room;
    this.my = mySide; // 'p1'|'p2'
    this.request = null;
    this.opp = []; // [{num, level, hp, status, sleepBit, moves:Set, active}]
    this.boosts = {p1: {}, p2: {}};
    this.vols = {p1: {}, p2: {}};
    this.lastUsed = {p1: 0, p2: 0};
    this.turn = 0;
    this.myStatusExtra = new Map(); // ident -> slp bit tracking
  }

  oppActive() {
    return this.opp.find(m => m.active);
  }

  handle(parts) {
    const [cmd, ...rest] = parts;
    const side = rest[0]?.slice(0, 2);
    const mine = side === this.my;
    switch (cmd) {
      case 'turn':
        this.turn = +rest[0];
        break;
      case 'switch': case 'drag': {
        if (mine) {
          this.vols[this.my] = {};
          this.boosts[this.my] = {};
          break;
        }
        this.vols[side] = {};
        this.boosts[side] = {};
        for (const m of this.opp) m.active = false;
        const specNum = SPECIES.get(norm(rest[1].split(',')[0])) ?? 0;
        let mon = this.opp.find(m => m.num === specNum);
        if (!mon) {
          const level = +(rest[1].match(/, L(\d+)/)?.[1] ?? 100);
          mon = {num: specNum, level, hp: 1, status: 0, moves: new Set(), active: false};
          this.opp.push(mon);
        }
        mon.active = true;
        mon.hp = parseHP(rest[2], mon.hp);
        break;
      }
      case 'move':
        this.lastUsed[side] = MOVES.get(norm(rest[1])) ?? 0;
        if (!mine) this.oppActive()?.moves.add(this.lastUsed[side]);
        break;
      case '-damage': case '-heal': case '-sethp':
        if (!mine && this.oppActive()) this.oppActive().hp = parseHP(rest[1], this.oppActive().hp);
        break;
      case 'faint':
        if (!mine && this.oppActive()) {
          this.oppActive().hp = 0;
          this.oppActive().status = 0;
        }
        break;
      case '-status':
        if (!mine && this.oppActive()) this.oppActive().status = STATUS_BITS[rest[1]] ?? 0;
        break;
      case '-curestatus':
        if (!mine) {
          const mon = this.opp.find(m => m.active); // only actives cure in gen1 (rest/haze)
          if (mon) mon.status = 0;
        }
        break;
      case '-boost': case '-unboost': {
        const stat = rest[1];
        const d = (cmd === '-boost' ? 1 : -1) * +rest[2];
        this.boosts[side][stat] = Math.max(-6, Math.min(6, (this.boosts[side][stat] ?? 0) + d));
        break;
      }
      case '-clearallboost':
        this.boosts = {p1: {}, p2: {}};
        break;
      case '-start': case '-activate': {
        const eff = norm((rest[1] ?? '').replace(/^move: /, ''));
        const key = VOLMAP[eff] ?? (eff === 'wrap' || eff === 'bind' || eff === 'firespin'
          || eff === 'clamp' ? 'partialtrappinglock' : null);
        if (key) this.vols[side][key] = true;
        break;
      }
      case '-end': {
        const eff = norm((rest[1] ?? '').replace(/^move: /, ''));
        if (VOLMAP[eff]) delete this.vols[side][VOLMAP[eff]];
        break;
      }
    }
  }

  /** obs layout v1 (see env/src/main.zig); returns {ints, floats, mask, acts} */
  encode() {
    const req = this.request;
    const ints = new Int32Array(80);
    const floats = new Float32Array(160);
    // own side (blocks 0-5) from request
    req.side.pokemon.forEach((p, j) => {
      if (j >= 6) return;
      const ii = j * 6;
      const fi = j * 8;
      ints[ii] = SPECIES.get(norm(p.details.split(',')[0])) ?? 0;
      const [hp, status] = parseCondition(p.condition);
      ints[ii + 1] = status;
      (p.moves ?? []).forEach((mv, k) => {
        if (k < 4) ints[ii + 2 + k] = MOVES.get(norm(mv)) ?? 0;
      });
      floats[fi] = hp;
      floats[fi + 1] = +(p.details.match(/, L(\d+)/)?.[1] ?? 100) / 100;
      floats[fi + 2] = p.active ? 1 : 0;
      floats[fi + 3] = 1;
      // pp only present on the active request block; approximate bench pp as full
      floats[fi + 4] = floats[fi + 5] = floats[fi + 6] = floats[fi + 7] = 1;
    });
    if (req.active?.[0]?.moves) {
      req.active[0].moves.forEach((m, k) => {
        if (k < 4) floats[4 + k] = (m.pp ?? 63) / 63;
      });
    }
    // opponent side (blocks 6-11): active first, then reveal order
    const opp = [...this.opp].sort((a, b) => (b.active ? 1 : 0) - (a.active ? 1 : 0));
    opp.forEach((m, j) => {
      if (j >= 6) return;
      const ii = (6 + j) * 6;
      const fi = (6 + j) * 8;
      ints[ii] = m.num;
      ints[ii + 1] = m.status;
      [...m.moves].forEach((mv, k) => {
        if (k < 4) ints[ii + 2 + k] = mv;
      });
      floats[fi] = m.hp;
      floats[fi + 1] = m.level / 100;
      floats[fi + 2] = m.active ? 1 : 0;
      floats[fi + 3] = 1;
    });
    for (let j = this.opp.length; j < 6; j++) floats[(6 + j) * 8] = 1.0; // unrevealed: full hp
    // extras
    const other = this.my === 'p1' ? 'p2' : 'p1';
    const BOOST_ORDER = ['atk', 'def', 'spe', 'spa', 'accuracy', 'evasion'];
    [[0, this.my], [1, other]].forEach(([k, s]) => {
      let bits = 0;
      VOLS.forEach((v, i) => {
        if (v && this.vols[s][v]) bits |= 1 << i;
      });
      ints[72 + k] = bits;
      let raw = 0;
      BOOST_ORDER.forEach((b, i) => {
        const val = this.boosts[s][b] ?? 0;
        floats[96 + k * 24 + i] = val / 6;
        raw |= (val & 0xf) << (i * 4);
      });
      for (let i = 0; i < 18; i++) floats[96 + k * 24 + 6 + i] = (bits >> i) & 1;
      ints[74 + k * 3] = raw;
      ints[76 + k * 3] = this.lastUsed[s];
    });
    ints[75] = this.lastUsed[this.my]; // own last selected ~ last used (approx)
    // own active modified stats ~ stored stats x boost table (approx)
    const a = req.side.pokemon.find(p => p.active);
    if (a?.stats) {
      const table = x => (x >= 0 ? (2 + x) / 2 : 2 / (2 - x));
      ['atk', 'def', 'spe', 'spa'].forEach((st, i) => {
        floats[145 + i] = Math.min(999, (a.stats[st] ?? 0)
          * table(this.boosts[this.my][st] ?? 0)) / 1000;
      });
    }
    floats[154] = this.turn / 500;

    // legal actions (mirrors env action space)
    const acts = [];
    if (req.forceSwitch?.[0]) {
      req.side.pokemon.forEach((p, j) => {
        if (j >= 1 && j < 6 && !p.condition.endsWith(' fnt') && p.condition !== '0 fnt') {
          acts.push(2 + (j + 1));
        }
      });
    } else if (req.active) {
      const act0 = req.active[0];
      if (!act0.trapped) {
        req.side.pokemon.forEach((p, j) => {
          if (j >= 1 && j < 6 && !p.condition.endsWith(' fnt') && p.condition !== '0 fnt') {
            acts.push(2 + (j + 1));
          }
        });
      }
      const before = acts.length;
      (act0.moves ?? []).forEach((m, k) => {
        if (!m.disabled && (m.pp === undefined || m.pp > 0)) acts.push(k);
      });
      if (acts.length === before) acts.push(9);
    }
    const mask = new Uint8Array(10);
    for (const x of acts) mask[x] = 1;
    return {ints: [...ints], floats: [...floats], mask: [...mask], acts};
  }
}

function parseHP(cond, fallback) {
  if (!cond) return fallback;
  if (cond.startsWith('0') || cond.includes('fnt')) return 0;
  const m = cond.match(/(\d+)\/(\d+)/);
  return m ? +m[1] / +m[2] : fallback;
}

function parseCondition(cond) {
  const hp = parseHP(cond, 1);
  const status = cond.split(' ')[1];
  return [hp, status && status !== 'fnt' ? (STATUS_BITS[status] ?? 0) : 0];
}

// --- websocket client -----------------------------------------------------------
const battles = new Map();
const ws = new WebSocket(SERVER);
const send = s => ws.send(s);

function actionLabel(req, a) {
  if (a <= 3) return req.active?.[0]?.moves?.[a]?.move ?? `move ${a + 1}`;
  if (a === 9) return 'locked/struggle';
  const p = req.side.pokemon[a - 3];
  return `switch ${p ? p.details.split(',')[0] : a - 2}`;
}

async function act(battle) {
  const req = battle.request;
  if (!req || req.wait) return;
  const {ints, floats, mask, acts} = battle.encode();
  if (!acts.length) return;
  let action = acts[0];
  if (acts.length > 1) {
    try {
      const res = await fetch(`${MODEL}/act`, {
        method: 'POST',
        body: JSON.stringify({battle: battle.room, ints, floats, mask}),
      }).then(r => r.json());
      action = res.action;
      if (!mask[action]) action = acts[0];
      // narrate the decision: options by model probability + win estimate
      const ranked = acts.map(a => [a, res.probs?.[a] ?? 0]).sort((x, y) => y[1] - x[1]);
      const opts = ranked.map(([a, p]) =>
        `${a === action ? '>' : ''}${actionLabel(req, a)} ${(p * 100).toFixed(0)}%`).join(', ');
      const line = `T${battle.turn} [win ${((res.value + 1) * 50).toFixed(0)}%] ${opts}`;
      console.log(`${battle.room} | ${line}`);
      if (args.narrate) send(`|/pm ${args.narrate}, ${line.slice(0, 240)}`);
    } catch (err) {
      console.error('model server error:', err.message);
    }
  }
  const choice = action <= 3 ? `move ${action + 1}`
    : action === 9 ? 'move 1' : `switch ${action - 2}`;
  send(`${battle.room}|/choose ${choice}|${req.rqid ?? ''}`);
}

let loggedIn = false;
setTimeout(() => {
  if (!loggedIn) {
    console.error('login did not complete within 30s (login server slow/unreachable?) - exiting');
    process.exit(2); // nonzero so ladder.sh auto-resume retries
  }
}, 30000);

async function fetchText(url, opts = {}, tries = 3) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, {...opts, signal: AbortSignal.timeout(10000)});
      return await res.text();
    } catch (err) {
      console.error(`login-server fetch failed (${i + 1}/${tries}): ${err.message}`);
      await new Promise(r => setTimeout(r, 2000 * (i + 1)));
    }
  }
  console.error('login server unreachable - exiting');
  process.exit(2);
}

async function handleLine(room, line) {
  if (!line.startsWith('|')) return;
  const parts = line.slice(1).split('|');
  const [cmd] = parts;
  if (cmd === 'challstr') {
    const challstr = parts.slice(1).join('|');
    if (args.pass) {
      const res = await fetchText('https://play.pokemonshowdown.com/action.php', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `act=login&name=${encodeURIComponent(NAME)}&pass=${encodeURIComponent(args.pass)}&challstr=${challstr}`,
      });
      const assertion = JSON.parse(res.slice(1)).assertion;
      if (!assertion) {
        console.error('login failed (bad credentials?):', res.slice(0, 120));
        process.exit(2);
      }
      send(`|/trn ${NAME},0,${assertion}`);
    } else if (/psim\.us|pokemonshowdown\.com/.test(SERVER)) {
      // official servers: even guest names need a login-server assertion
      const res = await fetchText(
        `https://play.pokemonshowdown.com/action.php?act=getassertion&userid=${norm(NAME)}&challstr=${encodeURIComponent(challstr)}`);
      if (res.startsWith(';')) {
        console.error(`name '${NAME}' is registered; use --pass - exiting`);
        process.exit(2);
      }
      if (!res || res.startsWith('<') || res.length < 32) {
        console.error('login server returned garbage (rate limit/challenge?):',
          JSON.stringify(res.slice(0, 120)));
        process.exit(2);
      }
      send(`|/trn ${NAME},0,${res}`);
    } else {
      send(`|/trn ${NAME},0,`); // local server without security
    }
  } else if (cmd === 'updatesearch') {
    try {
      const s = JSON.parse(parts.slice(1).join('|'));
      searchingNow = (s.searching ?? []).length > 0;
      const games = s.games ? Object.keys(s.games).length : 0;
      console.log(`search state: searching=${searchingNow} activeGames=${games} remaining=${searches}`);
    } catch { /* ignore */ }
  } else if (cmd === 'popup' || cmd === 'nametaken') {
    console.error(`server says: |${cmd}|`, parts.slice(1).join('|').slice(0, 200));
    if (!loggedIn) process.exit(2);
  } else if (cmd === 'updateuser' && parts[1].trim().replace(/^[!@#$%^&*+~ ]/, '') === NAME) {
    loggedIn = true;
    console.log(`logged in as ${NAME}`);
    if (searches > 0) send(`|/search ${FORMAT}`);
    if (args.challenge) send(`|/challenge ${args.challenge}, ${FORMAT}`);
  } else if (cmd === 'pm' && parts[3]?.startsWith('/challenge')) {
    const who = parts[1].replace(/[^A-Za-z0-9]/g, ''); // strip rank prefix -> userid
    if (args.accept !== undefined && parts[3].includes(FORMAT)) {
      console.log(`accepting challenge from ${who}`);
      send(`|/accept ${who}`);
    }
  } else if (cmd === 'init' && parts[1] === 'battle') {
    battles.set(room, new Battle(room, 'p1'));
    const http = SERVER.replace(/^ws(s?):\/\//, 'http$1://').replace(/\/showdown\/websocket$/, '');
    const url = /psim\.us|pokemonshowdown\.com/.test(SERVER)
      ? `https://play.pokemonshowdown.com/${room}` : `${http}/${room}`;
    console.log(`battle started: ${room}\n  >> spectate live: ${url}`);
    if (args.timer) send(`${room}|/timer on`);
  } else if (battles.has(room)) {
    const b = battles.get(room);
    if (cmd === 'player' && parts[2]?.trim() === NAME) {
      b.my = parts[1];
    } else if (cmd === 'request' && parts[1]) {
      b.request = JSON.parse(parts[1]);
      setTimeout(() => act(b), 300); // let this message block finish processing
    } else if (cmd === 'error') {
      console.error(room, line);
      const {acts} = b.encode();
      if (acts.length) {
        const a = acts[Math.floor(Math.random() * acts.length)];
        const c = a <= 3 ? `move ${a + 1}` : a === 9 ? 'move 1' : `switch ${a - 2}`;
        send(`${b.room}|/choose ${c}|${b.request?.rqid ?? ''}`);
      }
    } else if (cmd === 'win' || cmd === 'tie') {
      console.log(`${room}: ${cmd} ${parts[1] ?? ''}`);
      fetch(`${MODEL}/end`, {method: 'POST', body: JSON.stringify({battle: room})}).catch(() => {});
      send(`${room}|gg`);
      send(`${room}|/savereplay`); // permanent replay link (official servers)
      setTimeout(() => {
        send(`|/leave ${room}`);
        battles.delete(room);
        if (--searches > 0) send(`|/search ${FORMAT}`); // polite re-queue
      }, 5000);
    } else if (cmd === 'raw' && /rating|Ladder update/i.test(parts.join('|'))) {
      console.log(`${room} | ${parts.join('|').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()}`);
    } else {
      b.handle(parts);
    }
  }
}

let searchingNow = false;
// watchdog: heal dropped ladder searches (server restarts, races, cancellations)
setInterval(() => {
  if (loggedIn && searches > 0 && battles.size === 0 && !searchingNow) {
    console.log('watchdog: no active search or battle - re-issuing /search');
    send(`|/search ${FORMAT}`);
  }
}, 60000);

ws.on('open', () => console.log(`connected to ${SERVER}`));
ws.on('message', data => {
  const msg = data.toString();
  let room = '';
  for (const line of msg.split('\n')) {
    if (line.startsWith('>')) room = line.slice(1);
    else handleLine(room, line).catch(err => console.error(err));
  }
});
ws.on('close', () => {
  console.log('disconnected');
  process.exit(0);
});
