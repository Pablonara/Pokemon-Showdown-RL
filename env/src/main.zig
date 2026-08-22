//! Vectorized Gen 1 Random Battle environment over libpkmn (showdown mode).
//!
//! Action space (fixed, 10): 0-3 = move slot 1-4, 4-8 = switch to party slot
//! 2-6, 9 = locked move/Struggle (the engine's `move 0`). `pass` is never
//! exposed: any step where a player has <= 1 legal choice is auto-resolved
//! internally, so policies only ever see states with >= 2 legal actions.
//!
//! Masking contract (safety-critical: an illegal choice is UB in the engine):
//!   - masks are built from the engine's own `choices()` (single source of truth)
//!   - `step` re-derives the legal set and panics on any unmasked action
//!
//! Two observation variants per player are exposed:
//!   ints/floats     ground truth (belief-head targets, oracle debugging)
//!   m_ints/m_floats identical layout, opponent info masked to ladder-visible:
//!     unrevealed mons zeroed (except hp_frac=1: bench mons are provably
//!     undamaged in gen1), unseen moves zeroed, opponent pp/exact stats/sleep
//!     counter/last-selected-move hidden. Public info (active species, status,
//!     boosts, volatiles, last-used move) stays. NOTE: a selected-but-blocked
//!     move is currently marked revealed (slight over-reveal; TODO).
//! Obs layout v1 (per player, perspective-flipped: "mine" first):
//!   ints  [80]: 2 sides x 6 mons x [species, status, move1..4]  (72)
//!               my/their active volatiles (low 32 bits)          (2)
//!               my/their active boosts (raw u32)                 (2)
//!               my/their last_selected_move, last_used_move      (4)
//!   floats[160]: 2 sides x 6 mons x [hp_frac, level/100, active,
//!                revealed, pp1..4 / 63]                          (96)
//!               my/their active boosts / 6                       (12)
//!               my/their active volatile flags (18 each)         (36)
//!               my/their active sleep turns / 7                  (2)
//!               my/their active modified stats / 1000            (8)
//!               turn / 500                                       (1)
//!               padding                                          (5)

const std = @import("std");
const pkmn = @import("pkmn");

const gen1 = pkmn.gen1;
const Choice = pkmn.Choice;
const Player = pkmn.Player;
const Result = pkmn.Result;

pub const VERSION: u32 = 5;

/// Opponent hp as the ladder shows it: integer percent, min 1 while alive
/// (same integer math as PS getHealth).
fn pubHpPct(hp: u16, maxhp: u16) f32 {
    if (hp == 0 or maxhp == 0) return 0;
    const pct = @max(1, @as(u32, hp) * 100 / maxhp);
    return @as(f32, @floatFromInt(pct)) / 100.0;
}

/// Opponent max hp as the deployment bot estimates it (identical formula):
/// randbats spread ev=63/iv=30; exact for standard sets, +-2hp on tweaked.
fn estMaxHP(species_num: i32, level: u8) f32 {
    if (species_num <= 0) return 0;
    const base: u32 = gen1.Species.get(@enumFromInt(@as(u8, @intCast(species_num)))).stats.hp;
    return @floatFromInt((2 * base + 30 + 63 + 100) * @as(u32, level) / 100 + 10);
}

/// Ladder-visible status byte: the engine's sleep countdown (low 3 bits) is
/// server-side hidden RNG, so any sleep is canonicalized to a fixed "asleep,
/// duration unknown" value (4). Turns-elapsed is countable from sequence
/// memory. TOX stays raw (public). Deployment bot writes the same values.
fn pubStatus(num: u8) u8 {
    if (num == gen1.Status.TOX) return num;
    if (gen1.Status.duration(num) > 0) return 4;
    return num;
}
pub const N_ACTIONS: u32 = 10;
pub const INTS_PER_PLAYER: u32 = 80;
// obs v3: 12 mons x 13 floats [hp, level, active, revealed, pp1-4, maxhp/1000,
// used1-4/16] = 156, + extras [boosts 6 + volatile flags 18] x2 = 48,
// + [sleep counter + 4 modified stats] x2 = 10, + turn = 1, + last-move event
// flags [attempted, missed, crit] mine/theirs = 6, + pad = 224
pub const FLOATS_PER_PLAYER: u32 = 224;
const MON_F: usize = 13;
const MAX_TURNS: u16 = 1000; // Endless Battle Clause surrogate: draw
const ACTION_SPECIAL: i32 = 9; // engine `move 0`: locked move / Struggle

const BattleT = gen1.Battle(gen1.PRNG);
const VOLATILE_FLAGS = [_][]const u8{
    "Bide",       "Thrashing",  "MultiHit",   "Flinch",    "Charging",  "Binding",
    "Invulnerable", "Confusion", "Mist",      "FocusEnergy", "Substitute", "Recharging",
    "Rage",       "LeechSeed",  "Toxic",      "LightScreen", "Reflect",  "Transform",
};

fn splitmix64(x: u64) u64 {
    var z = x +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// --- team pool ---------------------------------------------------------------

const Pool = struct {
    // [species, level, move1..4, ev hp/atk/def/spa/spd/spe, iv hp/atk/def/spa/spd/spe]
    const MON_BYTES = 18;
    const TEAM_BYTES = 6 * MON_BYTES;

    teams: []const u8, // count * TEAM_BYTES
    count: u32,
    owned: bool,

    fn load(alloc: std.mem.Allocator, path: [*:0]const u8) !Pool {
        const f = std.c.fopen(path, "rb") orelse return error.FileNotFound;
        defer _ = std.c.fclose(f);
        var header: [12]u8 = undefined;
        if (std.c.fread(&header, 1, 12, f) != 12) return error.BadHeader;
        if (!std.mem.eql(u8, header[0..4], "G1PK") or header[4] != 2) return error.BadMagic;
        const count = std.mem.readInt(u32, header[8..12], .little);
        if (count == 0) return error.EmptyPool;
        const bytes = try alloc.alloc(u8, @as(usize, count) * TEAM_BYTES);
        errdefer alloc.free(bytes);
        if (std.c.fread(bytes.ptr, 1, bytes.len, f) != bytes.len) return error.Truncated;
        return .{ .teams = bytes, .count = count, .owned = true };
    }

    fn team(self: *const Pool, idx: u64) *const [TEAM_BYTES]u8 {
        const i: usize = @intCast(idx % self.count);
        return self.teams[i * TEAM_BYTES ..][0..TEAM_BYTES];
    }

    /// Builds the 6 helper Pokemon for one packed team record. `moves` is
    /// backing storage that must outlive the returned slice's use.
    fn unpack(rec: *const [TEAM_BYTES]u8, out: *[6]gen1.helpers.Pokemon, moves: *[6][4]gen1.Move) []const gen1.helpers.Pokemon {
        for (0..6) |j| {
            const mon = rec[j * MON_BYTES ..][0..MON_BYTES];
            var n: usize = 0;
            for (0..4) |k| {
                if (mon[2 + k] == 0) break;
                moves[j][n] = @enumFromInt(mon[2 + k]);
                n += 1;
            }
            // evs at [6..12], ivs at [12..18], order hp/atk/def/spa/spd/spe.
            // PS uses the modern formula (2*base + iv + ev/4) in gen 1; with
            // even ivs (asserted by make_pool) it equals the engine's cartridge
            // form given dv = iv >> 1 and statExp = ev^2 (ceil(sqrt(ev^2)) = ev).
            const exp = struct {
                fn of(ev: u8) u16 {
                    return @as(u16, ev) * ev;
                }
            }.of;
            out[j] = .{
                .species = @enumFromInt(mon[0]),
                .level = mon[1],
                .moves = moves[j][0..n],
                .dvs = .{
                    .atk = @intCast(mon[13] >> 1),
                    .def = @intCast(mon[14] >> 1),
                    .spe = @intCast(mon[17] >> 1),
                    .spc = @intCast(mon[15] >> 1),
                },
                .stats = .{
                    .hp = exp(mon[6]),
                    .atk = exp(mon[7]),
                    .def = exp(mon[8]),
                    .spe = exp(mon[11]),
                    .spc = exp(mon[9]),
                },
            };
        }
        return out[0..6];
    }
};

// --- environment -------------------------------------------------------------

const EnvState = struct {
    battle: BattleT,
    result: Result,
    battle_idx: u64, // global battle counter (teams 2k, 2k+1; seed f(env_seed, k))
    revealed: [2][6]bool, // by original party index (order-invariant)
    reveal_seq: [2][6]u8, // mon ids in first-reveal order (ladder-visible order)
    reveal_n: [2]u8,
    usage: [2][6][4]u8, // move selection counts (masking: >0 = revealed)
    events: [2][3]bool, // per player since last expose: [attempted, missed, crit]
    last_move_side: u8, // walker state for LastMiss attribution (2 = none)
};

const OptsT = @TypeOf(pkmn.battle.options(
    @as(pkmn.protocol.FixedLog, undefined),
    pkmn.gen1.chance.NULL,
    pkmn.gen1.calc.NULL,
));

pub const Env = struct {
    alloc: std.mem.Allocator,
    pool: Pool,
    n: u32,
    seed: u64,
    next_battle: u64,
    states: []EnvState,
    log_buf: [pkmn.LOGS_SIZE]u8,
    writer: pkmn.protocol.Writer,
    opts: OptsT,
    // exposed buffers (Python views these; layout must match gen1env.py)
    actions: []i32, // n*2, written by caller
    masks: []u8, // n*2*N_ACTIONS
    needs: []u8, // n*2
    ints: []i32, // n*2*INTS_PER_PLAYER (ground truth)
    floats: []f32, // n*2*FLOATS_PER_PLAYER (ground truth)
    m_ints: []i32, // masked variant, same layout
    m_floats: []f32,
    rewards: []f32, // n*2 (nonzero only on done)
    dones: []u8, // n
    ep_turns: []u16, // n, turns of the episode that ended (valid when done)
    ep_idx: []u64, // n, global battle index of the episode that ended (pairing)
    episodes: u64,

    fn create(alloc: std.mem.Allocator, pool_path: [*:0]const u8, n: u32, seed: u64) !*Env {
        const env = try alloc.create(Env);
        errdefer alloc.destroy(env);
        env.* = .{
            .alloc = alloc,
            .pool = try Pool.load(alloc, pool_path),
            .n = n,
            .seed = seed,
            .next_battle = 0,
            .states = try alloc.alloc(EnvState, n),
            .log_buf = undefined,
            .writer = undefined,
            .opts = undefined,
            .actions = try alloc.alloc(i32, n * 2),
            .masks = try alloc.alloc(u8, n * 2 * N_ACTIONS),
            .needs = try alloc.alloc(u8, n * 2),
            .ints = try alloc.alloc(i32, n * 2 * INTS_PER_PLAYER),
            .floats = try alloc.alloc(f32, n * 2 * FLOATS_PER_PLAYER),
            .m_ints = try alloc.alloc(i32, n * 2 * INTS_PER_PLAYER),
            .m_floats = try alloc.alloc(f32, n * 2 * FLOATS_PER_PLAYER),
            .rewards = try alloc.alloc(f32, n * 2),
            .dones = try alloc.alloc(u8, n),
            .ep_turns = try alloc.alloc(u16, n),
            .ep_idx = try alloc.alloc(u64, n),
            .episodes = 0,
        };
        @memset(env.actions, 0);
        @memset(env.rewards, 0);
        @memset(env.dones, 0);
        @memset(env.ep_turns, 0);
        @memset(env.ep_idx, 0);
        // log plumbing: env is heap-pinned, so interior pointers are stable
        env.writer = .{ .buffer = &env.log_buf };
        env.opts = pkmn.battle.options(
            pkmn.protocol.FixedLog{ .writer = &env.writer },
            pkmn.gen1.chance.NULL,
            pkmn.gen1.calc.NULL,
        );
        for (0..n) |i| {
            env.resetBattle(@intCast(i));
            env.advance(@intCast(i));
            env.expose(@intCast(i));
        }
        return env;
    }

    fn destroy(env: *Env) void {
        const alloc = env.alloc;
        if (env.pool.owned) alloc.free(env.pool.teams);
        alloc.free(env.states);
        alloc.free(env.actions);
        alloc.free(env.masks);
        alloc.free(env.needs);
        alloc.free(env.ints);
        alloc.free(env.floats);
        alloc.free(env.m_ints);
        alloc.free(env.m_floats);
        alloc.free(env.rewards);
        alloc.free(env.dones);
        alloc.free(env.ep_turns);
        alloc.free(env.ep_idx);
        alloc.destroy(env);
    }

    fn resetBattle(env: *Env, i: u32) void {
        const k = env.next_battle;
        env.next_battle += 1;
        var p1: [6]gen1.helpers.Pokemon = undefined;
        var p2: [6]gen1.helpers.Pokemon = undefined;
        var m1: [6][4]gen1.Move = undefined;
        var m2: [6][4]gen1.Move = undefined;
        const t1 = Pool.unpack(env.pool.team(k * 2), &p1, &m1);
        const t2 = Pool.unpack(env.pool.team(k * 2 + 1), &p2, &m2);
        const s = &env.states[i];
        s.* = .{
            .battle = gen1.helpers.Battle.init(splitmix64(env.seed ^ k), t1, t2),
            .result = .{},
            .battle_idx = k,
            .revealed = @splat(@splat(false)),
            .reveal_seq = @splat(@splat(0)),
            .reveal_n = @splat(0),
            .usage = @splat(@splat(@splat(0))),
            .events = @splat(@splat(false)),
            .last_move_side = 2,
        };
        s.result = env.update2(s, .{}, .{}); // leads
    }

    /// Ends the current battle with the given rewards and starts a new one.
    fn finish(env: *Env, i: u32, r1: f32, r2: f32) void {
        env.rewards[i * 2] = r1;
        env.rewards[i * 2 + 1] = r2;
        env.dones[i] = 1;
        env.ep_turns[i] = env.states[i].battle.turn;
        env.ep_idx[i] = env.states[i].battle_idx;
        env.episodes += 1;
        env.resetBattle(i);
    }

    fn legal(s: *EnvState, player: Player, buf: []Choice) u8 {
        const request = if (player == .P1) s.result.p1 else s.result.p2;
        return s.battle.choices(player, request, buf);
    }

    /// Auto-resolves all steps where neither player has a real decision
    /// (<= 1 legal choice each), handling terminals along the way. Afterwards
    /// either the battle is waiting on a decision or was reset to one that is.
    fn advance(env: *Env, i: u32) void {
        var s = &env.states[i];
        while (true) {
            if (s.result.type != .None) {
                switch (s.result.type) {
                    .Win => env.finish(i, 1, -1),
                    .Lose => env.finish(i, -1, 1),
                    .Tie => env.finish(i, 0, 0),
                    .Error => std.debug.panic("engine error, battle {d}", .{s.battle_idx}),
                    .None => unreachable,
                }
                s = &env.states[i];
                continue;
            }
            if (s.battle.turn > MAX_TURNS) {
                env.finish(i, 0, 0); // stall guard: draw
                s = &env.states[i];
                continue;
            }
            var buf1: [pkmn.CHOICES_SIZE]Choice = undefined;
            var buf2: [pkmn.CHOICES_SIZE]Choice = undefined;
            const n1 = legal(s, .P1, &buf1);
            const n2 = legal(s, .P2, &buf2);
            if (n1 == 0 or n2 == 0) {
                // gen1 Transform+Mirror Move/Metronome soft-lock (non-showdown
                // builds only; unreachable with -Dshowdown, guarded anyway)
                env.finish(i, 0, 0);
                s = &env.states[i];
                continue;
            }
            if (n1 > 1 or n2 > 1) return; // real decision: expose to policy
            s.result = env.update2(s, buf1[0], buf2[0]);
        }
    }

    /// update + parse the emitted binary protocol for event flags. Byte
    /// lengths transcribed from the driver's decoders (src/pkg/protocol.ts);
    /// unknown opcodes panic (fail loud beats silent corruption).
    fn update2(env: *Env, s: *EnvState, c1: Choice, c2: Choice) Result {
        env.writer.pos = 0;
        const r = s.battle.update(c1, c2, &env.opts) catch unreachable;
        const b = env.log_buf[0..env.writer.pos];
        var i: usize = 0;
        while (i < b.len) {
            const op = b[i];
            i += 1;
            switch (op) {
                0 => break, // None: end of update
                1 => {}, // LastStill
                2 => if (s.last_move_side < 2) { // LastMiss -> mover missed
                    s.events[s.last_move_side][1] = true;
                },
                3 => { // Move [ident, move, target, reason(, from)]
                    const side: u8 = if ((b[i] >> 3) == 0) 0 else 1;
                    s.events[side][0] = true;
                    s.last_move_side = side;
                    // usage counts EXECUTED moves only (= what the ladder
                    // shows). Selection-time counting leaked blocked moves
                    // (asleep/full-para) into the reveal gate. Moves called
                    // by Metronome/Mirror Move match no slot and are skipped.
                    const sd = s.battle.side(@enumFromInt(side));
                    const id = sd.order[0];
                    if (id != 0 and b[i + 1] != 0) {
                        for (0..4) |k| {
                            if (@intFromEnum(sd.active.moves[k].id) == b[i + 1]) {
                                s.usage[side][id - 1][k] +|= 1;
                                break;
                            }
                        }
                    }
                    i += if (b[i + 3] == 1) 5 else 4;
                },
                4 => i += 8, // Switch
                5 => i += if (b[i + 1] == 5) 3 else 2, // Cant (+Disable move)
                6 => i += 1, // Faint
                7 => i += 2, // Turn
                8 => i += 1, // Win
                9 => {}, // Tie
                10 => i += if (b[i + 6] == 5) 8 else 7, // Damage (+RecoilOf)
                11 => i += if (b[i + 6] == 2) 8 else 7, // Heal (+Drain of)
                12 => i += if (b[i + 2] == 2) 4 else 3, // Status (+From move)
                13 => i += 3, // CureStatus
                14 => i += 3, // Boost
                15 => {}, // ClearAllBoost
                16 => i += 2, // Fail
                17 => { // Miss: ident = attacker
                    s.events[if ((b[i] >> 3) == 0) 0 else 1][1] = true;
                    i += 1;
                },
                18, 19 => i += 2, // HitCount, Prepare
                20 => i += 1, // MustRecharge
                21 => i += 2, // Activate
                22 => {}, // FieldActivate
                23 => { // Start (+TypeChange: types+of, +Disable/Mimic: move)
                    const reason = b[i + 1];
                    i += if (reason == 9) 4 else if (reason == 10 or reason == 11) 3 else 2;
                },
                24 => i += 2, // End
                25 => {}, // OHKO
                26 => { // Crit: ident = defender -> credit the attacker
                    s.events[(if ((b[i] >> 3) == 0) @as(u8, 0) else 1) ^ 1][2] = true;
                    i += 1;
                },
                27, 28 => i += 1, // SuperEffective, Resisted
                29 => i += 2, // Immune (+reason)
                30 => i += 2, // Transform
                else => std.debug.panic("unknown protocol opcode {d}", .{op}),
            }
        }
        return r;
    }

    fn actionToChoice(action: i32) Choice {
        return switch (action) {
            0...3 => .{ .type = .Move, .data = @intCast(action + 1) },
            4...8 => .{ .type = .Switch, .data = @intCast(action - 2) },
            ACTION_SPECIAL => .{ .type = .Move, .data = 0 },
            else => std.debug.panic("action {d} out of range", .{action}),
        };
    }

    fn choiceToActionBit(c: Choice) ?u4 {
        return switch (c.type) {
            .Pass => null,
            .Move => if (c.data == 0) @intCast(ACTION_SPECIAL) else @intCast(c.data - 1),
            .Switch => @intCast(c.data + 2),
        };
    }

    fn step(env: *Env) void {
        for (0..env.n) |ui| {
            const i: u32 = @intCast(ui);
            env.rewards[i * 2] = 0;
            env.rewards[i * 2 + 1] = 0;
            env.dones[i] = 0;
            const s = &env.states[i];

            var choice: [2]Choice = .{ .{}, .{} };
            var bufs: [2][pkmn.CHOICES_SIZE]Choice = undefined;
            inline for (.{ Player.P1, Player.P2 }, 0..) |player, p| {
                const nc = legal(s, player, &bufs[p]);
                if (nc == 1) {
                    choice[p] = bufs[p][0];
                } else {
                    const action = env.actions[i * 2 + p];
                    const c = actionToChoice(action);
                    const ok = for (bufs[p][0..nc]) |lc| {
                        if (lc.type == c.type and lc.data == c.data) break true;
                    } else false;
                    if (!ok) std.debug.panic(
                        "illegal action {d} (env {d} player {d} battle {d})",
                        .{ action, i, p, s.battle_idx },
                    );
                    choice[p] = c;
                }
            }
            s.result = env.update2(s, choice[0], choice[1]);
            env.advance(i);
            env.expose(i);
        }
    }

    /// Writes masks/needs and observations for the currently waiting state.
    fn expose(env: *Env, i: u32) void {
        const s = &env.states[i];
        var bufs: [2][pkmn.CHOICES_SIZE]Choice = undefined;
        inline for (.{ Player.P1, Player.P2 }, 0..) |player, p| {
            const mask = env.masks[(i * 2 + p) * N_ACTIONS ..][0..N_ACTIONS];
            @memset(mask, 0);
            const nc = legal(s, player, &bufs[p]);
            env.needs[i * 2 + p] = @intFromBool(nc > 1);
            if (nc > 1) for (bufs[p][0..nc]) |c| {
                if (choiceToActionBit(c)) |bit| mask[bit] = 1;
            };
            // mark both actives revealed (lazily covers every switch-in)
            const side = s.battle.side(player);
            const aid = side.order[0];
            if (aid > 0 and !s.revealed[p][aid - 1]) {
                s.revealed[p][aid - 1] = true;
                s.reveal_seq[p][s.reveal_n[p]] = aid;
                s.reveal_n[p] += 1;
            }
        }
        inline for (.{ Player.P1, Player.P2 }, 0..) |player, p| {
            env.encode(i, player, p);
            env.maskObs(i, player, p);
        }
        s.events = @splat(@splat(false)); // consumed by this expose
        s.last_move_side = 2;
    }

    // Layout landmarks for maskObs (must match encode(); see header comment):
    // opponent mon int blocks 36..72; opponent mon float blocks 78..156
    // (6 x MON_F); int 78 = opponent last_selected_move; floats 209..214 =
    // opponent sleep-counter + modified stats. Event flags/usage counts are
    // public. maxhp of revealed opponents is public-computable in randbats.
    const OPP_LSM_INT = 78;
    const OPP_PRIVATE_F = [2]usize{ 209, 214 };

    /// Foe mon-block order as the ladder client sees it: active first, then
    /// first-reveal order; unrevealed trail in engine order (belief targets
    /// only - their masked blocks are zeroed).
    fn foeOrder(s: *const EnvState, side: *const gen1.Side, opp: usize) [6]u8 {
        var out: [6]u8 = @splat(0);
        var n: usize = 0;
        const active = side.order[0];
        if (active != 0) {
            out[0] = active;
            n = 1;
        }
        for (0..s.reveal_n[opp]) |k| {
            const id = s.reveal_seq[opp][k];
            if (id != active) {
                out[n] = id;
                n += 1;
            }
        }
        outer: for (side.order) |id| {
            if (id == 0) continue;
            for (out[0..n]) |x| {
                if (x == id) continue :outer;
            }
            out[n] = id;
            n += 1;
        }
        return out;
    }

    fn maskObs(env: *Env, i: u32, player: Player, p: usize) void {
        const base = (i * 2 + @as(u32, @intCast(p)));
        const ti = env.ints[base * INTS_PER_PLAYER ..][0..INTS_PER_PLAYER];
        const tf = env.floats[base * FLOATS_PER_PLAYER ..][0..FLOATS_PER_PLAYER];
        const mi = env.m_ints[base * INTS_PER_PLAYER ..][0..INTS_PER_PLAYER];
        const mf = env.m_floats[base * FLOATS_PER_PLAYER ..][0..FLOATS_PER_PLAYER];
        @memcpy(mi, ti);
        @memcpy(mf, tf);

        const s = &env.states[i];
        const opp: usize = @intFromEnum(player.foe());
        const side = s.battle.side(player.foe());
        const ord = foeOrder(s, side, opp);
        for (0..6) |j| {
            const ii = 36 + j * 6;
            const fi = 6 * MON_F + j * MON_F;
            const id = ord[j];
            if (id == 0) continue;
            if (!s.revealed[opp][id - 1]) {
                @memset(mi[ii .. ii + 6], 0);
                @memset(mf[fi .. fi + MON_F], 0);
                mf[fi] = 1.0; // bench mons are undamaged in gen1
            } else {
                for (0..4) |k| {
                    if (s.usage[opp][id - 1][k] == 0) mi[ii + 2 + k] = 0;
                }
                @memset(mf[fi + 4 .. fi + 8], 0); // exact pp hidden
                const mon = &side.pokemon[id - 1];
                mf[fi] = pubHpPct(mon.hp, mon.stats.hp); // integer-percent hp
                mf[fi + 8] = estMaxHP(mi[ii], mon.level) / 1000.0;
            }
        }
        mi[OPP_LSM_INT] = 0;
        @memset(mf[OPP_PRIVATE_F[0]..OPP_PRIVATE_F[1]], 0);
    }

    fn encode(env: *Env, i: u32, player: Player, p: usize) void {
        const s = &env.states[i];
        const ints = env.ints[(i * 2 + p) * INTS_PER_PLAYER ..][0..INTS_PER_PLAYER];
        const floats = env.floats[(i * 2 + p) * FLOATS_PER_PLAYER ..][0..FLOATS_PER_PLAYER];
        @memset(ints, 0);
        @memset(floats, 0);

        var ii: usize = 0;
        var fi: usize = 0;
        const sides = [2]*const gen1.Side{ s.battle.side(player), s.battle.foe(player) };
        const pidx = [2]usize{ @intFromEnum(player), @intFromEnum(player.foe()) };

        for (sides, pidx, 0..) |side, sp, si| {
            // foe blocks in ladder-visible reveal order (true+masked stay
            // block-aligned for the belief targets)
            const ord: ?[6]u8 = if (si == 1) foeOrder(s, side, sp) else null;
            for (1..7) |slot| {
                const id = if (ord) |o| o[slot - 1] else side.order[slot - 1];
                if (id == 0) { // absent (never in randbats; keep layout stable)
                    ii += 6;
                    fi += MON_F;
                    continue;
                }
                const mon = &side.pokemon[id - 1];
                const is_active = slot == 1;
                ints[ii] = @intFromEnum(mon.species);
                ints[ii + 1] = pubStatus(mon.status);
                const moves = if (is_active) &side.active.moves else &mon.moves;
                for (0..4) |k| ints[ii + 2 + k] = @intFromEnum(moves[k].id);
                ii += 6;
                const max_hp: f32 = @floatFromInt(mon.stats.hp);
                floats[fi] = if (max_hp > 0) @as(f32, @floatFromInt(mon.hp)) / max_hp else 0;
                floats[fi + 1] = @as(f32, @floatFromInt(mon.level)) / 100.0;
                floats[fi + 2] = @floatFromInt(@intFromBool(is_active));
                floats[fi + 3] = @floatFromInt(@intFromBool(s.revealed[sp][id - 1]));
                for (0..4) |k| floats[fi + 4 + k] = @as(f32, @floatFromInt(moves[k].pp)) / 63.0;
                floats[fi + 8] = max_hp / 1000.0;
                for (0..4) |k| {
                    floats[fi + 9 + k] =
                        @min(1.0, @as(f32, @floatFromInt(s.usage[sp][id - 1][k])) / 16.0);
                }
                fi += MON_F;
            }
        }
        std.debug.assert(ii == 72 and fi == 12 * MON_F);

        for (sides) |side| {
            const active = &side.active;
            // flags only (bits 0-17): higher bits hold hidden counters
            // (confusion/binding turns = server RNG, bide accumulator)
            ints[ii] = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(active.volatiles)))) & 0x3FFFF);
            ii += 1;
            inline for (.{ "atk", "def", "spe", "spc", "accuracy", "evasion" }) |b| {
                floats[fi] = @as(f32, @floatFromInt(@field(active.boosts, b))) / 6.0;
                fi += 1;
            }
            inline for (VOLATILE_FLAGS) |v| {
                floats[fi] = @floatFromInt(@intFromBool(@field(active.volatiles, v)));
                fi += 1;
            }
        }
        for (sides) |side| {
            ints[ii] = @bitCast(@as(u32, @bitCast(side.active.boosts)));
            ints[ii + 1] = @intFromEnum(side.last_selected_move);
            ints[ii + 2] = @intFromEnum(side.last_used_move);
            ii += 3;
            floats[fi] = 0.0; // was sleep countdown: hidden RNG, not ladder-visible
            fi += 1;
            inline for (.{ "atk", "def", "spe", "spc" }) |st| {
                floats[fi] = @as(f32, @floatFromInt(@field(side.active.stats, st))) / 1000.0;
                fi += 1;
            }
        }
        floats[fi] = @as(f32, @floatFromInt(s.battle.turn)) / 500.0;
        fi += 1;
        for (pidx) |sp| { // event flags: [attempted, missed, crit] mine then theirs
            for (0..3) |e| {
                floats[fi] = @floatFromInt(@intFromBool(s.events[sp][e]));
                fi += 1;
            }
        }
        std.debug.assert(ii <= INTS_PER_PLAYER and fi <= FLOATS_PER_PLAYER);
    }
};

// --- C ABI --------------------------------------------------------------------

const Buffers = extern struct {
    actions: [*]i32,
    masks: [*]u8,
    needs: [*]u8,
    ints: [*]i32,
    floats: [*]f32,
    m_ints: [*]i32,
    m_floats: [*]f32,
    rewards: [*]f32,
    dones: [*]u8,
    ep_turns: [*]u16,
    ep_idx: [*]u64,
    n: u32,
    n_actions: u32,
    ints_per_player: u32,
    floats_per_player: u32,
};

export fn g1_version() u32 {
    return VERSION;
}

export fn g1_create(pool_path: [*:0]const u8, n: u32, seed: u64) ?*Env {
    if (n == 0) return null;
    return Env.create(std.heap.c_allocator, pool_path, n, seed) catch |err| {
        std.debug.print("g1_create failed: {}\n", .{err});
        return null;
    };
}

export fn g1_buffers(env: *Env, out: *Buffers) void {
    out.* = .{
        .actions = env.actions.ptr,
        .masks = env.masks.ptr,
        .needs = env.needs.ptr,
        .ints = env.ints.ptr,
        .floats = env.floats.ptr,
        .m_ints = env.m_ints.ptr,
        .m_floats = env.m_floats.ptr,
        .rewards = env.rewards.ptr,
        .dones = env.dones.ptr,
        .ep_turns = env.ep_turns.ptr,
        .ep_idx = env.ep_idx.ptr,
        .n = env.n,
        .n_actions = N_ACTIONS,
        .ints_per_player = INTS_PER_PLAYER,
        .floats_per_player = FLOATS_PER_PLAYER,
    };
}

export fn g1_step(env: *Env) void {
    env.step();
}

export fn g1_episodes(env: *Env) u64 {
    return env.episodes;
}

/// Test hook: raw stats of a stored Pokémon [hp, atk, def, spe, spc].
export fn g1_debug_stats(env: *Env, i: u32, side: u32, slot: u32, out: *[5]u16) void {
    const s = env.states[i].battle.side(if (side == 0) .P1 else .P2);
    const mon = &s.pokemon[s.order[slot] - 1];
    out.* = .{ mon.stats.hp, mon.stats.atk, mon.stats.def, mon.stats.spe, mon.stats.spc };
}

export fn g1_destroy(env: *Env) void {
    env.destroy();
}

// --- tests ---------------------------------------------------------------------

const testing = std.testing;

fn testPool(alloc: std.mem.Allocator) !Pool {
    // Two identical classic teams (cleric spreads)
    var bytes = try alloc.alloc(u8, 2 * Pool.TEAM_BYTES);
    const mons = [_][6]u8{
        .{ 128, 78, 34, 63, 59, 89 }, // Tauros: Body Slam, Hyper Beam, Blizzard, Earthquake
        .{ 113, 82, 85, 86, 135, 105 }, // Chansey: Thunderbolt, Thunder Wave, Soft-Boiled, Recover
        .{ 143, 76, 34, 156, 89, 58 }, // Snorlax: Body Slam, Rest, Earthquake, Ice Beam
        .{ 121, 73, 105, 86, 59, 85 }, // Starmie: Recover, TWave, Blizzard, Thunderbolt
        .{ 103, 74, 76, 94, 153, 79 }, // Exeggutor: Stun Spore, Psychic, Explosion, Sleep Powder
        .{ 65, 71, 94, 105, 86, 100 }, // Alakazam: Psychic, Recover, TWave, Teleport
    };
    for (0..2) |t| {
        for (0..6) |j| {
            const o = t * Pool.TEAM_BYTES + j * Pool.MON_BYTES;
            @memcpy(bytes[o..][0..6], &mons[j]);
            @memset(bytes[o + 6 ..][0..6], 255); // evs
            @memset(bytes[o + 12 ..][0..6], 30); // ivs
        }
    }
    return .{ .teams = bytes, .count = 2, .owned = true };
}

test "env: fuzz random-legal actions" {
    const alloc = testing.allocator;
    var pool = try testPool(alloc);
    const env = try alloc.create(Env);
    defer env.destroy();
    env.* = .{
        .alloc = alloc,
        .pool = pool,
        .n = 8,
        .seed = 0x1234,
        .next_battle = 0,
        .states = try alloc.alloc(EnvState, 8),
        .log_buf = undefined,
        .writer = undefined,
        .opts = undefined,
        .actions = try alloc.alloc(i32, 16),
        .masks = try alloc.alloc(u8, 16 * N_ACTIONS),
        .needs = try alloc.alloc(u8, 16),
        .ints = try alloc.alloc(i32, 16 * INTS_PER_PLAYER),
        .floats = try alloc.alloc(f32, 16 * FLOATS_PER_PLAYER),
        .m_ints = try alloc.alloc(i32, 16 * INTS_PER_PLAYER),
        .m_floats = try alloc.alloc(f32, 16 * FLOATS_PER_PLAYER),
        .rewards = try alloc.alloc(f32, 16),
        .dones = try alloc.alloc(u8, 8),
        .ep_turns = try alloc.alloc(u16, 8),
        .ep_idx = try alloc.alloc(u64, 8),
        .episodes = 0,
    };
    pool.owned = false; // env now owns the bytes via destroy()
    env.writer = .{ .buffer = &env.log_buf };
    env.opts = pkmn.battle.options(
        pkmn.protocol.FixedLog{ .writer = &env.writer },
        pkmn.gen1.chance.NULL,
        pkmn.gen1.calc.NULL,
    );
    @memset(env.actions, 0);
    @memset(env.rewards, 0);
    @memset(env.dones, 0);
    @memset(env.ep_turns, 0);
    @memset(env.ep_idx, 0);
    for (0..8) |i| {
        env.resetBattle(@intCast(i));
        env.advance(@intCast(i));
        env.expose(@intCast(i));
    }

    var prng = std.Random.DefaultPrng.init(99);
    const random = prng.random();
    var steps: usize = 0;
    while (env.episodes < 200) : (steps += 1) {
        for (0..env.n * 2) |j| {
            if (env.needs[j] == 0) continue;
            const mask = env.masks[j * N_ACTIONS ..][0..N_ACTIONS];
            var count: usize = 0;
            for (mask) |m| count += m;
            try testing.expect(count >= 2); // exposed states always have a real decision
            var pick = random.uintLessThan(usize, count);
            for (mask, 0..) |m, a| {
                if (m == 0) continue;
                if (pick == 0) {
                    env.actions[j] = @intCast(a);
                    break;
                }
                pick -= 1;
            }
        }
        env.step();
        for (0..env.n) |i| {
            if (env.dones[i] == 1) {
                const r = env.rewards[i * 2] + env.rewards[i * 2 + 1];
                try testing.expectEqual(@as(f32, 0), r); // zero-sum
            }
        }
        // masked-obs invariants for a sample player row
        const b: usize = 3; // env 1, player 1
        const ti = env.ints[b * INTS_PER_PLAYER ..][0..INTS_PER_PLAYER];
        const mi = env.m_ints[b * INTS_PER_PLAYER ..][0..INTS_PER_PLAYER];
        const mfl = env.m_floats[b * FLOATS_PER_PLAYER ..][0..FLOATS_PER_PLAYER];
        try testing.expectEqualSlices(i32, ti[0..36], mi[0..36]); // own side intact
        try testing.expectEqual(@as(i32, 0), mi[Env.OPP_LSM_INT]);
        for (0..6) |j| {
            const fi = 6 * MON_F + j * MON_F;
            const revealed = mfl[fi + 3];
            if (revealed == 0 and ti[36 + j * 6] != 0) {
                try testing.expectEqual(@as(i32, 0), mi[36 + j * 6]); // species hidden
                try testing.expectEqual(@as(f32, 1.0), mfl[fi]); // full hp known
            }
        }
        for (215..221) |fx| { // event flags are binary
            try testing.expect(mfl[fx] == 0.0 or mfl[fx] == 1.0);
        }
        try testing.expect(steps < 200_000);
    }
}

test "action mapping is a bijection on legal encodings" {
    for (0..N_ACTIONS) |a| {
        const c = Env.actionToChoice(@intCast(a));
        try testing.expectEqual(@as(u4, @intCast(a)), Env.choiceToActionBit(c).?);
    }
    try testing.expectEqual(null, Env.choiceToActionBit(.{}));
}
