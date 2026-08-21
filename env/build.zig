const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    // Engine in Pokémon Showdown compatibility mode. Logging is enabled since
    // obs v3: the env walks each update's binary protocol for hit/miss/crit
    // event flags (we are GPU-bound by ~100x, the logging cost is irrelevant).
    const pkmn = b.dependency("pkmn", .{ .showdown = true, .log = true });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "pkmn", .module = pkmn.module("pkmn") }},
        .link_libc = true,
    });

    const lib = b.addLibrary(.{
        .name = "gen1env",
        .root_module = mod,
        .linkage = .dynamic,
    });
    b.installArtifact(lib);

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run env tests").dependOn(&run_tests.step);
}
