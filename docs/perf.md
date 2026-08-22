# Training performance: measurements, diagnosis, roadmap

*(H100 80GB, d384 model: 3 encoder + 6 transformer layers, heads=6; 4096 envs;
PPO trainer `python/train_fast.py`. Numbers from runs #10/exp-A, 2026-08.)*

## Measurements

| Quantity | Value | How measured |
|---|---|---|
| MFU | ~3-5% | model FLOPs / iter time vs H100 peak |
| Launch-serialization cost | **+18%** (4.51 -> 5.33 s/iter) | `CUDA_LAUNCH_BLOCKING=1` A/B, 12 iters, median of steady-state deltas |
| Iter split (mirror mode) | collect ~1.4s + train ~1.8s | per-phase timers in logs |
| Iter split (exploit/league mode) | collect ~2.4s + train ~1.8s | frozen-opponent fwd for every p1 step, and only p0 makes rows |
| Collect throughput | 106k rows/s (mirror) / 40-70k (exploit) | log `rows/s` |

## Diagnosis

**Not launch-bound** (the +18% test is decisive: truly launch-bound workloads
balloon 2-5x under serialization). **Not FLOPs-bound** (3-5% MFU). We are
**latency/host-bound**:

1. At d384 the GEMMs are too small to saturate SMs; each kernel is internally
   memory/latency-bound. cuBLAS is not the problem; the *shapes* are.
2. The collect loop is Python-driven with per-step `.cpu().numpy()` syncs,
   numpy mask logic, and env round-trips; the GPU idles between steps.
3. MFU is structurally low because the model is small relative to the card —
   a design choice (d512 shelved until the era corpus lands), not waste.
   **KPI should be s/iter and env-steps/s/GPU, not MFU.**

## Roadmap (ranked by expected wall-clock ROI)

1. **Async collect/train overlap** (two streams, double-buffered slab).
   Iter -> max(collect, train). In league mode: 4.2 -> ~2.4 s/iter (-43%).
   Cost: one-iteration data staleness (PPO IS-ratios already handle it; we
   run 2 epochs over the slab anyway). Pure PyTorch engineering.
2. **Stacked-forward collect** for league/exploit modes: learner-inference and
   all frozen opponents share the arch -> stack their params (`torch.func`
   functional_call + vmap, or manual bmm over a leading model dim) and run
   **one** fused forward for every side needing an action. Turns N small
   forwards into 1 at N-fold batch; also the clean answer for Phase-B
   multi-league (N members without N dispatch streams). Est. collect -30-50%
   in league mode.
3. **torch.compile (mode="reduce-overhead")** on the train phase. Fuses the
   PPO elementwise soup, kills Python dispatch; needs shape bucketing (pad
   token-budget batches to fixed sizes). Est. train 1.8 -> ~1.0s (hidden
   entirely once #1 lands, but still saves GPU-seconds).
4. **envs 4096 -> 8192**: bigger batches = bigger GEMMs = the honest MFU
   increase at fixed d. Memory fits after bf16 caches (league cache at 4096
   envs measured 63.5GB total).
5. **CUDA-graph the collect step** — blocked by `ActorCache.make_room`'s
   dynamic `torch.roll` on full windows. Fix = ring-buffer cache (write pos =
   len % CTX) + rotated causal mask. This is the one place a **custom kernel
   is justified**: a varlen ring-buffer KV append+SDPA step. Real project,
   real teeth, transfers to any incremental-inference workload.
6. Fused masked-categorical sample+logprob (~6 kernels -> 1), fused 12-block
   mon embedding. Only after 1-5; est. +10-25% total.

Combined estimate for league-era runs: **4.2 -> ~1.5-2.0 s/iter (~2.5x)**
without any custom CUDA (items 1-4); items 5-6 push toward ~1.2s.

## MLE assignment menu

- Starter (high ROI, kernel-adjacent): items 1+3 — streams, double buffering,
  compile shape-bucketing.
- Core (novel, reusable): item 2 stacked multi-model forward; item 5
  ring-buffer cache + graph capture (+ the custom step kernel).
- Scale note: all fusion work compounds when d512+ returns with the era
  corpus — bigger shapes make every fused op matter more.
