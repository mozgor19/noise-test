# Session Handoff

This file is the first file an AI assistant should read when continuing work.

It should be updated after meaningful progress, before stopping, and before switching tools.

---

## Last updated
2026-10-11

## Current goal
Fix the audit findings of the S&P denoising comparison study (paper), critical-first, step by step.
User rules: never add Claude as co-author in commits/PRs; after each step show an architecture diagram
(used vs. legacy files) and explain the logic of the step in Turkish.

---

## Current status

### Done (Step 1 — DL pipeline, code)
- Noise model unified to channel-wise (MATLAB `imnoise` equivalent): `utils/spmask.py::add_sp_noise`.
- Blind channel mask `sp_mask_channel` (value 0/255 per channel) used in train + infer (recall 100%).
- DnCNN-SP input = RGB + 3-ch mask (6 ch), per-channel gating.
- SeConvUNet: real partial conv with valid-mask propagation; 3-ch mask input.
- train.py: correct density (no with-replacement sampling), patch 64 (no upscaling), seed,
  val set (tiny-imagenet-200/val, 500 imgs x 5 levels), best-by-val ckpt + `_last.pth` full resume.
- `load_checkpoint` strict=True; infer requires `noise_model == channelwise_v3` tag.
- `to_uint8` rounds instead of truncating; `write_rgb` raises on failure.
- New configs: `configs/{dncnn_v3,dncnn_sp_v3,seconv_sp_v3}.yaml` (30 epochs each).

### Done (Step 1 — training, 2026-10-12)
- `ai_based_models/train_all_v3.sh` trained all 3 models, 30/30 epochs, no errors. Logs: `ai_based_models/logs/train_*_v3.log`.
- Best ckpts (val avg PSNR): DnCNN-SP ep27 30.39, SeConvUNet ep30 30.13, DnCNN ep30 29.04.
- Verified on 200 Tiny test imgs via infer path: test PSNR ≈ val PSNR (no train/test mismatch);
  clean channels preserved exactly; 480p runs fine.

### Deferred to final joint review (user: do NOT add now)
- Val PSNR fluctuates (DnCNN up to ±7 dB, DnCNN-SP ±1.5 dB): constant LR / BatchNorm → consider cosine LR decay.
- SeConvUNet stuck ~13.8 dB for epochs 1–5 before converging (init sensitivity); still improving at ep30.
- DnCNN-SP run was stopped/resumed once at epoch 8 (valid, but not bit-identical to an uninterrupted run).
- Plain DnCNN on 480p is non-monotonic (10%: 31.26 dB < 30%: 34.30 dB) — likely same LR/BN instability.
- F1 definition (existing methodology): FP = clean pixel damaged, so the do-nothing "Noisy" baseline gets
  precision ≈ 0.96–0.99. Semantics should be reconsidered/explained in the paper.
- MDBUTMF border bug costs ~4.2 dB on Tiny at 10% (border-excluded 28.62 vs 24.42) — fixed in Step 3.

### Done (Step 2 — inference + unified evaluation, 2026-10-12)
- Old invalid v2 DL outputs moved (not deleted) to `Tiny-Imagenet/_legacy_v2/` and `480p-Test/_legacy_v2/`.
- `ai_based_models/infer_all_v3.sh` → v3 outputs in canonical folders (Tiny 5x1000, 480p 5x900 per model).
- `evaluate.py` (root) replaces notebook + test/full_eval_v2.py: all methods recomputed with same functions,
  channel-wise F1 GT mask, n/eksik per row, shape mismatch raises, no output cache.
  Outputs: `results/{summary,average_tiny,average_480p,delta_480p_minus_tiny}.csv`, `results/psnr_ssim.{png,pdf}`.
  Log: `ai_based_models/logs/evaluate.log`. Cross-check: classical PSNR/SSIM/MAE identical to old notebook,
  Acc within 0.1 pp.
- Note: Tiny_Noisy_70 contains a stray Windows shortcut `noise-test - Kısayol.lnk` (harmless, not .png).

### Done (Step 3 — classical fixes, MATLAB run by user 2026-10-12, verified 2026-10-13)
- `run_MDBUTMF_levels.m`: symmetric pad, all pixels processed (was skipping borders).
- `run_SMF_levels.m`: `medfilt2(..., 'symmetric')` (was zero padding).
- Verified via exact Python port: interior pixels 100% identical to existing MATLAB outputs. Expected Tiny PSNR
  after rerun: MDBUTMF %10≈28.84, %50≈21.58, %90≈12.91; SMF %10≈21.78, %50≈13.33, %90≈5.99 (100-img sample).
- User ran both scripts for Tiny + 480p. Outputs match exact Python port on 100% of pixels (incl. borders).
  `evaluate.py` rerun → `results/` (Step 2 results kept in `results/_onceki_adim2/`). Only SMF/MDBUTMF changed.
  MDBUTMF avg PSNR: Tiny 17.67→20.82, 480p 22.47→23.66. SMF: Tiny 13.21→13.58, 480p 14.87→14.93.
- NOTE: both MATLAB scripts are now left in Tiny mode (lines 7–8 active, 9–10 commented).
- Fairness check: classical outputs were generated from the current noisy files (0.000% non-extreme changes).
- PAPER CLAIM: user says paper argues classical (EMPR/HDMR etc.) are better. Old "classical better" result was an
  artifact of broken DL (old DL ≈ noisy input, ~9.3 dB). Corrected: DL ~6–8 dB better (Tiny), ~5 dB (480p);
  classical wins only a few cells vs plain DnCNN (480p 10%/90%) and F1 at 90%. EMPR≈HDMR≈DAMF among classical.
  EMPR/HDMR code confirmed correct by user (from advisor). Do not tune for a claim.

### Done (Step 4 — low-priority items, 2026-10-13)
- `generate_*_noise.m`: `rng(42,'twister')` + note (existing noisy data NOT regenerated).
- README: 900 imgs, EMPR/HDMR correct names, DnCNN/DnCNN-SP/SeConvUNet descriptions, metric definitions
  (Top-1 = agreement with clean-image prediction), new run order (train_all/infer_all/evaluate.py), folder tree.
- README example strips regenerated via `make_readme_examples.py` (Original.png restored from git — user's 854x480 version).
- Legacy files moved (not deleted) to `legacy/` with `legacy/README.md` explaining why. Active pipeline smoke-tested after move.
- Not touched (unknown provenance, not referenced by code): `distorted_images/` (1.8G), `noisy_images/` (BSDS-like). Ask user.

### Joint review — in progress (2026-10-13)
- v4 training started 12:01 (`VER=v4 ai_based_models/train_all.sh`, uninterrupted): same as v3 + cosine LR
  (`lr_schedule: cosine` in `configs/*_v4.yaml`, eta_min = lr*0.01). v3 weights/configs kept. Scripts renamed
  `train_all.sh` / `infer_all.sh` (VER env, default v4).
- v4 training DONE 14:03 (no errors). Best VAL avg PSNR v3→v4: DnCNN-SP 30.39→30.98 (+0.58, stable end),
  SeConvUNet 30.13→29.41 (−0.71, cosine decays LR too early after its 5-epoch slow start), DnCNN 29.04→28.80 (−0.25,
  still unstable at 10% even with low LR). Canonical outputs/results/ are STILL v3 (uniform recipe). Awaiting user
  decision: (a) uniform v4, (b) uniform v3 (current), (c) per-model by VAL (DnCNN-SP v4, others v3). Test set NOT used.
- DECISION (user: "academically strongest", 2026-10-13): NEW v5 = uniform recipe, cosine LR, 50 epochs, all 3 models
  (`configs/*_v5.yaml`, training started ~14:20 via `VER=v5 train_all.sh`). PRE-REGISTERED before any test look:
  v5 is the paper's DL result regardless of its val vs v3/v4. v4 (same recipe, 30 ep) is reported only as a
  convergence check (v4→v5 val change). After training: set train_all.sh default VER=v5 (don't edit while running),
  infer v5 → evaluate.py → stats.py → recheck advisor-message statements; move v3/v4 configs+weights to legacy/.
- Then: `VER=v4 ai_based_models/infer_all.sh` → `python evaluate.py` → `python stats.py`; compare v3 vs v4 on VAL
  (not test); re-check advisor message statements (6–8 dB Tiny, 3–5 dB 480p, >3x Top-1, %90 F1 findings).
- `evaluate.py` now writes `results/per_image.csv`; new `stats.py` → `results/stats_pairwise.csv` (paired Wilcoxon /
  exact McNemar, Holm). Tested on Step-3 per-image data: 29 significant non-DL > DL cells (all vs DnCNN, plus F1 480p %90
  vs DnCNN-SP; none vs SeConvUNet). Among non-DL: HDMR best PSNR/SSIM on Tiny; DAMF best MAE/F1, 480p PSNR %10–70.

- v5 training DONE 21:03: best VAL DnCNN-SP 31.56 (ep49), SeConvUNet 30.37 (ep50), DnCNN 30.27 (ep48) — best of v3/v4/v5
  for all three. Convergence check v4(30ep)→v5(50ep): +0.59 / +0.95 / +1.47 dB (still improving; any under-training
  biases AGAINST DL, so DL>non-DL conclusion is conservative). DnCNN 10% val still jumps at near-zero LR (BN stats).
- v3 DL outputs moved to `*/_legacy_v3/`, v3 results copied to `legacy/old_results/results_v3/`.
- Claude/VSCode session restarts KILL background jobs (even setsid). Fix: `loginctl enable-linger mustafa` (done) +
  `systemd-run --user --unit=noise-v5-chain <scratchpad>/chain_v5.sh` started 22:56 (idempotent: finishes SeConvUNet
  480p infer if <900 files newer than 2026-10-13 22:30, then evaluate.py, then stats.py).
  CHECK: `cat ai_based_models/logs/chain_v5.status` (expects "BİTTİ"); `systemctl --user status noise-v5-chain`.
- ROOT CAUSE of "session crashes": the WSL VM itself reboots (boots 22:14, 22:17, 22:49, 22:55, 23:01); journal shows
  `misc dxg: dxgk: send_create_allocation failed / Ioctl failed: -75` right before the crash → WSL GPU (dxg) path fails
  during 480p GPU inference. Nothing inside WSL survives a VM reboot (systemd-run also died).
- FIX (23:04): remaining work runs CPU-ONLY (CUDA_VISIBLE_DEVICES="") as an ENABLED systemd user service that restarts
  on every WSL boot (linger on): `~/.config/systemd/user/noise-v5-chain.service` → `~/.local/share/noise-v5/chain.sh`
  (idempotent; exits if status has "BİTTİ"; disables itself when done). Steps: `infer_resume.py` regenerates ALL
  SeConvUNet 480p v5 outputs on CPU (atomic tmp+rename writes, skips files newer than `~/.local/share/noise-v5/start_ts`
  that decode OK) → `evaluate.py` (CPU, incl. ResNet) → `stats.py`. ~1.2 s/img → infer ≈ 90 min, eval ≈ 45 min.
  CPU vs GPU outputs differ only by ±1 gray level on 0.2–0.4% of values (negligible).
- Already verified: v5 training uninterrupted (epochs 1..50, no resume, no NaN); 24,000 v5 outputs (all except SeConvUNet
  480p) decode OK, names/counts match inputs, written after 22:30, clean channels preserved.
- DONE 2026-10-14 00:55 (CPU chain, no WSL crash since 23:01 → GPU/dxg confirmed as trigger). VERIFIED: all 28,500 v5
  DL outputs OK (SeConvUNet 480p fully CPU-generated); results/ 100 rows, per_image 95,000 rows, 0 missing; non-DL
  PSNR/SSIM/MAE/F1 identical to v3 run, Top1 ≤0.2 pp diff (ResNet on CPU). systemd unit removed; linger still ON
  (`loginctl disable-linger mustafa` to undo). v3/v4 configs+weights moved to legacy/; README strips regenerated (v5).
- v5 TEST RESULTS (avg PSNR Tiny/480p): DnCNN-SP 31.03/32.15, SeConvUNet 29.78/30.97, DnCNN 29.61/30.11;
  HDMR 22.46/25.96, EMPR 22.29/25.79, DAMF 22.25/26.13. DL−(HDMR/EMPR/DAMF): Tiny 7.1–8.8 dB, 480p 4.0–6.4 dB.
  Top1 ratio 3.1–3.8x. Significant non-DL > DL cells: ONLY 4 (480p %10 SSIM vs plain DnCNN). The v3 "%90 F1"
  advantage REVERSED (DL now better everywhere). Non-DL beats DnCNN-SP per image in ≤1.1% (very smooth images).
  → The advisor paragraph sent earlier (v3-based) is OUTDATED; updated paragraph given to user 2026-10-14.
- GPU in WSL still crashes the VM (dxg alloc failures) — recommend Windows restart + NVIDIA driver + `wsl --update`
  before any further GPU work.
- (old) NEXT after BİTTİ: re-run integrity check on SeConvUNet 480p outputs; verify results/ (n, no missing), compare with v3 numbers & advisor-message statements, move v3/v4
  configs+weights to legacy/, update legacy/README + handoff, report to user.

### Open for the joint review (user wants to review all together)
- CLOSED 2026-10-13: EMPR/HDMR implementations confirmed correct by user (code comes from their advisor) — do not re-raise #17.
- CLOSED 2026-10-14: DAMF confirmed fine by user; README describes the algorithm (no expansion).
- Paper claim (classical better) not supported by corrected results — framing is user's/co-authors' decision.
- Deferred training items above (LR decay, SeConvUNet slow start, DnCNN instability), F1/Top-1 definitions.
- COMMITTED 2026-10-14: 5 commits (NO co-author lines), fast-forwarded into `main` (branch audit-fixes deleted). PUSHED by user (origin/main = c92ad61).
  Temp service scripts (~/.local/share/noise-v5) removed. Linger still ON. ~33 GB archived old outputs remain on disk
  (480p-Test/_legacy_v2 13G, _legacy_v3 11G, legacy/test/dl_outputs 8.2G) — offer deletion, user decides.
- AI context files (AGENTS.md, CLAUDE.md, .ai/**/*.md) committed on user request 2026-10-14 (no co-author).
  Still untracked by choice: .ai.zip, scripts/codex-review.sh.
  DL fixes+v5 configs / classical fixes / evaluate+stats+results / legacy archive / detailed README.
  AI tooling (.ai/, AGENTS.md, CLAUDE.md, scripts/, .ai.zip) intentionally left untracked.

### Not yet done (planned steps)
- None — all 4 planned steps done.

---

## Files touched in the last session
- `ai_based_models/src/utils/spmask.py`, `utils/config.py`, `utils/io.py`
- `ai_based_models/src/models/dncnn_sp.py`, `models/seconv_unet.py`, `models/external_wrappers.py`
- `ai_based_models/src/train.py`, `ai_based_models/src/infer.py`
- `ai_based_models/configs/*_v3.yaml` (new)

## Commands/tests run
```bash
cd ai_based_models/src
python train.py --config ../configs/dncnn_sp_v3.yaml   # also seconv_sp_v3, dncnn_v3
python infer.py --config ../configs/dncnn_sp_v3.yaml --dataset tiny
python infer.py --config ../configs/dncnn_sp_v3.yaml --dataset 480p --noisy_root ../../480p-Test --out_root ../../480p-Test
```
Smoke tests (scratchpad): noise stats match MATLAB files, mask recall 100%, gating preserves clean
channels, mini-train/val/resume/infer OK, old v2 ckpts rejected.

## Important context for the next assistant
- Old v2 weights/outputs are methodologically invalid (pixel-wise noise vs channel-wise test noise;
  SeConvUNet v2 eval output was a constant colour). Do not reuse them.
- `test/full_eval_v2.py` no longer works with new models (6-ch input) — to be replaced in Step 2.
- MATLAB is not available in WSL.

## Next recommended step
Joint review with the user of the 'Open for the joint review' list; then (only if asked) commit — WITHOUT any Claude co-author line.

---

