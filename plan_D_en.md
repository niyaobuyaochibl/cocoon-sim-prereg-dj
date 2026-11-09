# Plan D: revision plan after review of TCSS-2026-09-1681

*Translation note.* English translation of `prereg_d.md` (Chinese). Registered together with `prereg_j.md` and its translation. Where they differ, the Chinese original governs.

Dates: 2026-11-09 (first draft), 2026-11-10 (revised before registration after two independent blind reviews).
**Written before any D-track dump.** Registered before any run as a release of a public GitHub repository archived by Zenodo with a DOI (the Zenodo timestamp is the registration time). The review manuscript withholds the link. Target venue: see `prereg_j.md` section 0.

Does not change `prereg.md`, `prereg_a.md`, `prereg_r.md`, or `prereg_r_loops.md`. The 10% line and the 0.05 line are unchanged. The "not supported" verdicts on A2 and R3 stand; no result from this track is used to overturn them.

## 0. Why this track

Consensus of the six TCSS reviewers: the closed loop had a single seed; forced clicking with 2.6 clicks per impression versus 1.56 for humans is a confound; on random slates the only reference was a theoretical uniform clicker; the step size of one swap was misstated (it is \(1/k\), \(k\in\{1,2,3\}\)), and the swap target in E1 and replay came from the human clicks on the same impression (oracle); no replication package. This track addresses only these. The second dataset, third model family, subcategory metrics and prompt variants are in `prereg_j.md`.

## 1. Generators

| Code | Model | Status |
|------|-------|--------|
| G1 | DeepSeek-V4-Flash, September 2026 dumps in `runs/a_track/` | Frozen. The alias `deepseek-v4-flash` is now served by `deepseek-flash` (probe on 2026-11-09 returned `model: deepseek-flash`). **G1 cannot be rerun**; offline recomputation only |
| G2 | Qwen2.5-7B-Instruct, local, greedy, `LOCAL_ATTN=eager`, torch 2.4.1+cu121, transformers 4.46.3 | Same configuration as the R track; rerunnable |
| G3 | `deepseek-flash`, requested explicitly by this name, temperature 0, thinking off | New generator. Every call logs the returned `model`, `system_fingerprint`, `usage`; full raw responses archived |

G3 is a third generator, not a substitute for G1. G1 and G3 are reported separately and never pooled.

## 2. New arms

All LLM arms use the A-track instrument (`allow_empty=false`, up to 2 retries on empty parses) unless stated. All new loops and replays log, per round and user, the shown slate, raw clicks and post-swap clicks (`events.jsonl`); old dumps lack this layer.

### D1 Click-count control (forced clicking, click-count confound)

Same E1 impressions (`impressions_dev.parquet`, `random_state=0`, n=1500).

- **D1-one**: prompt asks for exactly one index; parser cap 1. G2, G3.
- **D1-empty**: original 7 September wording (allows `[]`, no retry). G2, G3. Report the empty rate and ER on non-empty impressions. Exempt from the 10% gate; the empty rate is itself a measurement.
- **D1-top1**: offline. First index of the list returned by the 1–3 arm. G1, G2, G3. Descriptive.

### D2 Correcting the swap operator (step size, oracle target, 2.63 vs 2.62)

All offline: replay `match_exploration` on cached `pred_raw` with the original seeds.

- **D2-paired**: apply the oracle-target swap to the cached clicks of the retry arm, giving a calibrated result paired impression by impression with the retry arm. E1 of G1 and G2. The separately generated cal arm is kept as the "independently generated" version, with the source of the difference stated (221 of G1's 1500 raw click lists differ).
- **D2-train**: the target no longer comes from the evaluated click.
  - E1 (dev impressions, later in time than all training sessions): target = the user's mean human ER over all non-empty training sessions (`load_user_target_er`); users without training sessions get the session-weighted mean human ER over all non-empty training sessions (`train_mean_er`; **not** 0.245, which is a number from the evaluation sample). E1 of G1 and G2.
  - Replay (the evaluated sessions are training sessions): the target for replayed session \(t\) uses only that user's training sessions with smaller `session_idx`; if there is none, the session-weighted mean over all non-empty training sessions of users outside the replay sample. The evaluated session and later sessions never enter the target. New replays of G2 and G3.
- **D2-book**: swap accounting: share with no swappable candidate, share where \(|e'-p|\) grows after the swap, step sizes by \(k\).
- **Reduction**: \(R = |\Delta_{\rm raw}| - |\Delta_{\rm cal}|\), with \(\Delta\) the aggregate gap defined in section 4 (not a per-impression absolute error). \(R>0\) means the gap shrinks. Both terms are recomputed in every bootstrap resample.

### D3 Reference clickers on random slates (no human reference)

Same 200 users (`sample_users`, seed 0), same random slates (`random.Random(0)`; slates do not depend on clicks), 10 rounds, CPU only.

- **D3-uniform**: clicks per round \(k\) drawn from the empirical distribution of human click counts in training sessions (capped to 1–3); items drawn uniformly from the 20 shown.
- **D3-fitted**: same \(k\). Each click is, with probability \(\pi\), drawn uniformly from items outside the current history categories; otherwise drawn with probability proportional to the frequency of the item's category in the user's history; uniform if no candidate. \(\pi\) is fitted once on training sessions: on the grid 0.00–1.00, step 0.01, choose the value whose mean ER on logged training slates is closest to the human mean ER. Fitting uses training sessions only, never the E1 sample. Fitting details (`fit_pi` in `sim/ref_chooser.py`): if there are more than 20,000 non-empty training sessions, sample 20,000 with `random_state=0`, and compute the human mean ER on the same 20,000; during fitting \(k\) is the session's human click count (capped to 1–3); one simulation per session, with a random seed determined by the session ID and shared across grid points; ties go to the smaller \(\pi\). The fitted \(\pi\) is never overridden by hand. CIs for reference clickers are conditional on the fitted \(\pi\).
- Per arm and round, report: out-of-history supply \(S_t\), ER\(_t\), ER\(_t/S_t\), click repeat, number of history categories covered. Among LLM arms, G2 reruns the two random cells (greedy decoding is reproducible; the rerun should match `runs/rep7b/random_*` round by round in Category@20 and click repeat, which also answers "the reconstructed slates were checked only on Category@20"); G3 runs new; G1 uses user-level click repeat from `users.jsonl`.

### D4 Multi-seed loops (single seed)

- G2: NRMS / NAML / DKN × {retry, cal} × seeds {1, 2}, plus the two Random cells × seeds {1, 2}. 200 users × 10 rounds per cell. The seed changes both user sampling and the swap random numbers.
- G3: Random, NRMS, NAML, DKN × {retry, cal} × seeds {0, 1, 2}.
- Rule-based clicker: existing curves reused.

### D5 Seed-0 robustness replay (planned but not run earlier)

G2 and G3, seed 0, 200 users, up to 5 sessions, three arms: uncalibrated, calibrated with the oracle target, calibrated with the training target (replay rule of D2-train). New replays log items; offline additions: ER on both sides against a common reference history (human history categories), per-session sample sizes, a balanced cohort of users with \(\ge 5\) sessions.

### D6 Candidate-order permutation

E1 with `order_arm=shuffled`, 1500 impressions each for G2 and G3. Descriptive.

### D7 Temperature

E1 retry arm: G3 at temperature 1.0 once; G2 with `do_sample=True, temperature=1.0` once. Descriptive, 1500 each, single seed.

### D8 AUC references (offline)

State that AUC on binary scores equals \((\mathrm{TPR}+\mathrm{TNR})/2\). Add two references: uniform random choice with the simulator's per-impression click count; uniform choice restricted to history categories with the same click count (if fewer than \(k\) in-history candidates exist, the shortfall is filled uniformly from the remaining candidates and the number of such impressions is reported). 1000 repetitions each, averaged; repetition \(r\) uses seed \(r\). AUC is pooled over items; the evaluation set is fixed to the impressions used for the simulator's own AUC (both label classes and both score values present), and the references are evaluated on the same set. G1, G2, G3.

## 3. Criteria

| ID | Prediction | Supported | Not supported |
|----|------------|-----------|---------------|
| **D0** | Empty clicks \(\le 10\%\) in G3's E1 retry arm, every loop cell and every replay arm (D1-empty exempt) | Judged per cell. Empty rate = impressions with no click (loops: user × round) / cell total, pooled over rounds | Cells above 10% are marked "instrument failure" and go to sensitivity only; dependent criteria per section 4 |
| **D1** | In the D1-one arm the simulator still explores less than humans | On the E1 subset where the human clicked exactly one item, D1-one's \(\Delta_{ER}\) point estimate \(\le -0.05\) and 95% CI upper bound \(<0\), for both G2 and G3 | Any generator fails: write "the registered gap is not supported under single-click"; no attribution of the gap to click count (D1-one changes both the prompt and the parser cap) |
| **D2** | The paired oracle swap shrinks the gap; the training-target swap shrinks it less | For E1 of G1 and of G2 separately: 95% CI lower bound of \(R_{\rm oracle}\) \(>0\), and 95% CI lower bound of \(R_{\rm oracle}-R_{\rm train}\) \(>0\) | Any condition fails for any generator. \(R_{\rm train}\) is reported with its CI and may be negative |
| **D3** | On random slates, round 1 (seed 0), LLM click repeat exceeds the fitted clicker | Paired per-user difference (LLM − fitted) point estimate \(\ge 0.10\) and 95% CI lower bound \(>0\), for the uncalibrated arms of G1, G2, G3 separately | For a failing generator, check (LLM − uniform): only if its 95% CI lower bound \(>0\) may we write "above the uniform reference"; otherwise "not above either reference" |
| **D4** | NAML's last-round Category@20 is below NRMS and DKN on new seeds and the new generator | For every (seed, arm) of G2 seeds 1, 2 and G3 seeds 0–2, the paired CIs of NAML−NRMS and NAML−DKN exclude 0 and are negative | Any (seed, arm) fails; report which cells. NRMS−DKN reported only, not predicted. The claim is about last-round ordering only, not that feedback caused narrowing |
| **D5** | Descriptive, not a hypothesis test | Seed-0 replay, session 1, \(\Delta_{ER}=ER_{sim}-ER_{human}\), three arms × two generators, classified in order: (1) equivalent: \(-0.05<\) lower and upper \(<0.05\); (2) beyond the line: upper \(\le -0.05\) or lower \(\ge 0.05\); (3) other, excludes 0: upper \(<0\) or lower \(>0\); (4) otherwise inconclusive | — |

The original A2 and R3 criteria and their "not supported" verdicts are reported as registered and are not reinterpreted retroactively as equivalence tests. D5's four-way classification applies only to the new replays.

## 4. Statistics

- **Estimands.** E1: \(\Delta_{ER}\) = equally weighted mean of per-impression \(ER_{sim}-ER_{human}\) within a common mask; the common mask = impressions where the human clicked \(\ge 1\) item and the simulator (after retries) clicked \(\ge 1\); mask size and exclusions are reported. Loops and replays: average over defined rounds (sessions) within user, then average users with equal weight. ER always uses the category set of the real or simulated history at that impression (session, round) as reference, unless "common reference history" is stated. \(ER/S\) only on impressions with \(S>0\); the number with \(S=0\) is reported.
- **Intervals.** User-clustered percentile bootstrap, 1000 resamples, seed 0: each resample draws users and recomputes the whole estimator on their impressions (both terms of \(R\), both sides of paired differences).
- Equivalence uses 95% intervals, more conservative than the 90% intervals usual in two one-sided tests.
- **Gate failure.** When a criterion requires each generator to hold, and any required cell is marked instrument failure under D0, the criterion is "inconclusive"; support is never declared on the remaining generators.
- Order: before this file is archived on Zenodo, only smoke tests with at most 5 users are run, and no quantity with a criterion is computed, including offline recomputations on existing dumps (D2, D3, D8, D1-top1).
- Paired differences in D3 and D4 are paired at the user level.
- Loop curves use paired last-round and first-round differences rather than GEE slopes alone; GEE slopes become supplementary.
- Multiple seeds: a verdict per seed, plus the between-seed standard deviation of last-round means.

## 5. Stopping

- No prompt change because D1 fails.
- No change to the fitted clicker's definition or to the fitting of \(\pi\) because D3 fails.
- No extra seeds because D4 fails on some seed.
- No human experiment. Second dataset and prompt variants only as in `prereg_j.md`.
- G1 is not rerun; G3 is never presented as G1.
- Results of this track are not used to rewrite A2 or R3.
- **Technical fallbacks**: if the local model falls back from the attention implementation, or shrinks the batch or truncates context on out-of-memory, the cell stops and the event is logged; after fixing the environment, the cell is rerun with an identical configuration and both attempts are reported. Empty output caused by infrastructure failure is logged separately from a model-returned `[]` and from unparsable output. If the API's returned `model` changes mid-cell, the cell stops. After an interruption, resume after the last completed user; never rerun a whole cell and choose. If fewer than 1500 usable E1 impressions exist, stop; no smaller sample is accepted.
- If G2's rerun of the two random cells differs from `runs/rep7b/` round by round, the difference is reported and seed-0 criteria use `runs/rep7b/`.

## 6. Outputs and commands

Outputs: `runs/d_track/g2/`, `runs/d_track/g3/`, `runs/d_track/ref/` (reference clickers), `runs/d_track/analysis/` (offline recomputation). Nothing in `runs/a_track/` or `runs/rep7b/` is overwritten. All G1 baselines and G2's E1 retry, cal and seed-0 loops are reused read-only from `runs/a_track/` and `runs/rep7b/`, not rerun.

```text
# Local CPU
python scripts/d_track.py ref --chooser uniform
python scripts/d_track.py ref --chooser fitted
python analysis/d_offline.py             # D1-top1, D2-*, D8
python analysis/d_loops.py               # D3, D4 paired differences, D5 classes, replay offline items

# G2 (AutoDL)
python scripts/d_track.py e1 --gen g2 --arm one|empty|shuffled|temp1
python scripts/d_track.py placebo --gen g2 --seed {0,1,2} [--calibrate]
python scripts/d_track.py e3 --gen g2 --rec {nrms,naml,dkn} --seed {1,2} [--calibrate]
python scripts/d_track.py replay --gen g2 --seed 0 [--calibrate]
python scripts/d_track.py replay --gen g2 --seed 0 --calibrate --target-mode train

# G3 (API)
python scripts/d_track.py e1 --gen g3 --arm retry|cal|one|empty|shuffled|temp1
python scripts/d_track.py placebo --gen g3 --seed {0,1,2} [--calibrate]
python scripts/d_track.py e3 --gen g3 --rec {nrms,naml,dkn} --seed {0,1,2} [--calibrate]
python scripts/d_track.py replay --gen g3 --seed 0 [--calibrate]
python scripts/d_track.py replay --gen g3 --seed 0 --calibrate --target-mode train
```
