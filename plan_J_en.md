# Plan J: journal extension of Plan D (for Information Sciences)

*Translation note.* English translation of `prereg_j.md` (Chinese). Registered together with `prereg_d.md` and its translation. Where they differ, the Chinese original governs.

Dates: 2026-11-09 (first draft), 2026-11-10 (revised before registration after two independent blind reviews).
**Written before any D-track or J-track dump.** Registered together with `prereg_d.md` before any run (a release of a public GitHub repository archived by Zenodo with a DOI).

## 0. Relation to Plan D

The target venue changes from DASFAA 2027 to the journal Information Sciences (no page limit). This file only adds arms and criteria; it does not change any criterion of `prereg_d.md`.

- D0–D4 are judged on G1–G3 exactly as written in `prereg_d.md`. The new G4 is judged by the same rules in a separate row and is not merged into D-track verdicts, so the D criteria become neither stricter nor looser because of an extra generator.
- Estimands, intervals, gate-failure handling and technical-fallback rules follow `prereg_d.md` sections 4 and 5.
- `prereg.md`, `prereg_a.md`, `prereg_r.md`, `prereg_r_loops.md` are unchanged. The 10% and 0.05 lines are unchanged. The "not supported" verdicts on A2 and R3 stand.

## 1. Generators

| Code | Model | Notes |
|------|-------|-------|
| G1–G3 | As in `prereg_d.md` | Unchanged |
| G4 | Llama-3.1-8B-Instruct, local, greedy, same environment as G2 (torch 2.4.1+cu121, transformers 4.46.3, `LOCAL_ATTN=eager`) | Third model family. Weight source and per-file hashes recorded. The model's own chat template; system prompt unchanged |

## 2. New arms

### J1 Second dataset: EB-NeRD

EB-NeRD changes language (Danish), category taxonomy and impression structure at once. J1 tests whether the conclusions hold on logs from another language and platform; it cannot be attributed to any single one of these changes, and the paper words it accordingly.

**Conversion** (`data/ebnerd_prepare.py`; based on field structure only, never on simulation results):

- Use only the train and validation splits and the articles table of EB-NeRD small.
- `news.parquet`: one row per article. `category` = `category_str`; `subcategory` = first element of the `subcategory` list, missing if the list is empty; `title` in the original Danish, untranslated; `popularity` = number of clicks on the article in the train split. Articles with missing `category_str` never enter candidates, histories or clicks.
- `impressions_dev.parquet`: one row per validation impression. `history` = the articles in the user's validation history file, ascending by read time; `candidates` = `article_ids_inview` in original order, de-duplicated keeping the first occurrence, dropping articles absent from the articles table or lacking a category; `clicked` = `article_ids_clicked` ∩ `candidates`.
- `sessions_train.parquet`: one row per train impression, same rules; `session_idx` = rank of the user's impressions by time ascending (from 1; ties broken by impression ID ascending); `history` from the user's train history file.
- `users.parquet`: one row per validation user, `history` as in `impressions_dev`.
- All IDs cast to strings.
- **Impression filter**: `impressions_dev` keeps only impressions with \(\ge 1\) click, \(\ge 5\) history items and 5–50 candidates; the number removed at each step is reported.

**Arms** (statistics as in `prereg_d.md` section 4):

- Human ER on logged slates, out-of-history supply \(S\), ratio \(ER/S\).
- E1 retry: G2, G3, G4. E1 cal (oracle target): G2, G4. Both on `sample(n=1500, random_state=0)` of the filtered impression table; all generators share the same 1500 impressions.
- Random-slate loop: uncalibrated arm of G2, G3, G4, seed 0, 200 users × 10 rounds. Users drawn with `sample_users(users, n=200, seed=0, min_hist=10)` from `users.parquet`, as on MIND; random slates drawn from all articles in `news.parquet`. D3's uniform and fitted reference clickers run on the same slates, with \(\pi\) refitted on EB-NeRD's `sessions_train` following the D3 details in `prereg_d.md`.
- NRMS loop: G2, G4, seed 0, uncalibrated and calibrated arms; descriptive (no criterion). NRMS is trained once with the same configuration as the MIND NRMS without GloVe (`recsys/train_nrms.py`: randomly initialized word embeddings, dimension 256, 2 epochs, batch 256, Adam lr \(10^{-4}\), 2 negatives per positive, seeds of `random`, `numpy`, `torch` set to 0), keeping the epoch with the highest validation AUC; AUC is pooled item-level AUC on a `random_state=0` sample of validation impressions, computed as on MIND. The loop runs only if AUC \(\ge 0.55\); below that, on training failure or missing AUC, it is not run and the reason is reported.
- No replay, no multiple seeds.

### J2 Third model family (on MIND)

G4 runs the following arms, under `runs/j_track/g4/`:

- E1: retry, cal, one, empty, shuffled. No temp1.
- Offline: D1-top1, D2-paired, D2-train (E1 rule), D2-book, D8, as in `prereg_d.md`.
- Random-slate loop, seeds 0–2, both arms; NRMS, NAML, DKN loops, seeds 0–2, both arms.
- Seed-0 replay, three arms (uncalibrated, oracle-target calibrated, training-target calibrated).

If G4's E1 retry empty rate exceeds 10%, neither loops nor replays are run.

### J3 Subcategory metrics (offline)

On every run with item records: subcategory ER (share of clicked items whose subcategory is not in the history subcategory set) and Subcat@20 (number of distinct subcategories in the top 20). Items with missing subcategory are excluded from the click denominator and from the history subcategory set; an impression whose clicks all lack a subcategory is not counted. MIND has 270 subcategories; EB-NeRD uses the converted subcategory above. Old dumps without per-round slates: E1 only.

### J4 Prompt sensitivity (MIND E1)

Same 1500 impressions, retry arm, G2 and G4. Three factors, 2×2×2 = 8 variants:

- **Profile summary**: present / absent (the whole `Reader profile` block in `sim/prompts.py` kept or removed).
- **Click-rate hint**: present / absent (only the sentence "Real readers click about … of the headlines they see." is removed; the return-format sentence stays).
- **History length** \(h\): 30 / 10. Both the history list in the prompt and the profile summary use only the last \(h\) items. The ER reference history is always the impression's full history and does not vary with \(h\).

"Present, present, 30" is the main retry arm (G2 from `runs/rep7b/`, G4 from J2's retry) and is not rerun; each generator runs 7 new variants. Retry text on empty parses is identical to the main experiment. The 8 full templates are generated by code and archived in `runs/j_track/prompt/templates/` before any run; no other wording, persona or example is added.

## 3. Criteria

| ID | Prediction | Supported | Not supported |
|----|------------|-----------|---------------|
| **J0** | Empty clicks \(\le 10\%\) in every G4 cell, every LLM cell on EB-NeRD, every J4 variant (G4's D1-empty exempt) | Judged per cell, as D0 | Cells above 10% are marked "instrument failure", sensitivity only |
| **J1a** | On EB-NeRD the simulator explores less than humans | E1 retry arm of each of G2, G3, G4: \(\Delta_{ER}\) point estimate \(\le -0.05\) and 95% CI upper bound \(<0\) | Any generator fails: report which and by how much. Any generator with instrument failure: "inconclusive", with the other generators' estimates reported |
| **J1b** | On EB-NeRD random slates, round 1, LLM click repeat exceeds the fitted clicker | Rule as in D3, for each of G2, G3, G4 | As in D3: check (LLM − uniform); "above the uniform reference" only if its CI lower bound \(>0\) |
| **J2** | G4 reproduces D1–D4 on MIND | A separate G4 row under the original D1–D4 rules (D1, D2 on G4 alone; D4 on G4 seeds 0–2, both arms) | Report each failing criterion |
| **J3** | The simulator also explores less at subcategory level | MIND E1 retry arm: subcategory \(\Delta_{ER}\) point estimate \(\le -0.05\) and 95% CI upper bound \(<0\), for each of G1–G4 | Any generator fails: category-level conclusions are not extended to subcategories |
| **J4** | The direction of under-exploration does not depend on prompt wording | For each of G2 and G4, all 8 variants pass J0 and all satisfy \(\Delta_{ER}\le -0.05\) with 95% CI upper bound \(<0\) | Report the number of variants satisfying it, and the failing and instrument-failure variants; the paper does not claim robustness to prompts |

J4 also reports descriptively: the range of \(\Delta_{ER}\) over the 8 variants, and the mean difference between the two levels of each factor.

## 4. Statistics

As in `prereg_d.md` section 4. Each criterion gets its own verdict; no multiplicity correction; the paper does not combine criteria into an overall verdict.

## 5. Stopping

- EB-NeRD small is the only second dataset; no third.
- Only the 8 J4 variants; no variants added or wording changed because of results.
- Only G4 is added as a model family; no model swapped or added because of results.
- After seeing results, EB-NeRD conversion rules, impression filter, category field and language handling are not changed.
- If after download EB-NeRD's fields differ from section 2, any change based on field structure alone is appended as a dated amendment to this file and published as a new release, archived by Zenodo as a new version, before any LLM arm runs.
- NRMS on EB-NeRD is trained once; no tuning, no retraining with other seeds.
- The rules in `prereg_d.md` section 5 still apply.

## 6. Outputs and commands

Outputs: `runs/j_track/ebnerd/`, `runs/j_track/g4/`, `runs/j_track/prompt/`, `runs/j_track/analysis/`. Nothing in `runs/a_track/`, `runs/rep7b/`, `runs/d_track/` is overwritten. EB-NeRD's NRMS checkpoint goes to `checkpoints/ebnerd/nrms/`, never mixed with MIND checkpoints.

```text
# Local CPU
python data/ebnerd_prepare.py                     # conversion + per-step filter counts
python scripts/d_track.py ref --data ebnerd --chooser uniform|fitted
python analysis/j_subcat.py                       # J3
python analysis/j_verdicts.py                     # J0–J4 verdict table

# G4 (AutoDL), MIND
python scripts/d_track.py e1 --gen g4 --arm retry|cal|one|empty|shuffled
python scripts/d_track.py placebo --gen g4 --seed {0,1,2} [--calibrate]
python scripts/d_track.py e3 --gen g4 --rec {nrms,naml,dkn} --seed {0,1,2} [--calibrate]
python scripts/d_track.py replay --gen g4 --seed 0 [--calibrate [--target-mode train]]

# Prompt variants (G2, G4; on/on/30 skipped)
python scripts/d_track.py e1 --gen {g2,g4} --arm retry --profile {on,off} --rate-hint {on,off} --hist {30,10}

# EB-NeRD
python recsys/train_nrms.py --data ebnerd --seed 0
python scripts/d_track.py e1 --data ebnerd --gen {g2,g3,g4} --arm retry
python scripts/d_track.py e1 --data ebnerd --gen {g2,g4} --arm cal
python scripts/d_track.py placebo --data ebnerd --gen {g2,g3,g4} --seed 0
python scripts/d_track.py e3 --data ebnerd --gen {g2,g4} --rec nrms --seed 0 [--calibrate]
```

## Amendment 1 (2026-11-10, before any EB-NeRD download or run)

Basis: EB-NeRD General License Terms (https://recsys.eb.dk/assets/pdf/general_license_terms.pdf), sections 4.2–4.5: the dataset may be used only in the user's own IT environment, may not be transferred to a third party, may not be copied beyond the download from an Ekstra Bladet server, and published results may not contain exact parts of it. The changes rest on the license terms only, not on any result.

1. G3 is not run on EB-NeRD. G3 would send EB-NeRD headlines to a third-party API, contrary to sections 4.4 and 4.5. On EB-NeRD, J1a and J1b in section 3 are judged on G2 and G4 only; the criteria themselves are unchanged. G3 is dropped from the EB-NeRD E1 retry arm and random-slate loop arm in section 2 (J1). G3 on MIND is unchanged.
2. EB-NeRD is downloaded from the Ekstra Bladet server directly to AutoDL. Conversion (`data/ebnerd_prepare.py`), the reference clickers and all EB-NeRD commands in section 6 run on AutoDL; no local copy is kept.
3. The paper, supplement and replication package contain no EB-NeRD content: no headlines or body text as examples, no converted parquet files, no per-impression dumps with article IDs or raw model outputs; only code and aggregate statistics are released.

The EB-NeRD commands in section 6 become:

```text
python scripts/d_track.py e1 --data ebnerd --gen {g2,g4} --arm retry
python scripts/d_track.py placebo --data ebnerd --gen {g2,g4} --seed 0
```
