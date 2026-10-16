# Founder Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify task-specific ECGFounder teachers and measure controlled transfer into the existing TITAN student in CEDIA.

**Architecture:** Pair manifest-directed 12-lead source signals with the unchanged six-lead student input. Adapt separate rhythm and pathology teachers, then export training-only soft targets and compare supervised controls against endpoint-specific and joint transfer. Every stage produces sealed inputs and results; an ineligible teacher stops only its transfer branch.

**Tech Stack:** Python, NumPy/SciPy/WFDB, PyTorch, scikit-learn, existing TITAN metrics and SHA manifests, SSH with strict host checks, CEDIA Slurm/A100.

**Spec:** `docs/superpowers/specs/2026-10-14-titan-pretrained-transfer-design.md`; written-spec approval received from the user after commit `3e32d17`. The user reviewed this plan and selected direct implementation with one independent final review on 2026-10-14. Checked steps below have observed execution evidence; unchecked steps remain pending.

## Global Constraints

Execution checkpoints (2026-10-15):
- [x] Task 1: paired inputs and complete source/tensor audit.
- [x] Task 2: pinned teacher adapter and restricted serialization.
- [x] Task 3: isolated runtime and zero-step A100 pilot (job 29619).
- [x] Task 4: endpoint teacher qualification (30 trials and deterministic repeat, job 29627).
- [x] Task 5: qualified training-only soft targets (22,540 keyed rows, job 29644).
- [x] Task 6: same-architecture student comparisons (nine fits and deterministic repeat, job 29650).
- [x] Task 7: selected-model source replay and reviewer packet (CEDIA 29656; independent CPU and clean Git archive checks completed 2026-10-16).

The detailed steps below preserve the original implementation instructions;
execution receipts are in `outputs/reviewer_verification/founder_transfer_2026-10-14/`.

- Deployed student: 1,143,751 parameters, four existing heads, six leads I, II, III, aVR, aVL, aVF, 125 Hz, float32 `[B,6,1250]`.
- Rhythm order: AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, 1AVB.
- Pathology order: IMI, ASMI, LVH, ISC_, ISCAL, NST_, ILMI, AMI, ALMI, LAE; Primary-5 indices `[0,8,6,9,3]`.
- Warm-start SHA-256: `72d3ece9b00e844d6a94f207f4ced71af0af69c59cef66b625031ff92b2f965b`; initialize weights, not a claimed optimizer resume.
- Preserve Gold, prior PTHs, original labels and all archived scripts/results. Do not change deployed strides or add deployed projection modules in this plan.
- Original training union: 22,567 records. Execution found 27 signal/tensor-identical historical aliases; preserve those as excluded records, yielding a clean fitting allowlist of 22,540 with 16,413 observed native pathology records. Keep unobserved targets masked; diagnostic likelihood zero is not a negative finding.
- Teacher and student fitting excludes rhythm validation 1,476, pathology validation 2,027, follow-up 2,171 and historical 672. Split before window derivation; check patients where available and source/tensor hashes.
- All neural fits and GPU probes run in Slurm, never on the login node or local CPU. CPU unit tests and data preparation are permitted.
- This plan implements subprojects A/B only. SSL continuation, event supervision and temporal-stride variants require their separate specs/plans; do not claim them implemented here.
- 90% accuracy and approximately 87% macro-F1 remain research targets, not conditions for altering references or continuing an unbounded search.

## Review Focus

- An alias, duplicated signal or patient appearing under another manifest must not enter fitting; Task 1 tests all three identities, including split-before-window behavior.
- A hash-matched file with an incompatible or unsafe serialized state must still fail closed; Task 2 tests hash, restricted loading and body/head schema independently.
- A reordered or incorrectly scaled lead can yield plausible predictions; Task 1 tests explicit lead names, unit conversion and pinned preprocessing fixtures.
- Empty/unknown masks or shuffled teacher rows must not invent negatives or misalign supervision; Task 5 tests zero gradients and keyed joins.
- A partial Slurm output or stale selection can look like a completed result; Tasks 3 and 7 test submission identity, sealed retrieval and prediction coverage before publication.

## Implementation Boundary and File Map

Create a small package at `src/titan_v4/reproducibility/pretrained_transfer/`:
`contracts.py` for IDs/configuration, `inputs.py` for paired tensors,
`founder.py` for safe model construction, `teachers.py` for teacher fits,
`distillation.py` for keyed soft-target loss, `students.py` for student fits,
and `selection.py` for validation decisions. Add `__init__.py` without eager
teacher imports. Tests mirror these responsibilities.

Create CLI wrappers `scripts/prepare_founder_transfer.py`,
`manage_founder_transfer_cedia.py`, `run_founder_transfer.py` and
`verify_founder_transfer.py`. Reuse existing helpers through imports; do not
reshape `manage_joint_cedia_run.py` to accept incompatible 12-lead inputs.
Vendor only pinned MIT `net1d.py` and its license under
`src/titan_v4/reproducibility/pretrained_transfer/vendor/ecgfounder/`, with
an upstream notice. No external weights are committed to Git.

Create `configs/founder_transfer_2026-10-14.json` and
`requirements/founder-transfer.txt` as part of their owning tasks. Results
go to `outputs/reviewer_verification/founder_transfer_2026-10-14/`;
large rebuildable tensors and third-party weights stay in ignored cache.
Commands in this plan describe interfaces to implement, not current runnable
entrypoints. Each implementation commit also refreshes the existing root
artifact inventory; preserve its numerical archives byte-for-byte.

## Fixed Trial Protocol and Resource Ceilings

- Seeds: `20261014`, `20261015`, `20261016`; use all three, never select a seed by its score. Fixed loader order per seed and deterministic settings; record unsupported deterministic kernels as failures rather than silently disabling determinism.
- Isolated CEDIA environment: Python 3.10.14, Torch 2.6.0 CUDA 11.8 wheel, NumPy 1.26.4, SciPy 1.13.1, WFDB 4.1.2, scikit-learn 1.3.2, pandas 2.2.3 and matplotlib 3.8.4. Resolve and record transitive versions/wheel hashes during the bootstrap allocation. Do not mutate `.venv_audit`.
- Torch wheel source: [official prior-version instructions](https://pytorch.org/get-started/previous-versions/), CUDA 11.8 distribution. Driver/GLIBC compatibility is a pilot result, not assumed from the login host.
- One A100 allocation at a time, one GPU, eight CPUs and 48 GiB host memory; GPU peak below 36 GiB. Exclude `compute-0-1`, whose NFS startup failed in prior runs. No automatic concurrent jobs.
- Total ceiling: 48 A100 GPU-hours, counted across failed/completed allocations. Preparation/bootstrap/features use at most 4.5 hours, teacher adaptation at most 21 hours, student comparisons/replay at most 22.5 hours. Reaching a ceiling stops further submission and preserves partial evidence; it does not imply research completion.
- Pilot: 40-minute allocation, batch 8, FP32 forward/backward on at most 64 training records. If memory fails, retry once at batch 4 with accumulation 2; do not alter labels, lead count or sampling rate. Fix the successful setting for all comparable runs. No lower precision switch after seeing metrics.
- Four preprocessing contracts registered before fitting: HEED versus canonical frontal order, each with global z-score either without filtering or with the pinned upstream filtering. The pinned filter applies a 50-Hz notch with Q=30, zero-phase fourth-order 0.67-40 Hz bandpass, then subtracts a 0.4-second odd-width median baseline. Teacher normalization uses the entire 12-by-5000 matrix; no per-lead substitution.
- Source-only execution audit found WFDB16 missing-sample sentinels in four fitting records and one rhythm-validation record. The strict raw preprocessor remains available; the registered source-reader mode verifies NaNs against digital -32768 codes before zero filling, logs counts/flags and rejects unexplained NaNs/infinities. No evaluation record is removed. Original student tensor bytes remain the control. [WFDB source-format reference](https://archive.physionet.org/faq.shtml).
- Frozen teacher head probes: four contracts, two tasks, three seeds, at most 24 dense-head fits. AdamW, LR `1e-3`, decay `1e-4`, maximum 20 epochs, patience 4. Cache 1024-dimensional features in Slurm; rank contracts by mean endpoint score with lexicographic contract ID as exact tie-break.
- Optional teacher adaptation: at most six fits, one selected contract per endpoint and three seeds. Initialize from that seed's frozen-head checkpoint; unfreeze `stage_list[-1]` and `dense` only. AdamW body/head LRs `1e-5`/`1e-4`, decay `1e-4`, at most five epochs, each allocation at most three hours. Retain a frozen teacher if adaptation does not qualify.
- Student arms: supervised-only; supervised-only with train-balanced BN recalibration; rhythm transfer; pathology transfer; joint transfer. Three seeds give at most 15 fits, each at most 90 minutes including replay; skip arms whose teachers are ineligible.
- Student fits: same pinned warm-start, `res_layers[3]`, `res_layers[4]`, rhythm/pathology heads trainable; stem, earlier blocks, quality and biometrics head parameters frozen. Keep BatchNorm buffers frozen except the separately named train-only recalibration control. AdamW backbone/head LRs `5e-5`/`1e-3`, decay `1e-4`, gradient norm cap 1, at most eight epochs. Use existing source weights and observed-target logic, not a new sampling population.
- Rhythm KD temperature 2, coefficient 0.25, applied only to observed rhythm labels where teacher argmax agrees with the reference. Pathology soft-target BCE coefficient 0.25, applied only to observed labels where the teacher's frozen validation threshold agrees with the reference. Supervised losses retain coefficient 1; task losses are normalized by their own observed supports. No unconditional high-weight Gold constraint.

## Predeclared Validation Decisions

Recompute all controls under the same input and threshold rules before fitting.
Retain both task-specific controls and the primary warm-start; do not compare
scores from different cohorts/calibration procedures as though identical.

Teacher eligibility requires a three-seed mean gain of at least 0.01 in rhythm
macro-F1 or pathology Primary-5 threshold-crossfit macro-F1 respectively.
Rhythm accuracy may fall by at most 0.005; pathology Primary-5 macro-AP must
improve by at least 0.005. Store native ten-label results too. For classes
with at least 20 observed positives, recall may fall by at most 0.05; for
source strata with at least 100 cases, endpoint macro-F1 may fall by at most
0.02. Report smaller strata in full, not as passed guards. These are prospective
research choices, not confidence intervals or guaranteed improvements.

Before any teacher fit, optional final-stage feasibility is resolved as follows:
the head probe's three-seed mean macro-F1 must be noninferior to both retained
and primary controls; rhythm accuracy must remain within 0.005 and pathology
macro-AP must be noninferior too. Select the strongest supported preprocessing
contract for its three final-stage fits. This feasibility check does not
qualify a teacher for transfer: full endpoint, class/source and ensemble rules
above still apply after fitting.

Each eligible endpoint retains all three teacher seed checkpoints. Its transfer
teacher is their mean-logit ensemble, not the best-scoring seed. Require the
ensemble itself to satisfy the same endpoint/class/source rules; calibrate
its pathology thresholds on validation and freeze them before export.

A student arm qualifies against supervised-only if either endpoint's
three-seed mean macro-F1 improves by at least 0.01 and the other falls by
at most 0.005, with the same class/source guards and rhythm accuracy guard.
Keep a Pareto table rather than hiding single-task regressions. Among qualified
joint candidates choose the largest equal-weight mean of rhythm macro-F1 and
pathology threshold-crossfit macro-F1, tie-breaking by lower total fit time,
then arm name. Select epochs by that validation utility; a seed with no eligible
epoch retains its warm-start and is reported as a failed-improvement replicate.
For a single deployed student, preselect seed `20261014`: it must independently
satisfy the same promotion rules. Report its own results alongside the
three-seed summary; do not attribute ensemble or mean scores to its PTH.

If no recalibration/transfer arm qualifies against supervised-only, retain the
supervised continuation when its three-seed mean and fixed deployment seed
both improve the warm-start under the same rules. Otherwise retain the warm
checkpoint. A supervised-only gain is not attributed to distillation.

Use existing `patient_folds` and `crossfit_pathology` on observed native
validation labels; use full-validation thresholds only for subsequently frozen
tracking inference. Source/class AP, support and calibration folds accompany
every score. Validation bootstraps share patient tokens for overlapping records
across tasks; do not concatenate the two validation sets as independent cases.
Previously examined 2,171 and 672 cases are evaluated only after selection,
without further fit decisions. This plan does not assert a new untouched test.

### Task 1: Audited Paired Inputs

**Files:** Create `contracts.py`, `inputs.py`, `scripts/prepare_founder_transfer.py`, `configs/founder_transfer_2026-10-14.json`; test `tests/test_founder_transfer_inputs.py`.

**Interfaces:** `PairedRecord` fields are `key: str` (source/version/record), `patient_token: str | None`, `role: str`, `record_path: str`, `signal_filename: str`, `header_sha256: str`, `signal_sha256: str`, `window_start: int`, `window_samples: int`, `rhythm_label: int` (-1 unknown), `path_targets: tuple[float, ...]`, `path_mask: tuple[bool, ...]` (both length ten), `student_tensor_sha256: str`. `TensorPair` fields are `teacher: np.ndarray`, `student: np.ndarray`, `key: str`, `contract: str`. `build_pair_manifest(root: Path, config: dict) -> list[PairedRecord]`; `load_pair(record: PairedRecord, cache: Path, contract: str) -> TensorPair`.

- [ ] **Step 1: Write failing input-contract tests.** Assert allowlist size 22,567, native pathology support 16,413, training overlap zero by patient/signal/tensor, duplicate-key rejection, source-byte hash rejection, split-before-window rejection and preserved target masks. Assert swapped aVL/aVF follows the requested named order, mV/uV conversion gives equal physical input, duplicate/missing leads and unsupported units/frequencies fail, and student tensor bytes match `preprocess_primary9_signal`. Constant finite signals stay finite with a quality flag; nonfinite source data fail explicitly without silent case deletion.
  ```python
  def test_pair_shapes_and_original_student_bytes(pair_fixture):
      pair = load_pair(pair_fixture['record'], pair_fixture['cache'], pair_fixture['contract'])
      assert pair.teacher.shape == (12, 5000)
      assert pair.student.shape == (6, 1250)
      assert pair.student.tobytes() == pair_fixture['expected_student'].tobytes()
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_inputs.py -q`; require a failure because the new interfaces do not exist.** Do not accept a skip as the failing baseline.
- [ ] **Step 3: Implement the interfaces and CLI `--output --data-root --offline`.** Read the three existing training manifests and current validation/tracking/historical manifests through structured CSV readers. Resolve PTB patient IDs from pinned metadata. Use SHA-256 checks before WFDB physical reads, full named-lead ordering, native 500-Hz inputs, first ten seconds and explicit right-zero-padding flags for short records. Normalize physical units to mV; never interpolate missing channels. Export sorted keyed manifests, exclusions, target masks, four preprocessing IDs and all configuration/source hashes. Fail if any required fitting input is unavailable; no substituted records. Stream records rather than materializing a 5.4-GB teacher tensor in RAM.
- [ ] **Step 4: Run the focused tests and offline manifest preflight.** Require all input tests passing, expected counts, seven original input hashes unchanged and every referenced source header/signal hash checked. Keep metadata-derived IDs private where publication rights require it; use source-public record IDs in the reviewer manifest.
- [ ] **Step 5: Commit this independently usable input/preflight deliverable, its tests and updated root hash inventory.**

### Task 2: Safe Pinned Teacher Adapter

**Files:** Create `founder.py`, pinned vendor `net1d.py`, `LICENSE`, `UPSTREAM.md`, `requirements/founder-transfer.txt`; test `tests/test_founder_transfer_model.py`.

**Interfaces:** `validate_founder_state(state: dict[str, torch.Tensor]) -> None`; `load_founder(checkpoint: Path, *, outputs: int, seed: int) -> torch.nn.Module`; `founder_features(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` returns logits and 1024-dimensional features. Outputs allowed: 150, 9, 10.

- [ ] **Step 1: Write failing tests** for invalid SHA, unsupported output count, extra/missing body state keys, mismatched first convolution, restricted-load rejection and dense-only replacement. Assert teacher `[64,12,16]` input weights and no student state modification. With a synthetic serialization fixture, test the narrow NumPy scalar/dtype allowlist rather than allowing arbitrary globals.
  ```python
  def test_unexpected_body_key_is_not_hidden(state_with_extra_key):
      with pytest.raises(ValueError, match='body state'):
          validate_founder_state(state_with_extra_key)
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_model.py -q`; verify the missing-interface failure.**
- [ ] **Step 3: Implement restricted loading and construction.** Pin upstream code commit `04edac702b61c91face519774ddcc0cd712fef23`; verify vendored net1d bytes SHA `832bf90e05e20e62db6f9c7007acf7f0b1f739938ab80fa9b2c1ad438e8da7f7`. Preserve MIT notices. Verify teacher weight SHA `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997` before `torch.load(weights_only=True)`. Allow only documented NumPy scalar/dtype serialization constructors; never use `weights_only=False`. Instantiate the pinned Net1D configuration from the spec/source and load all 150-head keys strictly before replacing `dense` for 9/10 outputs. Use `return_features=True`, not undocumented feature hooks. Ignore upstream optimizer/scheduler/stored AUROC for TITAN selection.
- [ ] **Step 4: Run adapter unit tests and static architecture inspection.** Record exact teacher parameter count separately from state-buffer elements; require state keys/body shapes matched and no new deployed student parameters. GPU forward/backward belongs to Task 3, not this local check.
- [ ] **Step 5: Commit adapter, license notices, runtime pins, tests and inventory.**

### Task 3: Isolated Slurm Pilot and Retrieval

**Files:** Create `scripts/manage_founder_transfer_cedia.py`, `scripts/run_founder_transfer.py`; test `tests/test_founder_transfer_staging.py`.

**Interfaces:** Manager CLI actions `prepare`, `submit`, `status`, `fetch`, each requiring `--state-dir --bundle`; submit also requires `--stage bootstrap|pilot|features|teachers|targets|students|replay`. The `targets` allocation separates Task 5 export from fitting and counts toward the unchanged teacher budget. Worker CLI requires `--bundle --stage`. Reuse existing strict SSH transport `manage_joint_cedia_run.rpc`; do not inherit its old NPZ assumptions. Save one immutable submission identity per stage/run.

- [ ] **Step 1: Write failing tests** rejecting path traversal, links escaping the project root, repeated stage submission, missing pinned files, over-budget submission, nonzero/partial Slurm results and unsealed retrieval. Assert only the immutable stage selection/configuration reaches the worker.
  ```python
  def test_unsealed_result_is_not_success(partial_run):
      with pytest.raises(ValueError, match='sealed'):
          fetch_run(partial_run)
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_staging.py -q`; require the unimplemented-manager failure.**
- [ ] **Step 3: Implement `fetch_run(state: dict) -> dict` and the manager/worker stage boundary.** Resolve `$HOME/V4_CEDIA_LINK` and a fresh `99_RESEARCH_RUNS/TITAN-V4-Founder-Transfer-2026-10-14` target, reject existing-run overwrite, verify remote raw source availability/hashes, upload only missing exact bytes and pinned code/weights. Stage no validation targets in teacher-output caches. Bootstrap the new environment inside a compute allocation, record wheel/version/driver receipts and preserve the old environment. Request the exact resource ceilings above and track elapsed GPU-hours from `sacct`; keep SSH secrets and personal absolute paths out of public logs.
- [ ] **Step 4: Run staging tests, then bootstrap and pilot in Slurm after plan approval.** Require explicit CUDA device/A100 receipt, finite `[B,150]` teacher logits/features, finite adapted-head gradients, reproducible student control outputs, zero input mutation and measured peak memory below 36 GiB. Failed pilot prevents long teacher jobs. Status/fetch may not report success until completed exit code zero plus a valid result seal.
- [ ] **Step 5: Commit manager, worker, tests and sanitized pilot evidence/inventory.**

### Task 4: Endpoint Teacher Qualification

**Files:** Create `teachers.py`, `selection.py`; extend worker stages `features`/`teachers`; test `tests/test_founder_transfer_teachers.py`.

**Interfaces:** `fit_teacher(bundle: Path, *, endpoint: str, contract: str, seed: int, stage: str) -> dict` writes one sealed fit with checkpoint/predictions. `qualify_teachers(results: list[dict], controls: dict) -> dict` uses the predeclared rules and returns eligible endpoint records containing `checkpoints` (ordered three-seed path/SHA list), ensemble `thresholds`, `contract` and eligibility metrics.

- [ ] **Step 1: Write failing tests** rejecting non-training fit rows, label-order mismatch, missing task masks, threshold fitting on tracking records and selection from tracking metrics. Assert frozen-head fits leave every body tensor/buffer unchanged, adapted fits change only the final teacher stage/head, and three-seed aggregation never chooses the highest-scoring seed alone.
  ```python
  def test_unknown_pathology_has_zero_gradient(masked_batch):
      loss = teacher_pathology_loss(**masked_batch)
      loss.backward()
      assert masked_batch['logits'].grad[masked_batch['mask'] == 0].eq(0).all()
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_teachers.py -q`; require missing teacher-fit/selection implementation.**
- [ ] **Step 3: Implement the interfaces plus `teacher_pathology_loss(logits, targets, mask) -> torch.Tensor`.** Cache frozen features in allocation using Task 1 inputs and Task 2 adapter; compute 24 bounded head probes, select preprocessing by validation only, then at most six final-stage fits. Reuse observed-target losses/calibration/patient folds and report all ten pathology labels. Original 150-head outputs are diagnostic only; no ILMI/ISC_ approximate unions. Publish seed-level control comparisons and eligibility reasons before generating transfer caches.
- [ ] **Step 4: Run tests and the registered Slurm stages.** Require strict checkpoint reload, deterministic repetition of one selected fit within the budget, per-class/source metrics, threshold-crossfit versus full-calibration distinction, empty-mask finite zero loss, and endpoint qualification decisions reproducible from CSV/JSON alone. If no task qualifies, seal the negative comparison and skip distillation without rewriting the protocol to force qualification.
- [ ] **Step 5: Commit teacher trainer, selector, tests and completed qualification evidence/inventory.**

### Task 5: Keyed Soft-Target Export and Losses

**Files:** Create `distillation.py`; test `tests/test_founder_transfer_distillation.py`.

**Interfaces:** `export_teacher_targets(bundle: Path, qualification: dict) -> Path` exports training-only keyed NPZ/CSV; `join_teacher_targets(records: list[PairedRecord], cache: Path) -> dict[str, np.ndarray]`; `transfer_losses(student_logits: dict, targets: dict, *, temperature: float = 2.0, coefficient: float = 0.25) -> dict[str, torch.Tensor]` returns separate rhythm/pathology KD losses and eligible supports.

- [ ] **Step 1: Write failing tests** for shuffled-key joins, duplicates, missing/extra keys, wrong checkpoint SHA, wrong class order, tracking rows, invalid/nonfinite probabilities and independent sigmoid semantics. Assert zero gradient for every masked/disagreeing reference and no inferred negative labels.
  ```python
  def test_shuffled_teacher_rows_are_joined_by_key(records, shuffled_cache):
      joined = join_teacher_targets(records, shuffled_cache)
      assert joined['record_keys'].tolist() == [r.key for r in records]
      assert np.array_equal(joined['rhythm_logits'], shuffled_cache.expected_by_key)
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_distillation.py -q`; require a missing loss/export implementation failure.**
- [ ] **Step 3: Implement the interfaces.** Bind each row to source/record/window, source/tensor SHA, all three teacher checkpoint SHAs, contract SHA and observed masks; use `allow_pickle=False` caches. Store mean ensemble logits, softmax rhythm and independent pathology probabilities with explicit orders. Define target dictionary fields `record_keys`, `rhythm_labels`, `path_targets`, `path_mask`, `rhythm_teacher_logits`, `path_teacher_logits`, `path_thresholds`; absent qualified endpoints have explicit zero eligible supports, not fabricated logits. Apply the fixed agreement rules and validate frozen thresholds. Do not export a fitted validation/tracking target cache or an unqualified endpoint. Normalize by eligible support, returning a differentiable finite zero when support is empty.
- [ ] **Step 4: Run tests and training-only export in the selected Slurm allocation.** Require original reference hashes unchanged, exact allowlist coverage for qualified endpoints, no fitting-exclusion keys and export reload hash equality.
- [ ] **Step 5: Commit export/loss code, tests and cache manifests/inventory; keep large numerical rebuild caches ignored.**

### Task 6: Same-Architecture Student Comparison

**Files:** Create `students.py`; extend worker `students` stage and `selection.py`; test `tests/test_founder_transfer_students.py`.

**Interfaces:** `fit_student(bundle: Path, *, arm: str, seed: int) -> dict`; `select_students(results: list[dict], controls: dict) -> dict`. Arms and eligibility rules are exactly those defined above. Selection produces frozen chosen checkpoints/thresholds and a complete non-winning trial table.

- [ ] **Step 1: Write failing tests** for wrong warm-start hash, added heads/projection parameters, changed strides, mismatched target masks, teacher-ineligible arms and tracking-driven selection. Assert quality/biometrics parameters and default BN buffers remain identical, while shared-extractor output changes are measured rather than declared impossible.
  ```python
  def test_student_export_preserves_deployment_contract(exported_student):
      assert sum(p.numel() for p in exported_student.parameters()) == 1143751
      assert len(exported_student.res_layers) == 5
      assert exported_student.morphology_dim == 0
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_students.py -q`; require missing student-fit/selection functions.**
- [ ] **Step 3: Implement fits and selectors.** Strictly load the existing PTH, use the fixed trainable names/schedule/loss coefficients, existing metadata-only source weights and native masks. Recalibration uses fitting records only with a saved before/after buffer diff. Save optimizer/scheduler/RNG states for new runs, but do not invent historical resume states. Run at most 15 eligible comparisons; save every seed/epoch and guarded validation decision, including warm-start fallback.
- [ ] **Step 4: Run tests and bounded Slurm comparisons.** Require parameter/state/stride checks for every export, repeated training check on one candidate, fixed validation thresholds, all three seed outcomes and explicit task/source regressions. Stop at the budget even if 90/87 is not reached. Tracking inference starts only after the selection artifact is sealed and cannot reopen training selection.
- [ ] **Step 5: Commit student trainer, tests, selected and non-winning completed evidence/inventory.** Preserve earlier Gold and candidate files.

### Task 7: Source Replay and Reviewer Publication

**Files:** Create `scripts/verify_founder_transfer.py`, `tests/test_founder_transfer_evidence.py`, bundle `README.md`; update root `README.md` and `REPRODUCIBILITY.md` only with measured outcomes.

**Interfaces:** `verify_founder_bundle(bundle: Path, *, data_root: Path, offline: bool) -> dict`; CLI `--bundle --data-root --offline --output`. Worker replay stage writes per-record predictions, masks, frozen thresholds, metric tables, environment/Slurm receipts and nested SHA inventory.

- [ ] **Step 1: Write failing tests** rejecting incomplete record coverage, stale selection hashes, modified labels/source files, unsealed PTH/prediction pairs, incompatible preprocessing and a tracking result presented as untouched-test evidence. Assert reviewer recomputation produces recorded metrics from paired references/predictions without training.
  ```python
  def test_partial_predictions_fail_coverage(partial_bundle, data_root):
      with pytest.raises(ValueError, match='coverage'):
          verify_founder_bundle(partial_bundle, data_root=data_root, offline=True)
  ```
- [ ] **Step 2: Run `python -m pytest tests/test_founder_transfer_evidence.py -q`; require the missing verifier failure.**
- [ ] **Step 3: Implement verifier, replay and documentation.** Rebuild student tensors from original sources, reproduce all chosen validation decisions and then the frozen 2,171 tracking and 672 historical predictions. Never adjust labels to fit the checkpoint. Assert repeated inference decisions identical and cross-runtime probabilities within `atol=1e-6`, `rtol=1e-5`; if a threshold decision differs, treat replay as failed even when aggregate metrics match. Preserve public-source licenses/teacher notices, absent-source instructions and measured-versus-planned scope. Do not publish private identity fields or upstream 370-MB weights in Git.
- [ ] **Step 4: Verify focused tests, full suite, root inventory and exact Git archive replay.** Require source-based student prediction coverage, metric recomputation, sealed artifacts and a clean checkout; record whether dependency installation itself was freshly tested. Reuse existing inference/replay helpers only where their contracts genuinely match; add no historical claims of 90/87 from new fits.
- [ ] **Step 5: Commit measured code/evidence/docs, push `codex/audit-fixes`, verify PR #1 head and attach it to the task.** No merge, no silent main/Gold replacement. A failed qualification still produces a valid comparison package, not a deployed improvement claim.

## Completion and Execution Handoff

The first subsystem is complete only when input/teacher contracts are tested,
every submitted job has a terminal audited state, qualified transfers (if any)
have replayed predictions and the reviewer package verifies from the exact
published revision. Scientific improvement is a separate measured outcome.

Plan self-review must check interface names, exact task masks, prospective
selection criteria, five review-focus failure modes and budget accounting.
The user selected native implementation in this session with independent final
review. Execution is staged because teacher eligibility must be established
before transfer; no further design approval is required for this fixed plan.
