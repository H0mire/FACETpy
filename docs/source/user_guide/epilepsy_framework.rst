.. _epilepsy_framework:

===================================================
Epilepsy Analysis Framework (TCCC / EVMC / Fused)
===================================================

.. note::

   This page documents the epilepsy analysis framework **as it is currently
   implemented** in ``src/facet/Epilepsy/``. Every parameter, threshold and
   default quoted below is taken directly from the source. Where the current
   implementation deviates from the original publications (Ebrahimzadeh et al.
   2021; Grouiller et al. 2011) or where a parameter exists but is not used,
   this is flagged explicitly in
   :ref:`epilepsy_adaptations`.

   The framework was inspected statically; the numbers here were **not** obtained
   by executing the pipeline.


.. _epilepsy_overview:

1. Overview
===========

Purpose
-------

The epilepsy framework isolates and localises interictal epileptiform
discharges (IEDs, "spikes") in scalp EEG and produces both EEG-only detections
and (optionally) EEG-informed fMRI regressors. It bundles three related
analyses:

* **TCCC — Template Component Cross-Correlation** (temporal/morphological).
  Isolates the epileptic ICA component(s) by matching each component's
  time course against a patient-specific IED template, then validates the
  surviving components by their scalp topography.
* **EVMC — Epileptic Voltage Map Correlation** (spatial/topographic).
  Builds an averaged-spike "epileptic map" and correlates it continuously
  against the EEG to quantify how strongly that field is present over time.
* **Fused pipeline** (integrated temporal + spatial). Runs both methodologies
  on the *same* recording and the *same* refined spike times and fuses their
  two evidence traces into discrete detections and a graded fMRI regressor.

Overall workflow
----------------

.. code-block:: text

   .mat file ──▶ preprocessing ──▶ TCCC component selection
                                     │  (template + multi-run ICA + acceptance)
                                     ├──▶ spatial validation of components
                                     │        (scalp-map |r| ≥ 0.50)
                                     ├──▶ EVMC map + spatial-correlation trace
                                     └──▶ fused temporal×spatial evidence
                                              ├──▶ EEG detections
                                              └──▶ fMRI regressor (if has_fmri)

Relationship between the three analyses
---------------------------------------

The single entry point is
:func:`run_fused_pipeline` in ``src/facet/Epilepsy/pipeline.py``. It calls the
TCCC stage (:func:`run_ebrahimzadeh_pipeline`), optionally the EVMC/Grouiller
regressor stage (:func:`run_grouiller_pipeline`, only when ``has_fmri=True``),
then **always** applies component spatial validation
(``apply_spatial_validation``) and finally builds the fused evidence and
detections. The three "analyses" are therefore not independent programs — TCCC
and EVMC are stages of the one fused pipeline, and the standalone EVMC/Grouiller
regressor is produced as a by-product when fMRI output is requested.

Implementing files
-------------------

============================================  ==================================================
Stage                                         File
============================================  ==================================================
Preprocessing, loading, spike parsing         ``helpers/preprocessing.py``
IED template construction                     ``helpers/shared_utils.py`` (``build_template``)
TCCC selection, ICA, acceptance               ``helpers/correlation_utils.py``
Component spatial validation                  ``helpers/spatial_validation.py``
EVMC maps, spatial correlation, HRF, regressors ``helpers/regressors.py``
Pipeline orchestration                        ``pipeline.py``
Result container                              ``Models/pipeline_results.py`` (``TemplateICADetection``)
Evaluation / reporting / figures              ``evaluation/`` (``evaluate_subject.py``, ``evaluation_helpers.py``, ``evaluate_group.py``, ``plots.py``, ``config.py``)
============================================  ==================================================


.. _epilepsy_reproducing:

Reproducing the Analysis
========================

A short, practical guide to running the epilepsy analysis end to end.

1. Configuration & data
-----------------------

* Install dependencies once with ``poetry install`` and run commands via
  ``poetry run`` from the repository root.
* Subject ``.mat`` files go in the **existing dataset folder used by**
  ``src/facet/Epilepsy/evaluation/config.py`` — that is
  ``MAT_DIR = <project_root>/examples/datasets/MAT_Files`` (one file per subject,
  e.g. ``DA00100T.mat``). Set project_root in config.py to the local FACETpy repository 
  path so that MAT_DIR resolves to examples/datasets/MAT_Files
* Main config values (in ``config.py``): ``SFREQ = 500.0``, ``TR = 2.5``,
  ``TH_RAW = 0.85``, ``HALF_WIN_S = 0.15``, ``MATCH_TOL_S = 0.1``. The
  single-subject evaluation always runs the pipeline with ``has_fmri=True`` and
  ``tr=TR``.

2. Process a single subject (no batch-list edit needed)
-------------------------------------------------------

A single subject can be processed **directly** from its ``.mat`` in ``MAT_DIR``;
it does **not** need to be added to ``subjects_batch.txt``.

.. code-block:: bash

   # by subject id (resolves DA00100T.mat in MAT_DIR)
   poetry run python -m facet.Epilepsy.evaluation.evaluate_subject --mat-file DA00100T

   # or a filename / absolute path
   poetry run python -m facet.Epilepsy.evaluation.evaluate_subject --mat-file /abs/path/to/file.mat

With no arguments it defaults to ``DA00100S.mat``.

3. Process multiple subjects (batch list)
-----------------------------------------

For batch processing via the subject-list file, the subject ID **must also be
added** to ``src/facet/Epilepsy/evaluation/subjects_batch.txt`` (one id per line;
blank lines are ignored), and its ``.mat`` file must be present in ``MAT_DIR``.
Each id is resolved to ``<id>.mat`` by ``_resolve_mat_path``.

.. code-block:: bash

   # run everything in subjects_batch.txt, using all CPU cores
   poetry run python -m facet.Epilepsy.evaluation.evaluate_subject \
       --subjects-file src/facet/Epilepsy/evaluation/subjects_batch.txt --jobs -1

   # or list a few subjects inline (also no batch-list edit needed)
   poetry run python -m facet.Epilepsy.evaluation.evaluate_subject --mat-file DA00100T DA00100S

Runs are deterministic (fixed ICA seeds), so parallel (``--jobs``) results match
sequential ones.

4. Group aggregation
--------------------

After the per-subject runs, aggregate them into group tables and figures:

.. code-block:: bash

   poetry run python -m facet.Epilepsy.evaluation.evaluate_group
   # useful flags: --skip-existing (skip already-done subjects),
   #               --aggregate-only (only aggregate existing CSVs),
   #               --excel (also write group_summary.xlsx)

5. Where outputs are written
----------------------------

* Per subject: ``src/facet/Epilepsy/evaluation/results/{subject}/`` (the
  ``results_{subject}_*.csv`` files, ``arrays_{subject}.npz`` and
  ``fig_{subject}_*.png``).
* Group level: ``src/facet/Epilepsy/evaluation/results/group/``.

The ``results/`` tree is git-ignored (regenerated locally, not versioned).


.. _epilepsy_input:

2. Input requirements
=====================

EEG data structure
------------------

Input is a MATLAB ``.mat`` file (``helpers/preprocessing.py::load_mat_to_mne``):

* Key ``eeg_data`` — a 2-D array of shape ``(n_channels, n_samples)``.
* **Units** are auto-detected: if ``abs(eeg).max() < 0.1`` the data is assumed to
  be already in **volts**; otherwise it is assumed to be **µV** and divided by
  ``1e6``. (MNE stores volts internally.)
* An EDF loader (``load_edf_to_mne``) also exists but is not used by the fused
  pipeline / evaluation path.

Sampling rate
-------------

There is **no sampling rate stored in the ``.mat`` file**. It is assumed to be
``sfreq = 500.0`` Hz everywhere (``config.SFREQ`` and the default argument of the
pipeline functions). The Raw object is created with this value regardless of the
true acquisition rate.

Channel names and montage
--------------------------

Channel names are taken, in order of preference, from the ``.mat`` keys
``channels``, ``channel_labels``, ``labels`` or ``chan_names``. If none is
present and the array has exactly **29 channels**, a hard-coded VEPISET layout is
assumed:

.. code-block:: text

   Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz
   PG1 PG2 A1 A2  ECG1 ECG2  EMG1 EMG2 EMG3 EMG4

Otherwise names default to ``EEG1 … EEGn``.

Channel **types** are inferred from the names (``_infer_ch_types``):
``ECG``/``EKG`` → ``ecg``; ``EMG`` → ``emg``; ``EOG`` → ``eog``;
``PG1``/``PG2``/``A1``/``A2`` → ``misc``; everything else → ``eeg``. Only
``eeg``-typed channels participate in template-channel selection, ICA and map
construction.

A ``standard_1020`` montage is applied with ``on_missing='ignore'``. A real
montage (electrode positions) is **required** for the coordinate-based region
labelling used in the evaluation, but the core detection/regressor pipeline runs
without it.

Spike / event annotations
--------------------------

Spike times come from the ``.mat`` key ``events``
(``preprocessing.py::parse_spike_times``): an array of rows where **column 0** is
a time string (seconds) and **column 2** is a label. A row is accepted as a spike
only when its label equals the marker ``"!"`` (the default ``label_marker``).
Times are parsed to ``float`` seconds, de-duplicated and sorted ascending.
Additional marker strings can be supplied but are not used by the default path.

fMRI / TR information
---------------------

No fMRI data is read. The fMRI **repetition time** ``tr`` is a scalar used only
for regressor resampling; its default is ``TR = 2.5`` s. Regressors are produced
only when ``has_fmri=True``. **The single-subject evaluation always calls the
pipeline with** ``has_fmri=True, tr=2.5`` (``run_pipeline_for_subject``), so the
Grouiller and fused fMRI regressors are generated for every subject using an
**assumed** TR of 2.5 s even for EEG-only datasets.


.. _epilepsy_preprocessing:

3. Preprocessing
================

All filtering routes through the shared FACETpy processors
(``NotchFilter``, ``BandPassFilter``) via
``helpers/preprocessing.py::apply_facet_filters`` so that epilepsy filtering is
identical to the rest of the toolbox.

``prepare_eeg_data(mat_path, sfreq=500.0)`` performs:

1. **Load** the ``.mat`` into an MNE Raw (volts, inferred channel types,
   ``standard_1020`` montage).
2. **Filter** (``filter_eeg``): a notch at **50 Hz and 100 Hz** on EEG channels,
   followed by a **1–30 Hz band-pass** on EEG channels.
3. Produce a second object ``raw_ica`` by applying an **additional 1–30 Hz
   band-pass** to the already-filtered ``raw``. In practice both objects are in
   the 1–30 Hz band; ``raw_ica`` is the object handed to ICA/template selection,
   while ``raw`` (notch + 1–30 Hz) is stored on the detection and used for the
   epileptic-map/spatial-correlation stages.
4. **Parse** spike times with marker ``"!"``.

Units/scaling
-------------

Data is held in volts internally (converted at load if it looked like µV).
No additional re-referencing is performed at load time; the epileptic-map stage
applies its own baseline correction and GFP normalisation, and the spatial
correlation is effectively reference-free because each time frame is
GFP-normalised and mean-centred before correlation.

ICA-specific preprocessing
--------------------------

ICA is fit on ``raw_ica`` (1–30 Hz). No amplitude thresholding, epoch rejection
or whitening beyond MNE's own ICA defaults is applied before fitting. Non-brain
channels are excluded by channel type, not removed from the file.

.. warning::

   The notch frequencies (50/100 Hz) assume **European mains**. There is no
   configuration switch for 60 Hz.


.. _epilepsy_tccc:

4. TCCC pipeline (step by step)
===============================

Entry point: ``pipeline.py::run_ebrahimzadeh_pipeline`` →
``correlation_utils.py::select_components_template_ica``. Defaults:
``half_win_s=0.15``, ``th_raw=0.85``, ``max_template_spikes=20``,
``template_band=(1., 30.)``.

4.1 Patient-specific template (``shared_utils.build_template``)
---------------------------------------------------------------

* **Template channel selection.** Over EEG channels only, the channel with the
  largest **cumulative peak-to-peak amplitude** summed across all spike windows
  is chosen. Restricting to EEG channels prevents ECG/EMG dominating.
* **Window length.** ``half_win_s = 0.15`` s → a ±150 ms window (301 samples at
  500 Hz) per spike.
* **Baseline correction.** Each snippet is baseline-subtracted using the mean of
  the ``baseline_ms = (-120, -20)`` ms pre-spike window.
* **Polarity handling.** Each snippet is flipped so the main deflection is
  positive (if ``|min| > |max|`` the snippet is multiplied by ``-1``).
* **Alignment.** Each snippet is rolled so its largest absolute deflection sits
  at the window centre, removing annotation lag; the **refined spike time** is
  recorded from this shift.
* **Minimum spikes.** At least **5** valid segments are required, else a
  ``ValueError`` is raised.
* **Template augmentation of the segment set (quality selection).** If more than
  ``max_spikes`` (=20 here) valid segments exist, only the 20 segments most
  correlated with a **leave-one-out grand average** are kept (each segment scored
  against the mean of all *other* segments, avoiding self-inclusion bias).
* **Averaging & normalisation.** The kept segments are averaged; optional 5-point
  smoothing is **off** by default; the average is **z-scored**
  (``(T - mean) / (std + 1e-12)``) to give ``template_z``.

4.2 Spike-set augmentation (``correlation_utils.augment_template``)
-------------------------------------------------------------------

Separately from the segment quality selection above, the **annotated spike set**
is augmented only when there are **fewer than 10** annotated spikes. On the
template channel, the normalised sliding template correlation is computed and
peaks with ``r ≥ 0.96`` (``high_r_min``) and a **0.15 s refractory** are detected.
A detected peak is added only if it lies more than 0.15 s from every existing
annotation. The augmented, sorted spike set becomes ``refined_times``
downstream.

4.3 Repeated ICA (``correlation_utils.multi_run_ica``)
------------------------------------------------------

* **Number of runs.** ``n_runs = 10``.
* **Algorithm & components.** MNE ``ICA`` with ``method='infomax'`` and
  ``n_components = min(20, n_eeg)``; each run uses ``random_state = run`` (0–9).
  ICA is fit on the already-1–30 Hz-filtered data.
* **λ (component weight).** For each component, ``λ`` = the **L2 norm of its
  mixing-matrix column**.
* **Cross-run clustering.** Every mixing column from every run is unit-normalised
  and clustered greedily by **absolute cosine similarity ≥ 0.8**
  (``cluster_threshold``), seeding each cluster from the highest-λ unassigned
  vector. Cluster centroids are sign-aligned, averaged and re-normalised.
* **Ranking / selection.** Clusters are ranked by (1) number of distinct runs
  they appear in ("most often"), then (2) mean λ. The top ``max_keep = 3``
  centroids are returned.

4.4 Final ICA and matching
--------------------------

A **single final ICA** is fit (``method='infomax'``, ``random_state=97``,
``n_components = min(20, n_eeg)``). The three stable cluster centroids are mapped
onto this fit's components by greedy **absolute Pearson correlation** of mixing
columns (``match_clusters_to_ica``), giving the candidate component indices.

4.5 Artifact rejection (``correlation_utils.find_artifact_components``)
----------------------------------------------------------------------

MNE detectors flag artifact ICs to remove from the candidate pool:
``find_bads_ecg`` (correlation, ``threshold='auto'``; needs an ECG channel),
``find_bads_eog`` (needs an EOG channel) and ``find_bads_muscle`` (spectral
heuristic). Detectors whose required channel is absent are skipped. The
exclusion is applied only if at least one candidate remains afterwards.

4.6 Temporal template matching & acceptance
-------------------------------------------

For each candidate component:

* The component time course is band-passed (``band_comp = (1., 30.)``) and matched
  against ``template_z`` with the **normalised sliding cross-correlation**
  ``sliding_template_correlation`` (locally mean/variance-normalised
  ``correlate(signal_z, template_z, 'same') / (L·std)``).
* **Acceptance metric** (``check_component_acceptance``): a **0.3 s window**
  (``window_s``) is slid at each IED time; the maximum ``|r|`` per IED is taken;
  the decision **score** is the **0.90 quantile** (``single_trial_quantile``) of
  those per-IED maxima.
* **Threshold.** A component is accepted if ``score ≥ th_raw = 0.85``.

4.7 Temporal fallback
---------------------

If **no** component reaches 0.85, the single **best-scoring** candidate is kept
and ``fallback_used=True`` is recorded. Accepted components are then ordered by
**descending median per-window correlation**.

4.8 Spatial validation of temporally selected components
--------------------------------------------------------

Implemented in ``helpers/spatial_validation.py::apply_spatial_validation`` and
**always** applied by the fused pipeline.

* **IED spatial reference.** One patient-specific reference map is built with
  ``_build_epileptic_map(raw, refined_times, half_win_s, band=(1,30))`` — the
  GFP-peak topography of the averaged annotated spikes (see EVMC below).
* **Per-component score.** Each temporally accepted component is reconstructed on
  its own (``ica.apply(include=[idx])``), its own GFP-peak map is built the same
  way, and it is scored by the **absolute whole-map Pearson correlation** ``|r|``
  with the reference (``_abs_map_corr``; requires ≥3 overlapping channels).
* **Threshold.** A component is **spatially accepted** iff
  ``|r| ≥ SPATIAL_ABS_CORR_THRESHOLD = 0.50``.
* **Spatial fallback.** If no component passes 0.50, the single component with
  the highest available ``|r|`` (or, if all reconstructions failed, the highest
  median window correlation) is kept, ``spatial_gate_fallback_used=True``, and the
  spatially-accepted set is reported as empty.

4.9 Retained set vs representative component
--------------------------------------------

* The **final TCCC set** stored on the detection is reduced to the spatial
  survivors (or the single fallback component); ``detection.accepted_components``
  and its aligned time courses are updated in place, with the **representative
  placed first**.
* The **representative standalone TCCC component** is the **highest-|r| survivor**
  (or the fallback). It is used for the single-map TCCC topography/peak/region and
  drives the TCCC regressor. The **fused** reconstruction, by contrast, uses **all**
  final TCCC components (see §6).

4.10 TCCC regressor (``regressors.compute_and_attach_ica_regressors``)
----------------------------------------------------------------------

* Uses the **leading** accepted component (``component_timecourses[0]`` — the
  representative after validation).
* ``generate_hrf_regressors`` convolves that time course with a double-gamma HRF
  kernel at peak delays ``peaks_s = [3, 5, 7, 9]`` s and resamples each to
  ``n_tr = floor(duration / tr)``; the stored ``regressor_ica`` is the **5 s**
  variant.
* If ``tr is None`` (EEG-only), ``regressor_ica`` / ``regressors_ica_all`` are
  explicitly **cleared** (set to ``None``).


.. _epilepsy_evmc:

5. EVMC pipeline (step by step)
===============================

Implemented in ``helpers/regressors.py``.

5.1 Epileptic voltage map (``_build_epileptic_map``)
----------------------------------------------------

* **Band-pass** a copy of the EEG to ``band = (1., 30.)`` Hz.
* **Epoch** each spike at ``half_win_s = 0.15`` s (±150 ms); **≥3** valid epochs
  required.
* **Baseline correction.** Each epoch is corrected by the mean of its first
  ``hw`` samples (the pre-spike half-window).
* **Average** the epochs.
* **GFP** = spatial standard deviation across channels at each time point.
* **Peak search** is restricted to **±0.04 s (40 ms)** around the spike mark, so
  the map is the spike's field, not the largest-GFP frame in the epoch.
* **Map = voltage vector at the GFP peak, divided by that GFP value**
  (unit-GFP-normalised topography), shape ``(n_channels,)``.

5.2 Continuous spatial correlation (``_compute_spatial_correlation_timecourse``)
--------------------------------------------------------------------------------

* Band-pass the EEG (1–30 Hz).
* At every time frame, divide by the frame GFP, mean-centre, and compute the
  **Pearson correlation across channels** between the (mean-centred) map and the
  (mean-centred) frame.
* **Polarity treatment / squaring.** The returned trace is the **square**
  ``corr²`` (polarity-invariant; equivalently ``|corr|²``).

5.3 HRF convolution & resampling (``build_grouiller_regressor``)
----------------------------------------------------------------

* Convolve ``corr²`` with the **canonical double-gamma HRF** (``_double_gamma_hrf``;
  ``peak_time=6``, ``undershoot=16``, ``ratio=6``; normalised to unit max) using a
  **20 s** kernel at the EEG rate.
* **Resample to TR** by linear interpolation at ``n_tr = floor(duration / tr)``
  times.
* Returns ``(regressor, epileptic_map, corr_sq)``; ``corr_sq`` is the pre-HRF,
  EEG-rate trace exposed for EEG-resolution timing comparisons.


.. _epilepsy_fused:

6. Fused pipeline (step by step)
================================

Implemented in ``pipeline.py::run_fused_pipeline``. Defaults: ``fused_k=6.0``,
``match_tol_s=0.1``, ``band=(1., 30.)``, ``has_fmri=False``, ``tr=2.5``.

6.1 Components used for reconstruction
--------------------------------------

After spatial validation, ``fused_components = list(detection.accepted_components)``
— i.e. **all** final TCCC components (the spatial survivors, or the single
fallback). Components that failed spatial validation are excluded.

6.2 Fused spatial reference
---------------------------

The final components are reconstructed **together** in one physical
reconstruction (``ica.apply(include=fused_components)``), and the epileptic map is
built from that clean EEG with ``_build_epileptic_map(recon, refined_times)`` — one
real topography (component maps are never blended).

6.3 Temporal evidence
---------------------

Each fused component time course is sign-aligned to the template (sign taken from
the peak of its signed sliding correlation) and summed into a composite; the
temporal trace is ``r_temporal = |sliding_template_correlation(normalise(composite),
template_z)|``.

6.4 Spatial evidence
--------------------

``r_spatial = sqrt(clip(corr², 0))`` where ``corr²`` is
``_compute_spatial_correlation_timecourse(raw, epileptic_map)`` — i.e. the
**absolute** spatial correlation ``|corr|`` at each frame. Traces are truncated to
the common length.

6.5 Local / background normalisation
------------------------------------

Each trace is converted to a robust **relative elevation** over a **10 s** moving
baseline window (``base_win = round(10·sf)``):

.. code-block:: text

   μ  = moving_mean(x, base_win)                 # uniform_filter1d
   ex = clip(x - μ, 0, None)
   MAD = moving_mean(|x - μ|, base_win) + 1e-9
   z  = ex / MAD

giving ``z_t`` (temporal) and ``z_s`` (spatial).

6.6 Permitted temporal shift / alignment
-----------------------------------------

The spatial elevation is aligned to the temporal peak with a **rolling maximum**
over ±``match_tol_s`` (=0.1 s): ``z_s = maximum_filter1d(z_s, size=2·tol+1)`` with
``tol = round(match_tol_s·sf)`` (±50 samples at 500 Hz).

6.7 Geometric-mean combination & adaptive threshold
---------------------------------------------------

* **Combination.** ``r_fused = sqrt(z_t · z_s)`` (geometric mean of the two
  positive elevations — a spike needs waveform **and** topography to rise
  together).
* **Adaptive threshold.** ``height = median(r_fused) + fused_k · MAD``, with
  ``MAD = median(|r_fused − median(r_fused)|) + 1e-9`` and ``fused_k = 6.0``
  (self-calibrating per recording).

6.8 Peak detection
------------------

``detect_peaks`` (SciPy ``find_peaks``) with the adaptive ``height`` and a
**minimum peak distance / refractory of 0.5 s** (``round(0.5·sf)``). Detected peak
indices are converted to ``detections_sec``.

6.9 Continuous fused regressor (only when ``has_fmri and tr``)
--------------------------------------------------------------

* ``graded = r_temporal² · r_spatial²``.
* Convolve with the double-gamma HRF (20 s kernel at EEG rate).
* **Resample to TR** by linear interpolation → ``fused["regressor_fused"]``.


.. _epilepsy_fallback:

7. Fallback behaviour
=====================

The pipeline distinguishes several states, which downstream code can read from
``detection`` and ``results["spatial_gate"]``:

* **Successful execution (temporal + spatial).** One or more components pass the
  temporal criterion (score ≥ 0.85) **and** one or more of those pass the spatial
  criterion (|r| ≥ 0.50). ``accepted_components`` = spatial survivors;
  ``spatial_gate_fallback_used = False``; ``ica_selection_stats['fallback_used'] =
  False``. Downstream (TCCC map, fused reconstruction, regressors) uses these
  components.
* **Component satisfies the temporal criterion.** Recorded as a temporally
  accepted candidate (``baseline_accepted_components`` in the spatial summary).
* **Temporal fallback.** No component reached 0.85 → the best-scoring candidate is
  kept; ``ica_selection_stats['fallback_used'] = True``. Everything downstream
  still runs on this single component.
* **Component satisfies the spatial criterion.** Appears in
  ``spatial_accepted_components`` (|r| ≥ 0.50).
* **Spatial fallback.** No component reached |r| ≥ 0.50 → a single best-|r|
  component is kept; ``spatial_gate_fallback_used = True`` and
  ``spatial_accepted_components = []`` (the fallback is exposed only as
  ``final_tccc_accepted_components`` / the representative). Fused/TCCC still
  produce output from that one component.
* **No accepted components at all** (detection ``None`` or empty): ``results["fused"]
  = None`` and the pipeline returns early (no spatial validation, no fused output).

In every non-empty case, the fused reconstruction receives **all** components in
``detection.accepted_components``, and the standalone TCCC map/regressor receives
the **single representative** component.


.. _epilepsy_outputs:

8. Outputs
==========

Result container
----------------

The TCCC stage returns a ``TemplateICADetection`` dataclass
(``Models/pipeline_results.py``) with, among others: ``template_z``,
``refined_times``, ``accepted_components``, ``component_timecourses``,
``hrf_regressors``, ``ica``, ``regressor_ica``, ``regressors_ica_all``, ``raw``,
``original_spike_sec``, ``per_component_window_corr``, ``ica_selection_stats``.

``run_fused_pipeline`` returns a dict with keys ``"detection"``,
``"regressor_grouiller"`` and ``"regressor_ebrahimzadeh"`` (only when
``has_fmri``), ``"spatial_gate"`` (the validation summary), and ``"fused"`` =
``{"epileptic_map", "r_temporal", "r_spatial", "detections_sec",
"regressor_fused"?}`` (or ``None`` if nothing was accepted).

Subject-level files (``evaluation/evaluate_subject.py``)
--------------------------------------------------------

Written under ``src/facet/Epilepsy/evaluation/results/{subject}/``:

============================================================  =======================================================
File                                                          Contents
============================================================  =======================================================
``results_{subject}_summary.csv``                             one row: spike counts, template channel, accepted
                                                              component indices, thresholds, spatial-validation
                                                              summary, cross-method concordance, map correlations
``results_{subject}_component_detail.csv``                    one row per accepted component (window-corr stats, λ)
``results_{subject}_spatial_gate_detail.csv``                 per-candidate spatial log (temporal score, ``|r|``,
                                                              accepted/representative flags)
``arrays_{subject}.npz``                                      ``regressor_ebrahimzadeh``, ``regressor_grouiller``,
                                                              ``template_z``, ``epileptic_map``
``fig_{subject}_acceptance.png``                              acceptance summary (annotated/augmented/accepted)
``fig_{subject}_window_corr_distribution.png``                per-window correlation distribution
``fig_{subject}_template.png``                                the IED template
``fig_{subject}_ica_topomaps.png``                            accepted-component scalp topographies
``fig_{subject}_grouiller_map.png``                           EVMC/Grouiller epileptic map
``fig_{subject}_fused_map.png``                               fused epileptic map
============================================================  =======================================================

Group-level files (``evaluation/evaluate_group.py``)
----------------------------------------------------

Written under ``results/group/``: ``group_summary.csv``,
``group_component_detail.csv``, ``group_stats_summary.csv`` (median/IQR/range),
``group_stats_counts.csv``, ``group_accepted_components_distribution.csv``,
``group_template_channel_distribution.csv``, ``group_spatial_gate_detail.csv``,
``fig_group_acceptance.png``, ``fig_group_template_channel_distribution.png``, and
an optional ``group_summary.xlsx``.

.. note::

   The ``results/`` and ``results_tccc_spatial/`` output trees are **git-ignored**
   (regenerated locally, not versioned).


.. _epilepsy_adaptations:

9. Implementation-specific adaptations, assumptions & flags
===========================================================

Differences from the original methods
-------------------------------------

* **Spatial validation is scalp-space, not source-space.** The component spatial
  test is a fixed **whole-map |Pearson r| ≥ 0.50** on scalp topographies
  (``spatial_validation.py``). It is a pragmatic surrogate for, and **not**
  equivalent to, Ebrahimzadeh et al.'s source/dipole-space **50 mm** criterion.
  The pipeline has no forward model, inverse solution or dipole localisation.
* **EVMC map peak is windowed.** The GFP-peak search is constrained to ±40 ms of
  the spike mark, a stabilising choice not specified in Grouiller et al.
* **Repeated-ICA specifics.** Infomax ICA, ``n_components = min(20, n_eeg)``,
  cross-run clustering by absolute cosine similarity ≥ 0.8, selection by run
  frequency then mean λ, top 3 clusters — a concrete instantiation of the paper's
  "10 runs / most frequent / highest average λ / 3 components".
* **Template augmentation** (adding high-correlation detections when < 10 spikes,
  ``r ≥ 0.96``) and **leave-one-out segment quality selection** (best 20 segments)
  are implementation choices used as automated proxies for manual spike
  selection.
* **The evaluation always requests fMRI output** (``has_fmri=True``, ``tr=2.5``),
  so Grouiller and fused regressors are generated with an assumed TR even without
  real fMRI.

Parameters that exist but are **not used** (flagged discrepancies)
------------------------------------------------------------------

* ``run_fused_pipeline(spatial_th=0.85)`` — declared and documented but **never
  referenced** in the function body. The operative spatial threshold is
  ``SPATIAL_ABS_CORR_THRESHOLD = 0.50`` in ``spatial_validation.py``; fused
  detection uses ``fused_k`` relative-elevation, not ``spatial_th``.
* ``multi_run_ica(band_ica=(1., 100.))`` — declared but **not used**; ICA is fit on
  the already 1–30 Hz-filtered data, so the effective ICA band is **1–30 Hz**, not
  1–100 Hz.
* ``match_tol_s`` in ``select_components_template_ica`` / ``run_ebrahimzadeh_pipeline``
  — passed through but **not used** in TCCC selection (temporal acceptance uses a
  fixed 0.3 s window). ``match_tol_s`` **is** used in the fused stage for the ±0.1 s
  spatial-to-temporal alignment.

Assumptions & limitations visible from the code
-----------------------------------------------

* **Fixed 500 Hz** sampling rate (not read from data); **50/100 Hz** notch
  (European mains); **1–30 Hz** analysis band throughout.
* **Unit auto-detection** via a magnitude heuristic (``abs_max < 0.1`` ⇒ volts).
* **29-channel VEPISET fallback** montage is hard-coded; other layouts get generic
  names and no clinical-lobe labels.
* **Determinism.** The final ICA uses a fixed seed (``random_state=97``) and the
  10 multi-run ICAs use seeds 0–9, so a given ``.mat`` yields identical results
  across runs.
* **Dependence caveat.** The component spatial-validation reference map is derived
  from the same annotated spikes that drive the EVMC map, so "TCCC vs EVMC"
  spatial agreement is partly dependent and must not be read as independent
  validation. (The separate ``spike_field_consistency`` metric — inter-spike
  topographic consistency — is a *distinct* non-circular EEG-level check and is
  **not** the component spatial-validation gate, despite both concerning
  "spatial" information.)
* **Standalone TCCC map vs fused map.** With one final component they are identical
  by construction; with ≥2 they differ (TCCC map = representative only, fused map =
  all survivors).
