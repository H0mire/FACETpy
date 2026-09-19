Thesis evidence index
=====================

Generated from ``catalog.yaml``. Source records retain their original fields.

Find a thesis figure or table
-----------------------------

.. _ma-caption-0050:

* **- Figure  1  Gradient artifact (top, ±12 mV) dwarfs the AAS-corrected EEG (bottom, ±116 µV) by roughly two orders of magnitude on the same y-axis.**
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/datasets_evaluation`.
  Generator: :download:`source <../../../tools/plotting/artifact_amplitude.py>`.
  Command: ``uv run python tools/plotting/artifact_amplitude.py --dataset <proof-fit.npz> --out <figure.png>``.

.. _ma-caption-0119:

* **Tab le   1  Overview of evaluation metrics of FACETpy . ↓ = „the  Lower  the  better“ ; ↑ = „the higher the  better“ ; → 1 = the closer to one the better; → 0 = „the closer to zero the  better“**
  :doc:`/thesis_reference/code_provenance`.

.. _ma-caption-0188:

* **Figure  2  Screenshot of a running FACETpy pipeline    [26]**
  :doc:`/thesis_reference/code_provenance`.
  :download:`Original thesis figure <../../../docs/source/_static/thesis/figure02_pipeline_screenshot.png>`.
  Original illustration extracted unchanged from the thesis. Its editable figure source was not recovered; the current library and existing user-guide diagrams document the maintained architecture.

.. _ma-caption-0203:

* **Figure  3  Comparison of the FACETpy architecture. Left: the legacy version (0.1.0) encapsulates the entire correction workflow in a single stateful Facet object; successive method calls mutate hidden internal state (_eeg, _analytics, _correction, _evaluation)  with  no explicit data hand-off between steps. Right: the current version (2.1.0) models the same workflow as a declarative Pipeline of independent Processor objects, between which an immutable  ProcessingContext  ( ctx ) is passed explicitly from stage to stage.**
  Experiments: ``refactoring_engineering_indicators``.
  :doc:`/thesis_reference/phase_0_legacy`.
  :download:`Original thesis figure <../../../docs/source/_static/thesis/figure03_architecture_comparison.png>`.
  Original illustration extracted unchanged from the thesis. Its editable figure source was not recovered; the current library and existing user-guide diagrams document the maintained architecture.

.. _ma-caption-0215:

* **Figure  4  UML-style class diagram of Processor / Pipeline / Context / Registry /  ParallelExecutor**
  Experiments: ``refactoring_engineering_indicators``.
  :doc:`/thesis_reference/phase_0_legacy`.
  :download:`Original thesis figure <../../../docs/source/_static/thesis/figure04_core_contracts.png>`.
  Original illustration extracted unchanged from the thesis. Its editable figure source was not recovered; the current library and existing user-guide diagrams document the maintained architecture.

.. _ma-caption-0220:

* **Figure  5 A four-to-six-step correction reduces to a single Pipeline([...] ).run () constructor call (right), replacing the legacy chain of imperative method calls (left).**
  Experiments: ``refactoring_benchmark_legacy_vs_v2``.
  :doc:`/thesis_reference/phase_0_legacy`.
  :download:`Original thesis figure <../../../docs/source/_static/thesis/figure05_pipeline_call.png>`.
  Original illustration extracted unchanged from the thesis. Its editable figure source was not recovered; the current library and existing user-guide diagrams document the maintained architecture.

.. _ma-caption-0266:

* **Table  1  Comparison of the AAS-derived proof-of-fit dataset and the semi-synthetic Weg-A reference dataset used for EEG-fMRI artifact-correction experiments.**
  Datasets: ``proof_fit``, ``weg_a_v7``.
  :doc:`/thesis_reference/datasets_evaluation`.

.. _ma-caption-0271:

* **Table  2   Methodological adaptations introduced in Phase 2 that could materially influence artifact-correction performance.**
  Experiments: ``holdout_aas_baseline``, ``holdout_aas_naive_6nn``, ``holdout_cascaded_context_dae``, ``holdout_cascaded_dae``, ``holdout_conv_tasnet``, ``holdout_d4pm``, ``holdout_demucs``, ``holdout_denoise_mamba``, ``holdout_dhct_gan``, ``holdout_dhct_gan_v2``, ``holdout_dpae``, ``holdout_ic_unet``, ``holdout_nested_gan``, ``holdout_sepformer``, ``holdout_st_gnn``, ``holdout_vit_spectrogram``, ``deployment_cascaded_context_dae``, ``deployment_cascaded_dae``, ``deployment_conv_tasnet``, ``deployment_demucs``, ``deployment_denoise_mamba``, ``deployment_dhct_gan``, ``deployment_dhct_gan_v2``, ``deployment_dpae``, ``deployment_ic_unet``, ``deployment_nested_gan``, ``deployment_sepformer``, ``deployment_st_gnn``, ``deployment_vit_spectrogram``, ``deployment_d4pm``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/datasets_evaluation`.

.. _ma-caption-0279:

* **Figure  6  Overview of the DPAE**
  Experiments: ``deployment_dpae``.
  :doc:`/thesis_reference/models/dpae`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0280:

* **Table  3   Training configuration of the evaluated DPAE deployment variant.**
  Experiments: ``deployment_dpae``.
  :doc:`/thesis_reference/models/dpae`.

.. _ma-caption-0288:

* **Figure  7  Overview of the IC U-Net**
  Experiments: ``deployment_ic_unet``.
  :doc:`/thesis_reference/models/ic_unet`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0289:

* **Table  4   Training configuration of the evaluated multichannel IC-U-Net deployment variant.**
  Experiments: ``deployment_ic_unet``.
  :doc:`/thesis_reference/models/ic_unet`.

.. _ma-caption-0296:

* **Figure  8  Overview of Cascaded DAE**
  Experiments: ``deployment_cascaded_context_dae``.
  :doc:`/thesis_reference/models/cascaded_dae`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.
  The seven-epoch Cascaded DAE in this caption maps to the context variant; the single-epoch Cascaded DAE remains a distinct evaluated arm.

.. _ma-caption-0297:

* **Table  5   Training configuration of the evaluated Cascaded DAE with seven-epoch temporal context.**
  Experiments: ``deployment_cascaded_context_dae``.
  :doc:`/thesis_reference/models/cascaded_dae`.
  The seven-epoch Cascaded DAE in this caption maps to the context variant; the single-epoch Cascaded DAE remains a distinct evaluated arm.

.. _ma-caption-0306:

* **Figure  9  Overview of Nested GAN**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``.
  :doc:`/thesis_reference/models/nested_gan`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0307:

* **Table  6   Training configuration of the selected Nested GAN configuration used for the Phase-3 evaluation.**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``.
  :doc:`/thesis_reference/models/nested_gan`.

.. _ma-caption-0314:

* **Figure  10  Overview of DHCT - GAN**
  Experiments: ``run8_dhct_gan_lr0_0001_bc8_sisdr0_s42``.
  :doc:`/thesis_reference/models/dhct_gan`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0315:

* **Table  7   Training configuration of the evaluated DHCT-GAN with seven-epoch temporal context.**
  Experiments: ``run8_dhct_gan_lr0_0001_bc8_sisdr0_s42``.
  :doc:`/thesis_reference/models/dhct_gan`.

.. _ma-caption-0327:

* **Figure  11  Overview of the D4PM**
  Experiments: ``deployment_d4pm``.
  :doc:`/thesis_reference/models/d4pm`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0328:

* **Table  8   Training configuration of the evaluated single-branch D4PM deployment variant.**
  Experiments: ``deployment_d4pm``.
  :doc:`/thesis_reference/models/d4pm`.

.. _ma-caption-0338:

* **Figure  12  Overview of Denoise Mamba**
  Experiments: ``deployment_denoise_mamba``.
  :doc:`/thesis_reference/models/denoise_mamba`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0339:

* **Table  9   Training configuration of the evaluated  DenoiseMamba  deployment variant.**
  Experiments: ``deployment_denoise_mamba``.
  :doc:`/thesis_reference/models/denoise_mamba`.

.. _ma-caption-0350:

* **Figure  13  Overview of Conv- TasNet**
  Experiments: ``deployment_conv_tasnet``.
  :doc:`/thesis_reference/models/conv_tasnet`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0351:

* **Table  10   Training configuration of the evaluated Conv- TasNet  deployment variant.**
  Experiments: ``deployment_conv_tasnet``.
  :doc:`/thesis_reference/models/conv_tasnet`.

.. _ma-caption-0359:

* **Figure  14  Overview of Demucs architecture**
  Experiments: ``run8_demucs_lr0_001_ic64_sisdr0_s42``.
  :doc:`/thesis_reference/models/demucs`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0360:

* **Table  11   Training configuration of the selected Demucs configuration used for the Phase-3 evaluation.**
  Experiments: ``run8_demucs_lr0_001_ic64_sisdr0_s42``.
  :doc:`/thesis_reference/models/demucs`.

.. _ma-caption-0366:

* **Figure  15  Overview of  SepFormer**
  Experiments: ``deployment_sepformer``.
  :doc:`/thesis_reference/models/sepformer`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0367:

* **Table  12   Training configuration of the evaluated  SepFormer  deployment variant.**
  Experiments: ``deployment_sepformer``.
  :doc:`/thesis_reference/models/sepformer`.

.. _ma-caption-0377:

* **Figure  16  Overview of  ViT  spectrogram**
  Experiments: ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``.
  :doc:`/thesis_reference/models/vit_spectrogram`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0378:

* **Table  13   Training configuration of the selected Vision Transformer configuration used for the Phase-3 evaluation.**
  Experiments: ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``.
  :doc:`/thesis_reference/models/vit_spectrogram`.

.. _ma-caption-0386:

* **Figure  17  Overview of ST-GNN**
  Experiments: ``deployment_st_gnn``.
  :doc:`/thesis_reference/models/st_gnn`.
  Generator: :download:`source <../../../tools/diagrams/build_selected_variants.py>`.
  Command: ``uv run python tools/diagrams/build_selected_variants.py``.

.. _ma-caption-0387:

* **Table  14   Training configuration of the evaluated multichannel ST-GNN deployment variant.**
  Experiments: ``deployment_st_gnn``.
  :doc:`/thesis_reference/models/st_gnn`.

.. _ma-caption-0395:

* **Table   2   Hyperparameter selection and controlled search strategy**
  Experiments: ``run8_nested_gan_lr0_00015_ch32_sisdr0_s42``, ``run8_nested_gan_lr0_00015_ch32_sisdr1_s42``, ``run8_nested_gan_lr0_00015_ch32_sisdr3_s42``, ``run8_nested_gan_lr0_00015_ch48_sisdr0_s42``, ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_nested_gan_lr0_00015_ch48_sisdr3_s42``, ``run8_nested_gan_lr0_00015_ch72_sisdr0_s42``, ``run8_nested_gan_lr0_00015_ch72_sisdr1_s42``, ``run8_nested_gan_lr0_00015_ch72_sisdr3_s42``, ``run8_nested_gan_lr0_0005_ch32_sisdr0_s42``, ``run8_nested_gan_lr0_0005_ch32_sisdr1_s42``, ``run8_nested_gan_lr0_0005_ch32_sisdr3_s42``, ``run8_nested_gan_lr0_0005_ch48_sisdr0_s42``, ``run8_nested_gan_lr0_0005_ch48_sisdr1_s42``, ``run8_nested_gan_lr0_0005_ch48_sisdr3_s42``, ``run8_nested_gan_lr0_0005_ch72_sisdr0_s42``, ``run8_nested_gan_lr0_0005_ch72_sisdr1_s42``, ``run8_nested_gan_lr0_0005_ch72_sisdr3_s42``, ``run8_nested_gan_lr0_0015_ch32_sisdr0_s42``, ``run8_nested_gan_lr0_0015_ch32_sisdr1_s42``, ``run8_nested_gan_lr0_0015_ch32_sisdr3_s42``, ``run8_nested_gan_lr0_0015_ch48_sisdr0_s42``, ``run8_nested_gan_lr0_0015_ch48_sisdr1_s42``, ``run8_nested_gan_lr0_0015_ch48_sisdr3_s42``, ``run8_nested_gan_lr0_0015_ch72_sisdr0_s42``, ``run8_nested_gan_lr0_0015_ch72_sisdr1_s42``, ``run8_nested_gan_lr0_0015_ch72_sisdr3_s42``, ``run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42``, ``run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42``, ``run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42``, ``run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42``, ``run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42``, ``run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42``, ``run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42``, ``run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42``, ``run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42``, ``run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42``, ``run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42``, ``run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42``, ``run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42``, ``run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42``, ``run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42``, ``run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42``, ``run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42``, ``run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42``, ``run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42``, ``run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42``, ``run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42``, ``run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42``, ``run8_dhct_gan_lr0_0001_bc8_sisdr0_s42``, ``run8_dhct_gan_lr0_0001_bc8_sisdr1_s42``, ``run8_demucs_lr0_0001_ic32_sisdr0_s42``, ``run8_demucs_lr0_0001_ic32_sisdr1_s42``, ``run8_demucs_lr0_0001_ic32_sisdr3_s42``, ``run8_demucs_lr0_0001_ic64_sisdr0_s42``, ``run8_demucs_lr0_0001_ic64_sisdr1_s42``, ``run8_demucs_lr0_0001_ic64_sisdr3_s42``, ``run8_demucs_lr0_0001_ic96_sisdr0_s42``, ``run8_demucs_lr0_0001_ic96_sisdr1_s42``, ``run8_demucs_lr0_0001_ic96_sisdr3_s42``, ``run8_demucs_lr0_0003_ic32_sisdr0_s42``, ``run8_demucs_lr0_0003_ic32_sisdr1_s42``, ``run8_demucs_lr0_0003_ic32_sisdr3_s42``, ``run8_demucs_lr0_0003_ic64_sisdr0_s42``, ``run8_demucs_lr0_0003_ic64_sisdr1_s42``, ``run8_demucs_lr0_0003_ic64_sisdr3_s42``, ``run8_demucs_lr0_0003_ic96_sisdr0_s42``, ``run8_demucs_lr0_0003_ic96_sisdr1_s42``, ``run8_demucs_lr0_0003_ic96_sisdr3_s42``, ``run8_demucs_lr0_001_ic32_sisdr0_s42``, ``run8_demucs_lr0_001_ic32_sisdr1_s42``, ``run8_demucs_lr0_001_ic32_sisdr3_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_demucs_lr0_001_ic64_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr3_s42``, ``run8_demucs_lr0_001_ic96_sisdr0_s42``, ``run8_demucs_lr0_001_ic96_sisdr1_s42``, ``run8_demucs_lr0_001_ic96_sisdr3_s42``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_3_grid_search`.

.. _ma-caption-0403:

* **Figure  18  Training configuration and run artifacts of the  Niazy  proof fit run of Phase 2**
  Experiments: ``training_run7_dpae``, ``training_run7_ic_u_net``, ``training_run7_cascaded_dae``, ``training_run7_nested_gan``, ``training_run7_dhct_gan``, ``training_run7_dhct_gan_v2``, ``training_run7_d4pm``, ``training_run7_denoisemamba``, ``training_run7_conv_tasnet``, ``training_run7_demucs``, ``training_run7_sepformer``, ``training_run7_vision_transformer``, ``training_run7_st_gnn``, ``training_run7_cascaded_context_dae``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_2_pipeline_deployment`.
  Generator: :download:`source <../../../tools/diagrams/build_facetpy_proof_fit_dataset.py>`.
  Command: ``uv run python tools/diagrams/build_facetpy_proof_fit_dataset.py``.

.. _ma-caption-0407:

* **Figure  19  Configuration-driven FACETpy deep-learning framework from training to pipeline inference**
  :doc:`/thesis_reference/code_provenance`.
  Generator: :download:`source <../../../tools/diagrams/build_facetpy_dl_framework.py>`.
  Command: ``uv run python tools/diagrams/build_facetpy_dl_framework.py``.

.. _ma-caption-0412:

* **Table   3  Components and weights of the recovered-clean and spike-aware training objective.**
  Experiments: ``spike_aware_demucs``, ``spike_aware_nested_gan``, ``spike_aware_vit_spectrogram``.
  Datasets: ``weg_a_v10_locked_1ch``.
  :doc:`/thesis_reference/phase_3_grid_search`.

.. _ma-caption-0420:

* **Figure  20   Integration of a checkpoint-backed deep-learning model into the FACETpy correction pipeline, including validation gates enforced by the  DeepLearningCorrection  processor.**
  Experiments: ``deployment_cascaded_context_dae``, ``deployment_cascaded_dae``, ``deployment_conv_tasnet``, ``deployment_demucs``, ``deployment_denoise_mamba``, ``deployment_dhct_gan``, ``deployment_dhct_gan_v2``, ``deployment_dpae``, ``deployment_ic_unet``, ``deployment_nested_gan``, ``deployment_sepformer``, ``deployment_st_gnn``, ``deployment_vit_spectrogram``, ``deployment_d4pm``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_2_pipeline_deployment`.
  Generator: :download:`source <../../../tools/diagrams/build_facetpy_training_to_deployment.py>`.
  Command: ``uv run python tools/diagrams/build_facetpy_training_to_deployment.py``.

.. _ma-caption-0440:

* **Table  15   Reference-based metrics for artifact reduction and recovered-clean signal reconstruction.**
  Experiments: ``holdout_aas_baseline``, ``holdout_aas_naive_6nn``, ``holdout_cascaded_context_dae``, ``holdout_cascaded_dae``, ``holdout_conv_tasnet``, ``holdout_d4pm``, ``holdout_demucs``, ``holdout_denoise_mamba``, ``holdout_dhct_gan``, ``holdout_dhct_gan_v2``, ``holdout_dpae``, ``holdout_ic_unet``, ``holdout_nested_gan``, ``holdout_sepformer``, ``holdout_st_gnn``, ``holdout_vit_spectrogram``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_1_unified_holdout`.

.. _ma-caption-0444:

* **Table  16    Spike-preservation metrics for evaluating waveform fidelity, detectability, amplitude, and timing of injected interictal epileptiform discharges.**
  Experiments: ``spike_aware_demucs``, ``spike_aware_nested_gan``, ``spike_aware_vit_spectrogram``.
  Results: ``spike_holdout_summary``, ``spike_pipeline_summary``.
  Datasets: ``weg_a_v10_locked_1ch``.
  :doc:`/thesis_reference/phase_3_grid_search`.

.. _ma-caption-0450:

* **Table  17   Engineering indicators before and after the FACETpy refactoring.**
  Experiments: ``refactoring_engineering_indicators``.
  :doc:`/thesis_reference/phase_0_legacy`.

.. _ma-caption-0458:

* **Table  18   Behavioral parity and computational cost of the refactoring.**
  Experiments: ``refactoring_benchmark_legacy_vs_v2``.
  :doc:`/thesis_reference/phase_0_legacy`.

.. _ma-caption-0466:

* **Figure  21   Historical Phase 0 evaluation metrics for the legacy AAS reference and the legacy fully connected DAE; each panel retains its original metric scale and direction.**
  Experiments: ``legacy_delivery``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_0_legacy`.
  Generator: :download:`source <../../../tools/plotting/plot_phase0_signal_comparison.py>`.
  Command: ``uv run python tools/plotting/plot_phase0_signal_comparison.py --data-root <data-root> --out-dir <output>``.

.. _ma-caption-0478:

* **Figure  22   Three-second Fp1 signal segment after correction with legacy AAS and the legacy fully connected DAE; both traces use the same amplitude scale.**
  Experiments: ``legacy_delivery``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_0_legacy`.
  Generator: :download:`source <../../../tools/plotting/plot_phase0_signal_comparison.py>`.
  Command: ``uv run python tools/plotting/plot_phase0_signal_comparison.py --data-root <data-root> --out-dir <output>``.

.. _ma-caption-0481:

* **Figure  23   Three-second Fp1 comparison of gradient-artifact estimates derived as the difference between the raw signal and the signal corrected by legacy AAS or the legacy fully connected DAE.**
  Experiments: ``legacy_fc_dae``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_0_legacy`.
  Generator: :download:`source <../../../tools/plotting/plot_phase0_signal_comparison.py>`.
  Command: ``uv run python tools/plotting/plot_phase0_signal_comparison.py --data-root <data-root> --out-dir <output>``.
  The original figure generator reads legacy_native.npz, whereas Figures 21 and 22 use the separate delivery run. Their checkpoint identities must not be merged.

.. _ma-caption-0487:

* **Figure  24   Clean-signal SNR improvement on the unified holdout for all evaluated model variants; higher values indicate closer reconstruction of the AAS-derived reference signal.**
  Experiments: ``holdout_aas_baseline``, ``holdout_aas_naive_6nn``, ``holdout_cascaded_context_dae``, ``holdout_cascaded_dae``, ``holdout_conv_tasnet``, ``holdout_d4pm``, ``holdout_demucs``, ``holdout_denoise_mamba``, ``holdout_dhct_gan``, ``holdout_dhct_gan_v2``, ``holdout_dpae``, ``holdout_ic_unet``, ``holdout_nested_gan``, ``holdout_sepformer``, ``holdout_st_gnn``, ``holdout_vit_spectrogram``.
  Results: ``table_phase1_unified_holdout_ranking``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_1_unified_holdout`.
  Generator: :download:`source <../../../tools/plotting/thesis_figures.py>`.
  Command: ``uv run python tools/plotting/thesis_figures.py ranking --out <figure.png>``.

.. _ma-caption-0491:

* **Figure  25   Corrected Fp1 signals from a common unified-holdout epoch for all evaluated model variants. Each panel shows the noisy input, the model-corrected signal, and the AAS-derived target; SNR gain is reported for the complete unified holdout.**
  Experiments: ``holdout_aas_baseline``, ``holdout_aas_naive_6nn``, ``holdout_cascaded_context_dae``, ``holdout_cascaded_dae``, ``holdout_conv_tasnet``, ``holdout_d4pm``, ``holdout_demucs``, ``holdout_denoise_mamba``, ``holdout_dhct_gan``, ``holdout_dhct_gan_v2``, ``holdout_dpae``, ``holdout_ic_unet``, ``holdout_nested_gan``, ``holdout_sepformer``, ``holdout_st_gnn``, ``holdout_vit_spectrogram``.
  Results: ``table_phase1_unified_holdout_ranking``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_1_unified_holdout`.
  Generator: :download:`source <../../../tools/plotting/plot_phase1_holdout_signal_comparison.py>`.
  Command: ``uv run python tools/plotting/plot_phase1_holdout_signal_comparison.py --data-root <data-root> --out-dir <output>``.

.. _ma-caption-0494:

* **Figure  26   Recorded training and validation loss histories for the neural model families evaluated in Phase 1. AAS baselines are not shown because they were not trained as neural models.**
  Experiments: ``holdout_aas_baseline``, ``holdout_aas_naive_6nn``, ``holdout_cascaded_context_dae``, ``holdout_cascaded_dae``, ``holdout_conv_tasnet``, ``holdout_d4pm``, ``holdout_demucs``, ``holdout_denoise_mamba``, ``holdout_dhct_gan``, ``holdout_dhct_gan_v2``, ``holdout_dpae``, ``holdout_ic_unet``, ``holdout_nested_gan``, ``holdout_sepformer``, ``holdout_st_gnn``, ``holdout_vit_spectrogram``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_1_unified_holdout`.
  Generator: :download:`source <../../../tools/plotting/training_histories.py>`.
  Command: ``uv run python tools/plotting/training_histories.py --phase 1 --out <figure.png>``.

.. _ma-caption-0501:

* **Figure  27   SNR gain of the selected Phase-2 deployment models on the unified AAS-derived holdout.**
  Experiments: ``deployment_cascaded_context_dae``, ``deployment_cascaded_dae``, ``deployment_conv_tasnet``, ``deployment_demucs``, ``deployment_denoise_mamba``, ``deployment_dhct_gan``, ``deployment_dhct_gan_v2``, ``deployment_dpae``, ``deployment_ic_unet``, ``deployment_nested_gan``, ``deployment_sepformer``, ``deployment_st_gnn``, ``deployment_vit_spectrogram``, ``deployment_d4pm``.
  Results: ``phase2_holdout_replay``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_2_pipeline_deployment`.
  :download:`Original thesis figure <../../../docs/source/_static/thesis/figure27_original_holdout.png>`.
  Generator: :download:`source <../../../masterthesis_guide/examples/evaluate_model.py>`.
  Command: ``uv run python -m masterthesis_guide.examples.evaluate_model deployment_demucs --data-root <data-root> --out <metrics.json>``.
  The original Figure-27 CSV was not found. The thesis image is retained; the linked table is a newly computed 166-window CPU replay and is not claimed to be the original table.

.. _ma-caption-0517:

* **Figure  28   Phase-2 model corrections over one trigger-aligned gradient-artifact window; grey: input signal, blue: model-corrected output.**
  Experiments: ``deployment_cascaded_context_dae``, ``deployment_cascaded_dae``, ``deployment_conv_tasnet``, ``deployment_demucs``, ``deployment_denoise_mamba``, ``deployment_dhct_gan``, ``deployment_dhct_gan_v2``, ``deployment_dpae``, ``deployment_ic_unet``, ``deployment_nested_gan``, ``deployment_sepformer``, ``deployment_st_gnn``, ``deployment_vit_spectrogram``, ``deployment_d4pm``.
  Results: ``table_phase2_selected_variant_results``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_2_pipeline_deployment`.
  Generator: :download:`source <../../../tools/plotting/plot_phase2_family_artifact_window_fast.py>`.
  Command: ``uv run python tools/plotting/plot_phase2_family_artifact_window_fast.py --edf <NiazyFMRI.edf> --out-dir <output>``.

.. _ma-caption-0520:

* **Figure  29   Training and validation loss curves for the selected Phase-2 deployment configurations.**
  Experiments: ``training_run7_dpae``, ``training_run7_ic_u_net``, ``training_run7_cascaded_dae``, ``training_run7_nested_gan``, ``training_run7_dhct_gan``, ``training_run7_dhct_gan_v2``, ``training_run7_d4pm``, ``training_run7_denoisemamba``, ``training_run7_conv_tasnet``, ``training_run7_demucs``, ``training_run7_sepformer``, ``training_run7_vision_transformer``, ``training_run7_st_gnn``, ``training_run7_cascaded_context_dae``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_2_pipeline_deployment`.
  Generator: :download:`source <../../../tools/plotting/training_histories.py>`.
  Command: ``uv run python tools/plotting/training_histories.py --phase 2 --out <figure.png>``.

.. _ma-caption-0526:

* **Figure  30   Plots of the five best models and their corrected signal from the phase 2 evaluation.**
  Experiments: ``deployment_nested_gan``, ``deployment_vit_spectrogram``, ``deployment_sepformer``, ``deployment_dhct_gan``, ``deployment_demucs``.
  Results: ``table_phase2_pipeline_results``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_2_pipeline_deployment`.
  Generator: :download:`source <../../../tools/plotting/plot_phase2_pipeline_signal_comparison.py>`.
  Command: ``uv run python tools/plotting/plot_phase2_pipeline_signal_comparison.py --arm-dir <primary-arms> --out-dir <output> --top 5``.
  Figure 30 uses the five best valid primary Phase-2 arms, including single-epoch DHCT-GAN. It is separate from the later seven-context DHCT selection.

.. _ma-caption-0543:

* **Table  19   Initial Hyperparameters for the  three-grid  search optimized models**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``.
  Results: ``table_phase3_before_after``.
  Datasets: ``proof_fit``.
  :doc:`/thesis_reference/phase_3_grid_search`.

.. _ma-caption-0546:

* **Figure  31   Phase-3 grid-search landscapes across learning rate, model capacity, and SI-SDR weight. Each cell shows residual gradient  artifact. The  green indicates lower residual artifact and outlined cells mark the best valid configuration**
  Experiments: ``run8_nested_gan_lr0_00015_ch32_sisdr0_s42``, ``run8_nested_gan_lr0_00015_ch32_sisdr1_s42``, ``run8_nested_gan_lr0_00015_ch32_sisdr3_s42``, ``run8_nested_gan_lr0_00015_ch48_sisdr0_s42``, ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_nested_gan_lr0_00015_ch48_sisdr3_s42``, ``run8_nested_gan_lr0_00015_ch72_sisdr0_s42``, ``run8_nested_gan_lr0_00015_ch72_sisdr1_s42``, ``run8_nested_gan_lr0_00015_ch72_sisdr3_s42``, ``run8_nested_gan_lr0_0005_ch32_sisdr0_s42``, ``run8_nested_gan_lr0_0005_ch32_sisdr1_s42``, ``run8_nested_gan_lr0_0005_ch32_sisdr3_s42``, ``run8_nested_gan_lr0_0005_ch48_sisdr0_s42``, ``run8_nested_gan_lr0_0005_ch48_sisdr1_s42``, ``run8_nested_gan_lr0_0005_ch48_sisdr3_s42``, ``run8_nested_gan_lr0_0005_ch72_sisdr0_s42``, ``run8_nested_gan_lr0_0005_ch72_sisdr1_s42``, ``run8_nested_gan_lr0_0005_ch72_sisdr3_s42``, ``run8_nested_gan_lr0_0015_ch32_sisdr0_s42``, ``run8_nested_gan_lr0_0015_ch32_sisdr1_s42``, ``run8_nested_gan_lr0_0015_ch32_sisdr3_s42``, ``run8_nested_gan_lr0_0015_ch48_sisdr0_s42``, ``run8_nested_gan_lr0_0015_ch48_sisdr1_s42``, ``run8_nested_gan_lr0_0015_ch48_sisdr3_s42``, ``run8_nested_gan_lr0_0015_ch72_sisdr0_s42``, ``run8_nested_gan_lr0_0015_ch72_sisdr1_s42``, ``run8_nested_gan_lr0_0015_ch72_sisdr3_s42``, ``run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42``, ``run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42``, ``run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42``, ``run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42``, ``run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42``, ``run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42``, ``run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42``, ``run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42``, ``run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42``, ``run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42``, ``run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42``, ``run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42``, ``run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42``, ``run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42``, ``run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42``, ``run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42``, ``run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42``, ``run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42``, ``run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42``, ``run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42``, ``run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42``, ``run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42``, ``run8_dhct_gan_lr0_0001_bc8_sisdr0_s42``, ``run8_dhct_gan_lr0_0001_bc8_sisdr1_s42``, ``run8_demucs_lr0_0001_ic32_sisdr0_s42``, ``run8_demucs_lr0_0001_ic32_sisdr1_s42``, ``run8_demucs_lr0_0001_ic32_sisdr3_s42``, ``run8_demucs_lr0_0001_ic64_sisdr0_s42``, ``run8_demucs_lr0_0001_ic64_sisdr1_s42``, ``run8_demucs_lr0_0001_ic64_sisdr3_s42``, ``run8_demucs_lr0_0001_ic96_sisdr0_s42``, ``run8_demucs_lr0_0001_ic96_sisdr1_s42``, ``run8_demucs_lr0_0001_ic96_sisdr3_s42``, ``run8_demucs_lr0_0003_ic32_sisdr0_s42``, ``run8_demucs_lr0_0003_ic32_sisdr1_s42``, ``run8_demucs_lr0_0003_ic32_sisdr3_s42``, ``run8_demucs_lr0_0003_ic64_sisdr0_s42``, ``run8_demucs_lr0_0003_ic64_sisdr1_s42``, ``run8_demucs_lr0_0003_ic64_sisdr3_s42``, ``run8_demucs_lr0_0003_ic96_sisdr0_s42``, ``run8_demucs_lr0_0003_ic96_sisdr1_s42``, ``run8_demucs_lr0_0003_ic96_sisdr3_s42``, ``run8_demucs_lr0_001_ic32_sisdr0_s42``, ``run8_demucs_lr0_001_ic32_sisdr1_s42``, ``run8_demucs_lr0_001_ic32_sisdr3_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_demucs_lr0_001_ic64_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr3_s42``, ``run8_demucs_lr0_001_ic96_sisdr0_s42``, ``run8_demucs_lr0_001_ic96_sisdr1_s42``, ``run8_demucs_lr0_001_ic96_sisdr3_s42``.
  Results: ``table_phase3_grid_runs``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_3_grid_search`.
  Generator: :download:`source <../../../tools/plotting/thesis_figures.py>`.
  Command: ``uv run python tools/plotting/thesis_figures.py grid --out <figure.png>``.

.. _ma-caption-0547:

* **Table  20   Winning configuration from the grid-search**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``.
  Results: ``table_phase3_before_after``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_3_grid_search`.

.. _ma-caption-0550:

* **Figure  32   Phase 3 Before and after the grid-search**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``.
  Results: ``table_phase3_before_after``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_3_grid_search`.
  Generator: :download:`source <../../../tools/plotting/thesis_figures.py>`.
  Command: ``uv run python tools/plotting/thesis_figures.py before_after --out <figure.png>``.

.. _ma-caption-0565:

* **Figure  33  Each phase 3 model compared spike preservation with  FARM and reference**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``, ``spike_aware_demucs``, ``spike_aware_nested_gan``, ``spike_aware_vit_spectrogram``.
  Results: ``spike_pipeline_summary``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_3_grid_search`.
  Generator: :download:`source <../../../tools/plotting/spike_waveforms.py>`.
  Command: ``uv run python tools/plotting/spike_waveforms.py --data-root <data-root> --out <figure.png>``.

.. _ma-caption-0567:

* **Table  21  Spike preservation measurements for different models compared to FARM and the non - corrected variant**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``, ``spike_aware_demucs``, ``spike_aware_nested_gan``, ``spike_aware_vit_spectrogram``.
  Results: ``spike_pipeline_summary``, ``spike_reference_residuals``, ``spike_reference_preservation``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_3_grid_search`.

.. _ma-caption-0573:

* **Figure  34   Frequency spectrum  comparison of the phase 3 models and FARM over the full Niazy-dataset**
  Experiments: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``run8_demucs_lr0_001_ic64_sisdr0_s42``, ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``.
  Datasets: ``niazy_recording``.
  :doc:`/thesis_reference/phase_3_grid_search`.
  :download:`Original thesis figure <../../../docs/source/_static/thesis/figure34_original_spectrum.png>`.
  The caption calls this the full dataset, but the image itself identifies Fp1 and 29.5–160 s. Preserve that narrower observed scope. The original Welch settings/generator were not recovered; the retained image is the original evidence.

.. _ma-wega-grid-result:

* **Independent-reference Weg-A grid and negative result**
  Experiments: ``wega_demucs_lr0_0001_ic32_sisdr0_s42``, ``wega_demucs_lr0_0001_ic32_sisdr1_s42``, ``wega_demucs_lr0_0001_ic32_sisdr3_s42``, ``wega_demucs_lr0_0001_ic64_sisdr0_s42``, ``wega_demucs_lr0_0001_ic64_sisdr1_s42``, ``wega_demucs_lr0_0001_ic64_sisdr3_s42``, ``wega_demucs_lr0_0001_ic96_sisdr0_s42``, ``wega_demucs_lr0_0001_ic96_sisdr1_s42``, ``wega_demucs_lr0_0001_ic96_sisdr3_s42``, ``wega_demucs_lr0_0003_ic32_sisdr0_s42``, ``wega_demucs_lr0_0003_ic32_sisdr1_s42``, ``wega_demucs_lr0_0003_ic32_sisdr3_s42``, ``wega_demucs_lr0_0003_ic64_sisdr0_s42``, ``wega_demucs_lr0_0003_ic64_sisdr1_s42``, ``wega_demucs_lr0_0003_ic64_sisdr3_s42``, ``wega_demucs_lr0_0003_ic96_sisdr0_s42``, ``wega_demucs_lr0_0003_ic96_sisdr1_s42``, ``wega_demucs_lr0_0003_ic96_sisdr3_s42``, ``wega_demucs_lr0_001_ic32_sisdr0_s42``, ``wega_demucs_lr0_001_ic32_sisdr1_s42``, ``wega_demucs_lr0_001_ic32_sisdr3_s42``, ``wega_demucs_lr0_001_ic64_sisdr0_s42``, ``wega_demucs_lr0_001_ic64_sisdr1_s42``, ``wega_demucs_lr0_001_ic64_sisdr3_s42``, ``wega_demucs_lr0_001_ic96_sisdr0_s42``, ``wega_demucs_lr0_001_ic96_sisdr1_s42``, ``wega_demucs_lr0_001_ic96_sisdr3_s42``, ``wega_dhct_gan_lr0_0001_bc8_sisdr0_s42``, ``wega_dhct_gan_lr0_0001_bc8_sisdr1_s42``, ``wega_nested_gan_lr0_00015_ch32_sisdr0_s42``, ``wega_nested_gan_lr0_00015_ch32_sisdr1_s42``, ``wega_nested_gan_lr0_00015_ch32_sisdr3_s42``, ``wega_nested_gan_lr0_00015_ch48_sisdr0_s42``, ``wega_nested_gan_lr0_00015_ch48_sisdr1_s42``, ``wega_nested_gan_lr0_00015_ch48_sisdr3_s42``, ``wega_nested_gan_lr0_00015_ch72_sisdr0_s42``, ``wega_nested_gan_lr0_00015_ch72_sisdr1_s42``, ``wega_nested_gan_lr0_00015_ch72_sisdr3_s42``, ``wega_nested_gan_lr0_0005_ch32_sisdr0_s42``, ``wega_nested_gan_lr0_0005_ch32_sisdr1_s42``, ``wega_nested_gan_lr0_0005_ch32_sisdr3_s42``, ``wega_nested_gan_lr0_0005_ch48_sisdr0_s42``, ``wega_nested_gan_lr0_0005_ch48_sisdr1_s42``, ``wega_nested_gan_lr0_0005_ch48_sisdr3_s42``, ``wega_nested_gan_lr0_0005_ch72_sisdr0_s42``, ``wega_nested_gan_lr0_0005_ch72_sisdr1_s42``, ``wega_nested_gan_lr0_0005_ch72_sisdr3_s42``, ``wega_nested_gan_lr0_0015_ch32_sisdr0_s42``, ``wega_nested_gan_lr0_0015_ch32_sisdr1_s42``, ``wega_nested_gan_lr0_0015_ch32_sisdr3_s42``, ``wega_nested_gan_lr0_0015_ch48_sisdr0_s42``, ``wega_nested_gan_lr0_0015_ch48_sisdr1_s42``, ``wega_nested_gan_lr0_0015_ch48_sisdr3_s42``, ``wega_nested_gan_lr0_0015_ch72_sisdr0_s42``, ``wega_nested_gan_lr0_0015_ch72_sisdr1_s42``, ``wega_nested_gan_lr0_0015_ch72_sisdr3_s42``, ``wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42``, ``wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42``, ``wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42``, ``wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42``, ``wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42``, ``wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42``, ``wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42``, ``wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42``, ``wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42``, ``wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42``, ``wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42``, ``wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42``, ``wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42``, ``wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42``, ``wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42``, ``wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42``, ``wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42``, ``wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42``, ``wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42``, ``wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42``, ``wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42``, ``wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42``, ``wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42``, ``wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42``, ``wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42``, ``wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42``, ``wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42``.
  Results: ``table_phase3_wega_grid_runs``.
  Datasets: ``weg_a_v7``.
  :doc:`/thesis_reference/phase_3_grid_search`.
  Generator: :download:`source <../../../tools/plotting/thesis_figures.py>`.
  Command: ``uv run python tools/plotting/thesis_figures.py wega_grid --out <figure.png>``.
  The unnumbered grid and following prose use selection-split reconstruction error, not pipeline residual artifact. The prose comparison with proof-fit therefore does not establish a matched numerical improvement or deterioration under one metric.

Datasets and saved splits
-------------------------

* ``proof_fit``: external ``output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz``.
  :download:`metadata <../../../masterthesis_guide/datasets/proof_fit/metadata.json>`.
  :download:`split <../../../masterthesis_guide/datasets/proof_fit/splits/holdout_v1_indices.json>`.
* ``weg_a_v7``: external ``output/weg_a_farm_v7_k6_512/weg_a_spatiotemporal_dataset.npz``.
  :download:`metadata <../../../masterthesis_guide/datasets/weg_a_v7/metadata.json>`.
* ``weg_a_v10_locked``: external ``output/weg_a_farm_v10_locked_512/weg_a_spatiotemporal_dataset.npz``.
  :download:`metadata <../../../masterthesis_guide/datasets/weg_a_v10_locked/metadata.json>`.
* ``weg_a_v10_locked_1ch``: external ``output/weg_a_farm_v10_locked_1ch/weg_a_spatiotemporal_dataset.npz``.
  :download:`metadata <../../../masterthesis_guide/datasets/weg_a_v10_locked_1ch/metadata.json>`.
* ``artifact_farm_pca4``: external ``output/artifact_libraries/niazy_farm_pca4_direct/niazy_aas_pca4_artifact.npz``.
  :download:`metadata <../../../masterthesis_guide/datasets/artifact_farm_pca4/metadata.json>`.
* ``artifact_aas_direct``: external ``output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz``.
  :download:`metadata <../../../masterthesis_guide/datasets/artifact_aas_direct/metadata.json>`.
* ``niazy_recording``: external ``examples/datasets/NiazyFMRI.edf``.

Original Phase-1 predictions
----------------------------

These arrays are retained in Git LFS and verified against the catalog before plotting.

* ``holdout_cascaded_context_dae``: ``artifacts/predictions/phase_1/cascaded_context_dae/predicted_artifact.npy``.
* ``holdout_cascaded_dae``: ``artifacts/predictions/phase_1/cascaded_dae/predicted_artifact.npy``.
* ``holdout_conv_tasnet``: ``artifacts/predictions/phase_1/conv_tasnet/predicted_artifact.npy``.
* ``holdout_d4pm``: ``artifacts/predictions/phase_1/d4pm/predicted_artifact.npy``.
* ``holdout_demucs``: ``artifacts/predictions/phase_1/demucs/predicted_artifact.npy``.
* ``holdout_denoise_mamba``: ``artifacts/predictions/phase_1/denoise_mamba/predicted_artifact.npy``.
* ``holdout_dhct_gan``: ``artifacts/predictions/phase_1/dhct_gan/predicted_artifact.npy``.
* ``holdout_dhct_gan_v2``: ``artifacts/predictions/phase_1/dhct_gan_v2/predicted_artifact.npy``.
* ``holdout_dpae``: ``artifacts/predictions/phase_1/dpae/predicted_artifact.npy``.
* ``holdout_ic_unet``: ``artifacts/predictions/phase_1/ic_unet/predicted_artifact.npy``.
* ``holdout_nested_gan``: ``artifacts/predictions/phase_1/nested_gan/predicted_artifact.npy``.
* ``holdout_sepformer``: ``artifacts/predictions/phase_1/sepformer/predicted_artifact.npy``.
* ``holdout_st_gnn``: ``artifacts/predictions/phase_1/st_gnn/predicted_artifact.npy``.
* ``holdout_vit_spectrogram``: ``artifacts/predictions/phase_1/vit_spectrogram/predicted_artifact.npy``.

Recorded comparison tables
--------------------------

* :download:`run8_nested_gan <../../../masterthesis_guide/results/run8_nested_gan/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`run8_vit_spectrogram <../../../masterthesis_guide/results/run8_vit_spectrogram/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`run8_dhct_gan <../../../masterthesis_guide/results/run8_dhct_gan/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`run8_demucs <../../../masterthesis_guide/results/run8_demucs/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`wega_demucs <../../../masterthesis_guide/results/wega_demucs/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`wega_dhct_gan <../../../masterthesis_guide/results/wega_dhct_gan/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`wega_nested_gan <../../../masterthesis_guide/results/wega_nested_gan/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`wega_vit_spectrogram <../../../masterthesis_guide/results/wega_vit_spectrogram/screen.json>`.
  Recorded measurements; see the associated protocol.
* :download:`table_phase0_legacy_aas_comparison <../../../masterthesis_guide/results/table_phase0_legacy_aas_comparison/metrics.csv>`.
  Separate current-pipeline residual comparison; not the historical metrics in Figure 21.
* :download:`table_phase1_inference_cost <../../../masterthesis_guide/results/table_phase1_inference_cost/metrics.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_phase1_unified_holdout_ranking <../../../masterthesis_guide/results/table_phase1_unified_holdout_ranking/metrics.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_phase2_pipeline_results <../../../masterthesis_guide/results/table_phase2_pipeline_results/metrics.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_phase2_selected_variant_results <../../../masterthesis_guide/results/table_phase2_selected_variant_results/metrics.csv>`.
  DHCT-GAN uses the recovered seven-context first Run-8 trial (3.7179 microvolts), not the original 2.442 microvolt Phase-2 arm or a completed grid winner.
* :download:`table_phase3_before_after <../../../masterthesis_guide/results/table_phase3_before_after/metrics.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_phase3_grid_runs <../../../masterthesis_guide/results/table_phase3_grid_runs/metrics.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_legacy_spike_preservation <../../../masterthesis_guide/results/table_legacy_spike_preservation/metrics.tsv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_spike_checkpoint_comparison <../../../masterthesis_guide/results/table_spike_checkpoint_comparison/metrics.tsv>`.
  Recorded measurements; see the associated protocol.
* :download:`table_phase3_wega_grid_runs <../../../masterthesis_guide/results/table_phase3_wega_grid_runs/metrics.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`spike_pipeline_summary <../../../masterthesis_guide/results/spike_pipeline_summary/pipeline_summary.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`spike_holdout_summary <../../../masterthesis_guide/results/spike_holdout_summary/holdout_summary.csv>`.
  Recorded measurements; see the associated protocol.
* :download:`spike_NiazyFMRI_spikes.truth <../../../masterthesis_guide/results/spike_NiazyFMRI_spikes.truth/NiazyFMRI_spikes.truth.json>`.
  Recorded measurements; see the associated protocol.
* :download:`spike_reference_residuals <../../../masterthesis_guide/results/spike_reference_residuals/residual_metrics.csv>`.
  Matched FARM and uncorrected baselines for thesis Table 21; same four injections and Fp1 measurement as the six neural rows.
* :download:`spike_reference_preservation <../../../masterthesis_guide/results/spike_reference_preservation/spike_preservation.csv>`.
  Matched FARM and uncorrected baselines for thesis Table 21; same four injections and Fp1 measurement as the six neural rows.
* :download:`phase2_holdout_replay <../../../masterthesis_guide/results/phase2_holdout_replay/metrics.csv>`.
  New CPU rerun on all 166 saved proof-fit holdout windows with the 13 retained primary deployment artifacts; not the missing original Figure-27 CSV. SNR uses global signal/error power, not a median across windows.

Phase 0 experiments
-------------------

* ``legacy_fc_dae``: legacy_cascaded_dae; outcome **valid**; verification ``blocked``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/legacy_fc_dae/provenance/legacy_dl_legacy_dl_training.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_0/legacy_fc_dae/provenance/legacy_dl_legacy_eval_variants.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_0/legacy_fc_dae/provenance/legacy_dl_legacy_metric_reproduction.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_0/legacy_fc_dae/provenance/legacy_dl_legacy_vs_current.json>`.
  :download:`Record 5 <../../../masterthesis_guide/experiments/phase_0/legacy_fc_dae/provenance/legacy_dl_torch_sweep.json>`.
  Artifacts: ``legacy_fc_dae_legacy_dl_cascade_pt``.
  Original full training environment and exact end-to-end equivalence are not established; current pipeline adapter has different epoch handling.

* ``refactoring_benchmark_legacy_vs_v2``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/refactoring_benchmark_legacy_vs_v2/metrics.json>`.

* ``refactoring_engineering_indicators``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/refactoring_engineering_indicators/metrics.json>`.

* ``refactoring_benchmark_models``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/refactoring_benchmark_models/metrics.json>`.

* ``refactoring_fidelity_register``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/refactoring_fidelity_register/metrics.json>`.

* ``refactoring_fidelity_per_model``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/refactoring_fidelity_per_model/metrics.csv>`.

* ``legacy_delivery``: legacy_cascaded_dae; outcome **valid**; verification ``blocked``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_0/legacy_delivery/provenance/legacy_dl_delivery_aas_vs_dae_metrics.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_0/legacy_delivery/provenance/legacy_dl_delivery_anc_contribution.json>`.
  Artifacts: ``legacy_fc_dae_dae_plotted_model_pt``.
  Original full training environment and exact end-to-end equivalence are not established; current pipeline adapter has different epoch handling.


Phase 1 experiments
-------------------

* ``holdout_aas_baseline``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_aas_baseline/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_aas_baseline/metrics.json>`.

* ``holdout_aas_naive_6nn``: engineering/baseline; outcome **valid**; verification ``not_run``.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_aas_naive_6nn/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_aas_naive_6nn/metrics.json>`.

* ``holdout_cascaded_context_dae``: cascaded_context_dae; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_context_dae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_context_dae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_context_dae/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_context_dae/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_context_dae/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_context_dae/provenance/training.jsonl>`.
  Artifacts: ``holdout_cascaded_context_dae_cascaded_context_dae_ts``.

* ``holdout_cascaded_dae``: cascaded_dae; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_dae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_dae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_dae/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_dae/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_dae/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_cascaded_dae/provenance/training.jsonl>`.
  Artifacts: ``holdout_cascaded_dae_cascaded_dae_ts``.

* ``holdout_conv_tasnet``: conv_tasnet; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_conv_tasnet/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_conv_tasnet/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_conv_tasnet/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_conv_tasnet/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_conv_tasnet/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_conv_tasnet/provenance/training.jsonl>`.
  Artifacts: ``holdout_conv_tasnet_conv_tasnet_ts``.

* ``holdout_d4pm``: d4pm; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_d4pm/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_d4pm/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_d4pm/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_d4pm/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_d4pm/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_d4pm/provenance/training.jsonl>`.
  Artifacts: ``holdout_d4pm_d4pm_ts``, ``holdout_d4pm_last_pt``.

* ``holdout_demucs``: demucs; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/provenance/training.jsonl>`.
  Artifacts: ``holdout_demucs_demucs_ts``, ``holdout_demucs_demucs_cpu_ts``.

* ``holdout_denoise_mamba``: denoise_mamba; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_denoise_mamba/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_denoise_mamba/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_denoise_mamba/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_denoise_mamba/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_denoise_mamba/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_denoise_mamba/provenance/training.jsonl>`.
  Artifacts: ``holdout_denoise_mamba_denoise_mamba_ts``, ``holdout_denoise_mamba_last_pt``.

* ``holdout_dhct_gan``: dhct_gan; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan/provenance/training.jsonl>`.
  Artifacts: ``holdout_dhct_gan_dhct_gan_ts``.

* ``holdout_dhct_gan_v2``: dhct_gan_v2; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan_v2/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan_v2/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan_v2/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan_v2/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan_v2/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_dhct_gan_v2/provenance/training.jsonl>`.
  Artifacts: ``holdout_dhct_gan_v2_dhct_gan_v2_ts``.

* ``holdout_dpae``: dpae; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_dpae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_dpae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_dpae/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_dpae/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_dpae/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_dpae/provenance/training.jsonl>`.
  Artifacts: ``holdout_dpae_dpae_ts``.

* ``holdout_ic_unet``: ic_unet; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_ic_unet/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_ic_unet/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_ic_unet/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_ic_unet/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_ic_unet/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_ic_unet/provenance/training.jsonl>`.
  Artifacts: ``holdout_ic_unet_ic_unet_ts``.

* ``holdout_nested_gan``: nested_gan; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_nested_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_nested_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_nested_gan/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_nested_gan/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_nested_gan/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_nested_gan/provenance/training.jsonl>`.
  Artifacts: ``holdout_nested_gan_nested_gan_ts``.

* ``holdout_sepformer``: sepformer; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_sepformer/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_sepformer/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_sepformer/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_sepformer/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_sepformer/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_sepformer/provenance/training.jsonl>`.
  Artifacts: ``holdout_sepformer_sepformer_ts``.

* ``holdout_st_gnn``: st_gnn; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_st_gnn/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_st_gnn/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_st_gnn/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_st_gnn/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_st_gnn/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_st_gnn/provenance/training.jsonl>`.
  Artifacts: ``holdout_st_gnn_st_gnn_ts``.

* ``holdout_vit_spectrogram``: vit_spectrogram; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_1/holdout_vit_spectrogram/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_1/holdout_vit_spectrogram/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_1/holdout_vit_spectrogram/evaluation_manifest.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_1/holdout_vit_spectrogram/metrics.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_1/holdout_vit_spectrogram/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_1/holdout_vit_spectrogram/provenance/training.jsonl>`.
  Artifacts: ``holdout_vit_spectrogram_vit_spectrogram_ts``, ``holdout_vit_spectrogram_vit_spectrogram_cpu_ts``.


Phase 2 experiments
-------------------

* ``deployment_cascaded_context_dae``: cascaded_context_dae_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_context_dae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_context_dae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_context_dae/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_context_dae/provenance/training.jsonl>`.
  Artifacts: ``deployment_cascaded_context_dae_cascaded_context_dae_deployment_ts``.

* ``deployment_cascaded_dae``: cascaded_dae_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_dae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_dae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_dae/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_cascaded_dae/provenance/training.jsonl>`.
  Artifacts: ``deployment_cascaded_dae_cascaded_dae_deployment_ts``.

* ``deployment_conv_tasnet``: conv_tasnet_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_conv_tasnet/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_conv_tasnet/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_conv_tasnet/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_conv_tasnet/provenance/training.jsonl>`.
  Artifacts: ``deployment_conv_tasnet_conv_tasnet_deployment_ts``.

* ``deployment_demucs``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_demucs/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_demucs/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_demucs/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_demucs/provenance/training.jsonl>`.
  Artifacts: ``deployment_demucs_demucs_deployment_ts``.

* ``deployment_denoise_mamba``: denoise_mamba_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_denoise_mamba/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_denoise_mamba/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_denoise_mamba/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_denoise_mamba/provenance/training.jsonl>`.
  Artifacts: ``deployment_denoise_mamba_denoise_mamba_deployment_ts``.

* ``deployment_dhct_gan``: dhct_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan/provenance/training.jsonl>`.
  Artifacts: ``deployment_dhct_gan_dhct_gan_deployment_ts``.

* ``deployment_dhct_gan_v2``: dhct_gan_v2_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan_v2/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan_v2/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan_v2/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_dhct_gan_v2/provenance/training.jsonl>`.
  Artifacts: ``deployment_dhct_gan_v2_dhct_gan_v2_deployment_ts``.

* ``deployment_dpae``: dpae_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_dpae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_dpae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_dpae/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_dpae/provenance/training.jsonl>`.
  Artifacts: ``deployment_dpae_dpae_deployment_ts``.

* ``deployment_ic_unet``: ic_unet_deployment_edition; outcome **invalid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_ic_unet/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_ic_unet/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_ic_unet/provenance/PROVENANCE.txt>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_ic_unet/provenance/facet_train_config.resolved.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_2/deployment_ic_unet/provenance/summary.json>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_2/deployment_ic_unet/provenance/training.jsonl>`.
  Artifacts: ``deployment_ic_unet_ic_unet_deployment_ts``.

* ``deployment_nested_gan``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_nested_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_nested_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_nested_gan/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_nested_gan/provenance/training.jsonl>`.
  Artifacts: ``deployment_nested_gan_nested_gan_deployment_ts``.

* ``deployment_sepformer``: sepformer_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_sepformer/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_sepformer/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_sepformer/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_sepformer/provenance/training.jsonl>`.
  Artifacts: ``deployment_sepformer_sepformer_deployment_ts``.

* ``deployment_st_gnn``: st_gnn_deployment_edition; outcome **invalid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_st_gnn/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_st_gnn/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_st_gnn/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_st_gnn/provenance/training.jsonl>`.
  Artifacts: ``deployment_st_gnn_st_gnn_deployment_ts``.

* ``deployment_vit_spectrogram``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/deployment_vit_spectrogram/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/deployment_vit_spectrogram/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/deployment_vit_spectrogram/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/deployment_vit_spectrogram/provenance/training.jsonl>`.
  Artifacts: ``deployment_vit_spectrogram_vit_spectrogram_deployment_ts``.

* ``deployment_d4pm``: d4pm_deployment_edition; outcome **unavailable**; verification ``blocked``.
  No final Phase-2 pipeline result; sampler was not wired into the original deployment adapter.

* ``training_run7_dpae``: dpae_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_dpae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_dpae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_dpae/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_dpae/provenance/training.jsonl>`.

* ``training_run7_ic_u_net``: ic_unet_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_ic_u_net/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_ic_u_net/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_ic_u_net/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_ic_u_net/provenance/training.jsonl>`.

* ``training_run7_cascaded_dae``: cascaded_dae_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_dae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_dae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_dae/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_dae/provenance/training.jsonl>`.

* ``training_run7_nested_gan``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_nested_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_nested_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_nested_gan/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_nested_gan/provenance/training.jsonl>`.

* ``training_run7_dhct_gan``: dhct_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan/provenance/training.jsonl>`.

* ``training_run7_dhct_gan_v2``: dhct_gan_v2_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan_v2/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan_v2/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan_v2/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_dhct_gan_v2/provenance/training.jsonl>`.

* ``training_run7_d4pm``: d4pm_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_d4pm/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_d4pm/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_d4pm/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_d4pm/provenance/training.jsonl>`.

* ``training_run7_denoisemamba``: denoise_mamba_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_denoisemamba/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_denoisemamba/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_denoisemamba/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_denoisemamba/provenance/training.jsonl>`.

* ``training_run7_conv_tasnet``: conv_tasnet_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_conv_tasnet/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_conv_tasnet/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_conv_tasnet/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_conv_tasnet/provenance/training.jsonl>`.

* ``training_run7_demucs``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_demucs/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_demucs/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_demucs/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_demucs/provenance/training.jsonl>`.

* ``training_run7_sepformer``: sepformer_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_sepformer/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_sepformer/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_sepformer/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_sepformer/provenance/training.jsonl>`.

* ``training_run7_vision_transformer``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_vision_transformer/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_vision_transformer/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_vision_transformer/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_vision_transformer/provenance/training.jsonl>`.

* ``training_run7_st_gnn``: st_gnn_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_st_gnn/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_st_gnn/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_st_gnn/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_st_gnn/provenance/training.jsonl>`.

* ``training_run7_cascaded_context_dae``: cascaded_context_dae_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_context_dae/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_context_dae/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_context_dae/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_2/training_run7_cascaded_context_dae/provenance/training.jsonl>`.


Phase 3 experiments
-------------------

* ``run8_nested_gan_lr0_00015_ch32_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch32_sisdr0_s42_epoch0080_val_loss0_1165_pt``.

* ``run8_nested_gan_lr0_00015_ch32_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch32_sisdr1_s42_epoch0080_val_loss_0_6612_pt``.

* ``run8_nested_gan_lr0_00015_ch32_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch32_sisdr3_s42_epoch0078_val_loss_2_2468_pt``.

* ``run8_nested_gan_lr0_00015_ch48_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch48_sisdr0_s42_epoch0074_val_loss0_1154_pt``.

* ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch48_sisdr1_s42_epoch0076_val_loss_0_6819_pt``.

* ``run8_nested_gan_lr0_00015_ch48_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch48_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch48_sisdr3_s42_epoch0076_val_loss_2_2806_pt``.

* ``run8_nested_gan_lr0_00015_ch72_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch72_sisdr0_s42_epoch0050_val_loss0_1109_pt``.

* ``run8_nested_gan_lr0_00015_ch72_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch72_sisdr1_s42_epoch0080_val_loss_0_6896_pt``.

* ``run8_nested_gan_lr0_00015_ch72_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_00015_ch72_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_00015_ch72_sisdr3_s42_epoch0080_val_loss_2_3020_pt``.

* ``run8_nested_gan_lr0_0005_ch32_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch32_sisdr0_s42_epoch0050_val_loss0_1082_pt``.

* ``run8_nested_gan_lr0_0005_ch32_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch32_sisdr1_s42_epoch0076_val_loss_0_7004_pt``.

* ``run8_nested_gan_lr0_0005_ch32_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch32_sisdr3_s42_epoch0078_val_loss_2_3288_pt``.

* ``run8_nested_gan_lr0_0005_ch48_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch48_sisdr0_s42_epoch0036_val_loss0_1129_pt``.

* ``run8_nested_gan_lr0_0005_ch48_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch48_sisdr1_s42_epoch0044_val_loss_0_6917_pt``.

* ``run8_nested_gan_lr0_0005_ch48_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch48_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch48_sisdr3_s42_epoch0064_val_loss_2_3169_pt``.

* ``run8_nested_gan_lr0_0005_ch72_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch72_sisdr0_s42_epoch0031_val_loss0_1069_pt``.

* ``run8_nested_gan_lr0_0005_ch72_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch72_sisdr1_s42_epoch0043_val_loss_0_6890_pt``.

* ``run8_nested_gan_lr0_0005_ch72_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0005_ch72_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0005_ch72_sisdr3_s42_epoch0076_val_loss_2_3132_pt``.

* ``run8_nested_gan_lr0_0015_ch32_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch32_sisdr0_s42_epoch0031_val_loss0_1106_pt``.

* ``run8_nested_gan_lr0_0015_ch32_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch32_sisdr1_s42_epoch0045_val_loss_0_6744_pt``.

* ``run8_nested_gan_lr0_0015_ch32_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch32_sisdr3_s42_epoch0054_val_loss_1_9012_pt``.

* ``run8_nested_gan_lr0_0015_ch48_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch48_sisdr0_s42_epoch0031_val_loss0_1109_pt``.

* ``run8_nested_gan_lr0_0015_ch48_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch48_sisdr1_s42_epoch0070_val_loss_0_6916_pt``.

* ``run8_nested_gan_lr0_0015_ch48_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch48_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch48_sisdr3_s42_epoch0069_val_loss_1_9745_pt``.

* ``run8_nested_gan_lr0_0015_ch72_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch72_sisdr0_s42_epoch0040_val_loss0_1110_pt``.

* ``run8_nested_gan_lr0_0015_ch72_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch72_sisdr1_s42_epoch0073_val_loss_0_6725_pt``.

* ``run8_nested_gan_lr0_0015_ch72_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_nested_gan_lr0_0015_ch72_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_nested_gan_lr0_0015_ch72_sisdr3_s42_epoch0078_val_loss_2_3274_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim120_sisdr0_s42_epoch0042_val_loss0_1268_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim120_sisdr1_s42_epoch0054_val_loss_0_6295_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim120_sisdr3_s42_epoch0058_val_loss_2_0728_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim192_sisdr0_s42_epoch0032_val_loss0_1364_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim192_sisdr1_s42_epoch0058_val_loss_0_6461_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim192_sisdr3_s42_epoch0057_val_loss_2_1295_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim288_sisdr0_s42_epoch0047_val_loss0_1362_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim288_sisdr1_s42_epoch0057_val_loss_0_6435_pt``.

* ``run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_00015_dim288_sisdr3_s42_epoch0058_val_loss_2_2299_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim120_sisdr0_s42_epoch0057_val_loss0_1390_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim120_sisdr1_s42_epoch0049_val_loss_0_6434_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim120_sisdr3_s42_epoch0059_val_loss_2_1583_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim192_sisdr0_s42_epoch0042_val_loss0_1345_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim192_sisdr1_s42_epoch0048_val_loss_0_6404_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim192_sisdr3_s42_epoch0057_val_loss_2_2335_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim288_sisdr0_s42_epoch0049_val_loss0_1295_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim288_sisdr1_s42_epoch0058_val_loss_0_6505_pt``.

* ``run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_0003_dim288_sisdr3_s42_epoch0053_val_loss_2_0939_pt``.

* ``run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim120_sisdr0_s42_epoch0056_val_loss0_1335_pt``.

* ``run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim120_sisdr1_s42_epoch0059_val_loss_0_6339_pt``.

* ``run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim120_sisdr3_s42_epoch0056_val_loss_2_1485_pt``.

* ``run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim192_sisdr0_s42_epoch0050_val_loss0_1256_pt``.

* ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42_epoch0047_val_loss_0_6089_pt``.

* ``run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim192_sisdr3_s42_epoch0058_val_loss_2_1225_pt``.

* ``run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim288_sisdr0_s42_epoch0060_val_loss0_1719_pt``.

* ``run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim288_sisdr1_s42_epoch0059_val_loss_0_5565_pt``.

* ``run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_vit_spectrogram_lr0_001_dim288_sisdr3_s42_epoch0053_val_loss_1_8869_pt``.

* ``run8_dhct_gan_lr0_0001_bc8_sisdr0_s42``: dhct_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_dhct_gan_lr0_0001_bc8_sisdr0_s42_epoch0076_val_loss0_3325_pt``.

* ``run8_dhct_gan_lr0_0001_bc8_sisdr1_s42``: dhct_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_dhct_gan_lr0_0001_bc8_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_dhct_gan_lr0_0001_bc8_sisdr1_s42_epoch0076_val_loss_0_2966_pt``.

* ``run8_demucs_lr0_0001_ic32_sisdr0_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic32_sisdr0_s42_epoch0060_val_loss0_1697_pt``.

* ``run8_demucs_lr0_0001_ic32_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr1_s42/config.yaml>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr1_s42/provenance/config.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr1_s42/provenance/training.jsonl>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr1_s42/provenance/same_grid_point_other_attempt.resolved.json>`.
  Artifacts: ``run8_demucs_lr0_0001_ic32_sisdr1_s42_epoch0057_val_loss_0_5721_pt``.

* ``run8_demucs_lr0_0001_ic32_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic32_sisdr3_s42_epoch0052_val_loss_1_9927_pt``.

* ``run8_demucs_lr0_0001_ic64_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic64_sisdr0_s42_epoch0017_val_loss0_1826_pt``.

* ``run8_demucs_lr0_0001_ic64_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic64_sisdr1_s42_epoch0057_val_loss_0_6382_pt``.

* ``run8_demucs_lr0_0001_ic64_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic64_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic64_sisdr3_s42_epoch0047_val_loss_2_1672_pt``.

* ``run8_demucs_lr0_0001_ic96_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic96_sisdr0_s42_epoch0060_val_loss0_1022_pt``.

* ``run8_demucs_lr0_0001_ic96_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic96_sisdr1_s42_epoch0028_val_loss_0_6227_pt``.

* ``run8_demucs_lr0_0001_ic96_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0001_ic96_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0001_ic96_sisdr3_s42_epoch0051_val_loss_2_1902_pt``.

* ``run8_demucs_lr0_0003_ic32_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic32_sisdr0_s42_epoch0051_val_loss0_1220_pt``.

* ``run8_demucs_lr0_0003_ic32_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic32_sisdr1_s42_epoch0057_val_loss_0_6596_pt``.

* ``run8_demucs_lr0_0003_ic32_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic32_sisdr3_s42_epoch0057_val_loss_2_1694_pt``.

* ``run8_demucs_lr0_0003_ic64_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic64_sisdr0_s42_epoch0054_val_loss0_0572_pt``.

* ``run8_demucs_lr0_0003_ic64_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic64_sisdr1_s42_epoch0058_val_loss_0_7183_pt``.

* ``run8_demucs_lr0_0003_ic64_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic64_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic64_sisdr3_s42_epoch0058_val_loss_2_1052_pt``.

* ``run8_demucs_lr0_0003_ic96_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic96_sisdr0_s42_epoch0058_val_loss0_0678_pt``.

* ``run8_demucs_lr0_0003_ic96_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic96_sisdr1_s42_epoch0060_val_loss_0_6901_pt``.

* ``run8_demucs_lr0_0003_ic96_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_0003_ic96_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_0003_ic96_sisdr3_s42_epoch0055_val_loss_2_2474_pt``.

* ``run8_demucs_lr0_001_ic32_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic32_sisdr0_s42_epoch0057_val_loss0_0876_pt``.

* ``run8_demucs_lr0_001_ic32_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic32_sisdr1_s42_epoch0051_val_loss_0_6520_pt``.

* ``run8_demucs_lr0_001_ic32_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic32_sisdr3_s42_epoch0059_val_loss_2_1442_pt``.

* ``run8_demucs_lr0_001_ic64_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic64_sisdr0_s42_epoch0057_val_loss0_0876_pt``.

* ``run8_demucs_lr0_001_ic64_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic64_sisdr1_s42_epoch0060_val_loss_0_6505_pt``.

* ``run8_demucs_lr0_001_ic64_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic64_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic64_sisdr3_s42_epoch0045_val_loss_2_0063_pt``.

* ``run8_demucs_lr0_001_ic96_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic96_sisdr0_s42_epoch0057_val_loss0_1009_pt``.

* ``run8_demucs_lr0_001_ic96_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic96_sisdr1_s42_epoch0057_val_loss_0_5991_pt``.

* ``run8_demucs_lr0_001_ic96_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/run8_demucs_lr0_001_ic96_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``run8_demucs_lr0_001_ic96_sisdr3_s42_epoch0045_val_loss_1_9411_pt``.

* ``wega_demucs_lr0_0001_ic32_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic32_sisdr0_s42_epoch0059_val_loss3_3849_pt``.

* ``wega_demucs_lr0_0001_ic32_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic32_sisdr1_s42_epoch0059_val_loss4_8967_pt``.

* ``wega_demucs_lr0_0001_ic32_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic32_sisdr3_s42_epoch0058_val_loss7_3953_pt``.

* ``wega_demucs_lr0_0001_ic64_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic64_sisdr0_s42_epoch0050_val_loss2_0701_pt``.

* ``wega_demucs_lr0_0001_ic64_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic64_sisdr1_s42_epoch0057_val_loss2_1167_pt``.

* ``wega_demucs_lr0_0001_ic64_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic64_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic64_sisdr3_s42_epoch0057_val_loss4_5462_pt``.

* ``wega_demucs_lr0_0001_ic96_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic96_sisdr0_s42_epoch0058_val_loss1_8175_pt``.

* ``wega_demucs_lr0_0001_ic96_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic96_sisdr1_s42_epoch0057_val_loss1_7040_pt``.

* ``wega_demucs_lr0_0001_ic96_sisdr3_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0001_ic96_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0001_ic96_sisdr3_s42_epoch0053_val_loss1_5065_pt``.

* ``wega_demucs_lr0_0003_ic32_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic32_sisdr0_s42_epoch0058_val_loss2_0158_pt``.

* ``wega_demucs_lr0_0003_ic32_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic32_sisdr1_s42_epoch0058_val_loss2_1068_pt``.

* ``wega_demucs_lr0_0003_ic32_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic32_sisdr3_s42_epoch0059_val_loss5_0412_pt``.

* ``wega_demucs_lr0_0003_ic64_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic64_sisdr0_s42_epoch0057_val_loss1_8825_pt``.

* ``wega_demucs_lr0_0003_ic64_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic64_sisdr1_s42_epoch0058_val_loss1_9014_pt``.

* ``wega_demucs_lr0_0003_ic64_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic64_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic64_sisdr3_s42_epoch0059_val_loss3_9644_pt``.

* ``wega_demucs_lr0_0003_ic96_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic96_sisdr0_s42_epoch0052_val_loss1_8947_pt``.

* ``wega_demucs_lr0_0003_ic96_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic96_sisdr1_s42_epoch0057_val_loss1_7964_pt``.

* ``wega_demucs_lr0_0003_ic96_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_0003_ic96_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_0003_ic96_sisdr3_s42_epoch0057_val_loss3_7835_pt``.

* ``wega_demucs_lr0_001_ic32_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic32_sisdr0_s42_epoch0057_val_loss2_3044_pt``.

* ``wega_demucs_lr0_001_ic32_sisdr1_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic32_sisdr1_s42_epoch0052_val_loss2_2927_pt``.

* ``wega_demucs_lr0_001_ic32_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic32_sisdr3_s42_epoch0060_val_loss4_5104_pt``.

* ``wega_demucs_lr0_001_ic64_sisdr0_s42``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic64_sisdr0_s42_epoch0055_val_loss2_1626_pt``.

* ``wega_demucs_lr0_001_ic64_sisdr1_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic64_sisdr1_s42_epoch0057_val_loss4_4680_pt``.

* ``wega_demucs_lr0_001_ic64_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic64_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic64_sisdr3_s42_epoch0060_val_loss4_6639_pt``.

* ``wega_demucs_lr0_001_ic96_sisdr0_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic96_sisdr0_s42_epoch0060_val_loss4_3618_pt``.

* ``wega_demucs_lr0_001_ic96_sisdr1_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic96_sisdr1_s42_epoch0058_val_loss9_2982_pt``.

* ``wega_demucs_lr0_001_ic96_sisdr3_s42``: demucs_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_demucs_lr0_001_ic96_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_demucs_lr0_001_ic96_sisdr3_s42_epoch0058_val_loss11_2743_pt``.

* ``wega_dhct_gan_lr0_0001_bc8_sisdr0_s42``: dhct_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_dhct_gan_lr0_0001_bc8_sisdr0_s42_epoch0073_val_loss4_8582_pt``.

* ``wega_dhct_gan_lr0_0001_bc8_sisdr1_s42``: dhct_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_dhct_gan_lr0_0001_bc8_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_dhct_gan_lr0_0001_bc8_sisdr1_s42_epoch0077_val_loss5_7550_pt``.

* ``wega_nested_gan_lr0_00015_ch32_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch32_sisdr0_s42_epoch0077_val_loss1_8833_pt``.

* ``wega_nested_gan_lr0_00015_ch32_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch32_sisdr1_s42_epoch0079_val_loss1_8087_pt``.

* ``wega_nested_gan_lr0_00015_ch32_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch32_sisdr3_s42_epoch0077_val_loss1_4974_pt``.

* ``wega_nested_gan_lr0_00015_ch48_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch48_sisdr0_s42_epoch0076_val_loss1_7495_pt``.

* ``wega_nested_gan_lr0_00015_ch48_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch48_sisdr1_s42_epoch0079_val_loss1_6051_pt``.

* ``wega_nested_gan_lr0_00015_ch48_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch48_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch48_sisdr3_s42_epoch0079_val_loss1_6734_pt``.

* ``wega_nested_gan_lr0_00015_ch72_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch72_sisdr0_s42_epoch0078_val_loss1_7329_pt``.

* ``wega_nested_gan_lr0_00015_ch72_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch72_sisdr1_s42_epoch0077_val_loss1_5811_pt``.

* ``wega_nested_gan_lr0_00015_ch72_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_00015_ch72_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_00015_ch72_sisdr3_s42_epoch0076_val_loss1_5599_pt``.

* ``wega_nested_gan_lr0_0005_ch32_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch32_sisdr0_s42_epoch0076_val_loss1_3657_pt``.

* ``wega_nested_gan_lr0_0005_ch32_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch32_sisdr1_s42_epoch0077_val_loss1_1089_pt``.

* ``wega_nested_gan_lr0_0005_ch32_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch32_sisdr3_s42_epoch0080_val_loss0_5922_pt``.

* ``wega_nested_gan_lr0_0005_ch48_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch48_sisdr0_s42_epoch0077_val_loss1_3465_pt``.

* ``wega_nested_gan_lr0_0005_ch48_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch48_sisdr1_s42_epoch0077_val_loss1_0067_pt``.

* ``wega_nested_gan_lr0_0005_ch48_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch48_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch48_sisdr3_s42_epoch0077_val_loss0_5362_pt``.

* ``wega_nested_gan_lr0_0005_ch72_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch72_sisdr0_s42_epoch0066_val_loss1_3109_pt``.

* ``wega_nested_gan_lr0_0005_ch72_sisdr1_s42``: nested_gan_deployment_edition; outcome **unavailable**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch72_sisdr1_s42_epoch0050_val_loss3_1086_pt``.

* ``wega_nested_gan_lr0_0005_ch72_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0005_ch72_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0005_ch72_sisdr3_s42_epoch0064_val_loss0_4536_pt``.

* ``wega_nested_gan_lr0_0015_ch32_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch32_sisdr0_s42_epoch0073_val_loss1_7954_pt``.

* ``wega_nested_gan_lr0_0015_ch32_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch32_sisdr1_s42_epoch0072_val_loss1_2293_pt``.

* ``wega_nested_gan_lr0_0015_ch32_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch32_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch32_sisdr3_s42_epoch0077_val_loss0_6700_pt``.

* ``wega_nested_gan_lr0_0015_ch48_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch48_sisdr0_s42_epoch0069_val_loss1_4802_pt``.

* ``wega_nested_gan_lr0_0015_ch48_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch48_sisdr1_s42_epoch0072_val_loss1_6985_pt``.

* ``wega_nested_gan_lr0_0015_ch48_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch48_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch48_sisdr3_s42_epoch0066_val_loss0_6095_pt``.

* ``wega_nested_gan_lr0_0015_ch72_sisdr0_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch72_sisdr0_s42_epoch0058_val_loss1_3818_pt``.

* ``wega_nested_gan_lr0_0015_ch72_sisdr1_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch72_sisdr1_s42_epoch0069_val_loss1_0167_pt``.

* ``wega_nested_gan_lr0_0015_ch72_sisdr3_s42``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_nested_gan_lr0_0015_ch72_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_nested_gan_lr0_0015_ch72_sisdr3_s42_epoch0073_val_loss0_5570_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim120_sisdr0_s42_epoch0060_val_loss2_8657_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim120_sisdr1_s42_epoch0060_val_loss3_1592_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim120_sisdr3_s42_epoch0055_val_loss3_8845_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim192_sisdr0_s42_epoch0060_val_loss2_5707_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim192_sisdr1_s42_epoch0060_val_loss2_8888_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim192_sisdr3_s42_epoch0059_val_loss3_3856_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim288_sisdr0_s42_epoch0060_val_loss2_5461_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim288_sisdr1_s42_epoch0059_val_loss3_0968_pt``.

* ``wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_00015_dim288_sisdr3_s42_epoch0059_val_loss3_6111_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim120_sisdr0_s42_epoch0060_val_loss2_6364_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim120_sisdr1_s42_epoch0060_val_loss3_1320_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim120_sisdr3_s42_epoch0054_val_loss4_0338_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim192_sisdr0_s42_epoch0057_val_loss2_6325_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim192_sisdr1_s42_epoch0053_val_loss3_1085_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim192_sisdr3_s42_epoch0056_val_loss3_8538_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim288_sisdr0_s42_epoch0058_val_loss2_6008_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim288_sisdr1_s42_epoch0052_val_loss3_5551_pt``.

* ``wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_0003_dim288_sisdr3_s42_epoch0054_val_loss4_4912_pt``.

* ``wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim120_sisdr0_s42_epoch0060_val_loss2_8028_pt``.

* ``wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim120_sisdr1_s42_epoch0058_val_loss3_4131_pt``.

* ``wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim120_sisdr3_s42_epoch0058_val_loss4_2850_pt``.

* ``wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim192_sisdr0_s42_epoch0058_val_loss3_1333_pt``.

* ``wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim192_sisdr1_s42_epoch0060_val_loss3_6385_pt``.

* ``wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim192_sisdr3_s42_epoch0055_val_loss4_7138_pt``.

* ``wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim288_sisdr0_s42_epoch0058_val_loss5_0533_pt``.

* ``wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim288_sisdr1_s42_epoch0060_val_loss4_3686_pt``.

* ``wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42/provenance/summary.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42/provenance/training.jsonl>`.
  Artifacts: ``wega_vit_spectrogram_lr0_001_dim288_sisdr3_s42_epoch0049_val_loss5_5802_pt``.

* ``spike_aware_demucs``: demucs_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/VERIFIED.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/summary.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/training.jsonl>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/comparison.json>`.
  :download:`Record 5 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/export_equivalence.json>`.
  :download:`Record 6 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/phase3/residual_metrics.csv>`.
  :download:`Record 7 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/phase3/residual_metrics.meta.json>`.
  :download:`Record 8 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/phase3/spike_preservation.csv>`.
  :download:`Record 9 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/spike_aware/residual_metrics.csv>`.
  :download:`Record 10 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/spike_aware/residual_metrics.meta.json>`.
  :download:`Record 11 <../../../masterthesis_guide/experiments/phase_3/spike_aware_demucs/provenance/phase3_spike_aware_comparison/demucs/pipeline/spike_aware/spike_preservation.csv>`.
  Artifacts: ``spike_aware_demucs_epoch0002_val_loss13_2668_pt``, ``spike_aware_demucs_best_checkpoint_cuda_ts``, ``spike_aware_demucs_best_checkpoint_cpu_ts``, ``spike_aware_demucs_comparison_baseline_ts``.

* ``spike_aware_nested_gan``: nested_gan_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/VERIFIED.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/summary.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/training.jsonl>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/comparison.json>`.
  :download:`Record 5 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/export_equivalence.json>`.
  :download:`Record 6 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/phase3/residual_metrics.csv>`.
  :download:`Record 7 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/phase3/residual_metrics.meta.json>`.
  :download:`Record 8 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/phase3/spike_preservation.csv>`.
  :download:`Record 9 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/spike_aware/residual_metrics.csv>`.
  :download:`Record 10 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/spike_aware/residual_metrics.meta.json>`.
  :download:`Record 11 <../../../masterthesis_guide/experiments/phase_3/spike_aware_nested_gan/provenance/phase3_spike_aware_comparison/nested_gan/pipeline/spike_aware/spike_preservation.csv>`.
  Artifacts: ``spike_aware_nested_gan_epoch0079_val_loss1_5766_pt``, ``spike_aware_nested_gan_best_checkpoint_cuda_ts``, ``spike_aware_nested_gan_best_checkpoint_cpu_ts``, ``spike_aware_nested_gan_comparison_baseline_ts``.

* ``spike_aware_vit_spectrogram``: vit_spectrogram_deployment_edition; outcome **valid**; verification ``not_run``.
  :download:`config <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/config.yaml>`.
  :download:`original_config <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/facet_train_config.resolved.json>`.
  :download:`Record 1 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/VERIFIED.json>`.
  :download:`Record 2 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/summary.json>`.
  :download:`Record 3 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/training.jsonl>`.
  :download:`Record 4 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/comparison.json>`.
  :download:`Record 5 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/export_equivalence.json>`.
  :download:`Record 6 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/phase3/residual_metrics.csv>`.
  :download:`Record 7 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/phase3/residual_metrics.meta.json>`.
  :download:`Record 8 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/phase3/spike_preservation.csv>`.
  :download:`Record 9 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/spike_aware/residual_metrics.csv>`.
  :download:`Record 10 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/spike_aware/residual_metrics.meta.json>`.
  :download:`Record 11 <../../../masterthesis_guide/experiments/phase_3/spike_aware_vit_spectrogram/provenance/phase3_spike_aware_comparison/vit_spectrogram/pipeline/spike_aware/spike_preservation.csv>`.
  Artifacts: ``spike_aware_vit_spectrogram_epoch0058_val_loss4_2328_pt``, ``spike_aware_vit_spectrogram_best_checkpoint_cuda_ts``, ``spike_aware_vit_spectrogram_best_checkpoint_cpu_ts``, ``spike_aware_vit_spectrogram_comparison_baseline_ts``.

Reproduction tools
------------------

* :download:`tools/dataset_building/build_spatiotemporal_reference_dataset.py <../../../tools/dataset_building/build_spatiotemporal_reference_dataset.py>`: Build the independent-reference Weg-A dataset with recorded augmentation settings.
* :download:`tools/dataset_building/derive_single_channel_weg_a.py <../../../tools/dataset_building/derive_single_channel_weg_a.py>`: Derive the single-channel locked Weg-A input without changing split membership.
* :download:`tools/dataset_building/edf_to_external_clean.py <../../../tools/dataset_building/edf_to_external_clean.py>`: Prepare explicit external clean-reference recordings for dataset construction.
* :download:`tools/dataset_building/export_channel_subset.py <../../../tools/dataset_building/export_channel_subset.py>`: Preserve a specified EEG channel subset and its order.
* :download:`tools/dataset_building/extract_niazy_aas_pca4_artifact.py <../../../tools/dataset_building/extract_niazy_aas_pca4_artifact.py>`: Recover the recorded FARM/PCA artifact library from Niazy.
* :download:`tools/dataset_building/lock_holdout_split.py <../../../tools/dataset_building/lock_holdout_split.py>`: Persist the selection/holdout partition used by the locked Weg-A protocol.
* :download:`tools/gpu_fleet/check_torch.py <../../../tools/gpu_fleet/check_torch.py>`: Validate the training backend on an explicitly selected worker.
* :download:`tools/gpu_fleet/fleet.py <../../../tools/gpu_fleet/fleet.py>`: Schedule independent single-GPU training jobs with a local queue.
* :download:`tools/gpu_fleet/fetch_runpod_results.sh <../../../tools/gpu_fleet/fetch_runpod_results.sh>`: Retrieve selected completed job records from an explicitly selected worker.
* :download:`tools/gpu_fleet/run_remote_training.sh <../../../tools/gpu_fleet/run_remote_training.sh>`: Run one configuration under a per-GPU lock and retain logs/exit status.
* :download:`tools/gpu_fleet/sync_worktree_to_runpod.sh <../../../tools/gpu_fleet/sync_worktree_to_runpod.sh>`: Transfer a selected checkout to an explicitly selected worker, excluding large data and artifacts.
* :download:`tools/gpu_fleet/workers.example.yaml <../../../tools/gpu_fleet/workers.example.yaml>`: Document the minimal worker configuration without personal hosts or credentials.
* :download:`tools/refactoring_comparison/benchmark_legacy_vs_v2.py <../../../tools/refactoring_comparison/benchmark_legacy_vs_v2.py>`: Compare matched classical pipelines in separate legacy/current interpreters.
* :download:`tools/refactoring_comparison/engineering_indicators.py <../../../tools/refactoring_comparison/engineering_indicators.py>`: Measure the fixed source-scope engineering indicators used by the thesis.
* :download:`tools/refactoring_comparison/benchmark_models.py <../../../tools/refactoring_comparison/benchmark_models.py>`: Measure Phase-1 inference cost with fresh-process repetitions and the saved holdout.
* :download:`tools/training/grid_search_run7.py <../../../tools/training/grid_search_run7.py>`: Run the recorded proof-fit/Weg-A screening and confirmation protocols.
* :download:`tools/training/derive_wega_loss_weights.py <../../../tools/training/derive_wega_loss_weights.py>`: Derive the recorded Weg-A objective scaling from the selected dataset.
* :download:`tools/evaluation/compare_phase3_spike_aware.py <../../../tools/evaluation/compare_phase3_spike_aware.py>`: Compare original and retrained models on the same locked holdout and labelled spikes.
* :download:`tools/evaluation/eval_run6_spike_preservation.py <../../../tools/evaluation/eval_run6_spike_preservation.py>`: Evaluate waveform fidelity on labelled spike examples with per-example outputs.
* :download:`tools/evaluation/paired_spike_comparison.py <../../../tools/evaluation/paired_spike_comparison.py>`: Compare matching examples while preserving event-level dependence.
* :download:`tools/pipeline_demo/inject_spikes.py <../../../tools/pipeline_demo/inject_spikes.py>`: Create the four matched 100-microvolt pipeline injections and their truth record.
* :download:`tools/pipeline_demo/measure_arms.py <../../../tools/pipeline_demo/measure_arms.py>`: Measure residual artifact and seam validity under the recorded pipeline protocol.
* :download:`tools/pipeline_demo/measure_spike_preservation.py <../../../tools/pipeline_demo/measure_spike_preservation.py>`: Measure the difference between matched injected/non-injected pipeline arms.
* :download:`tools/pipeline_demo/run_arms.py <../../../tools/pipeline_demo/run_arms.py>`: Generate matched FARM, uncorrected and selected model arms through the shared library pipeline.
* :download:`tools/plotting/artifact_amplitude.py <../../../tools/plotting/artifact_amplitude.py>`: Render the fixed proof-fit example used in thesis Figure 1.
* :download:`tools/plotting/plot_phase0_signal_comparison.py <../../../tools/plotting/plot_phase0_signal_comparison.py>`: Render the distinct delivery-run and earlier Phase-0 diagnostics (Figures 21–23).
* :download:`tools/plotting/plot_phase1_holdout_signal_comparison.py <../../../tools/plotting/plot_phase1_holdout_signal_comparison.py>`: Render one shared holdout epoch from original prediction arrays (Figure 25).
* :download:`tools/plotting/plot_phase2_family_artifact_window_fast.py <../../../tools/plotting/plot_phase2_family_artifact_window_fast.py>`: Render selected model corrections over one aligned artifact epoch (Figure 28).
* :download:`tools/plotting/plot_phase2_pipeline_signal_comparison.py <../../../tools/plotting/plot_phase2_pipeline_signal_comparison.py>`: Render matched primary deployment arms and their five-best comparison (Figure 30).
* :download:`tools/plotting/spike_waveforms.py <../../../tools/plotting/spike_waveforms.py>`: Render original/retrained/FARM/reference responses to the four injections (Figure 33).
* :download:`tools/plotting/thesis_figures.py <../../../tools/plotting/thesis_figures.py>`: Render rankings and grid landscapes from canonical CSV evidence (Figures 24, 31, 32 and Weg-A).
* :download:`tools/plotting/training_histories.py <../../../tools/plotting/training_histories.py>`: Render the original phase-specific training histories (Figures 26 and 29).
* :download:`tools/diagrams/build_selected_variants.py <../../../tools/diagrams/build_selected_variants.py>`: Draw the actual selected model variants from catalog configurations and canonical results.
* :download:`tools/diagrams/facetpy_svg.py <../../../tools/diagrams/facetpy_svg.py>`: Shared SVG primitives required by retained thesis diagram generators.
* :download:`tools/diagrams/build_facetpy_clean_reference_dataset.py <../../../tools/diagrams/build_facetpy_clean_reference_dataset.py>`: Draw the independent-reference dataset construction flow.
* :download:`tools/diagrams/build_facetpy_dl_framework.py <../../../tools/diagrams/build_facetpy_dl_framework.py>`: Draw the configuration-driven training and inference framework.
* :download:`tools/diagrams/build_facetpy_proof_fit_dataset.py <../../../tools/diagrams/build_facetpy_proof_fit_dataset.py>`: Draw the AAS-derived proof-fit construction and run artifacts.
* :download:`tools/diagrams/build_facetpy_training_to_deployment.py <../../../tools/diagrams/build_facetpy_training_to_deployment.py>`: Draw training orchestration and correction integration with explicit model ownership.

Availability gaps
-----------------

* {'experiment': 'run8_demucs_lr0_0001_ic32_sisdr1_s42', 'source': 'output/run8_fetched/gridpod4/grids/run8/demucs/lr0.0001_ic32_sisdr1_s42/grid_demucs_lr0.0001_ic32_sisdr1_s42_20260912_184729/facet_train_config.resolved.json', 'reason': 'Original full resolved configuration missing; executable reconstruction is explicitly identified and preserves the own-run training record.'}
* {'scope': 'Phase 0 original execution', 'reason': 'The retained legacy checkpoints and records do not establish the exact original FACETpy source revision and complete dependency environment. Current-pipeline adapter parity is verified separately.'}
* {'scope': 'Figure 27', 'reason': 'Original per-model CSV missing; original embedded figure retained and new CPU replay stored separately.'}
* {'scope': 'Figures 2–5 and 34', 'reason': 'Original images retained. Editable generators were not recovered. Figure 34 visibly uses Fp1, 29.5–160 s; its broad caption does not establish full-recording coverage.'}
