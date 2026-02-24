Project Overview
================

.. raw:: html

   <p style="text-align: center;">
     <span class="logo-container">
       <img src="_static/logo_light_theme.png" class="logo-light" alt="FACETpy logo" width="200"/>
       <img src="_static/logo_dark_theme.png" class="logo-dark" alt="FACETpy logo" width="200"/>
     </span>
   </p>

What is FACETpy?
----------------

**FACET** (*Flexible Artifact Correction and Evaluation Toolbox*) is a comprehensive Python library specifically designed for processing electroencephalogram (EEG) data recorded simultaneously with functional magnetic resonance imaging (fMRI). The primary challenge this toolbox addresses is the correction of severe artifacts introduced by the fMRI scanning process, which can completely obscure the underlying neural signals in EEG recordings.

The Problem: EEG-fMRI Artifacts
-------------------------------

When recording EEG inside an fMRI scanner, several types of artifacts are introduced:

**Gradient Artifacts (Slice Timing Artifacts)**
   Large-amplitude artifacts (up to 100 times the EEG signal amplitude) caused by rapidly switching magnetic field gradients during fMRI slice acquisition. These occur precisely at each fMRI volume acquisition and can completely mask the EEG signal.

**Ballistocardiogram (BCG) Artifacts**
   Artifacts caused by blood flow and cardiac pulsations in the strong magnetic field, creating periodic distortions in the EEG signal synchronized with the heartbeat.

**Movement Artifacts**
   Additional artifacts caused by head movements within the magnetic field, which can vary unpredictably during scanning sessions.

Without proper correction, these artifacts render EEG data collected during fMRI scanning virtually unusable for neuroscientific analysis.

Core Methodology: Averaged Artifact Subtraction (AAS)
-----------------------------------------------------

FACETpy implements the **Averaged Artifact Subtraction (AAS)** method, a proven technique for removing fMRI-related artifacts from EEG data:

1. **Artifact Template Creation**: The software identifies repeating artifact patterns by detecting fMRI trigger events and extracting artifact segments around each trigger.

2. **Template Averaging**: Multiple artifact occurrences are averaged to create a clean artifact template that represents the consistent artifact pattern while reducing random noise.

3. **Artifact Subtraction**: The averaged artifact template is then subtracted from each individual artifact occurrence in the original EEG data.

4. **Adaptive Processing**: The toolbox includes advanced alignment algorithms to account for slight timing variations between artifact occurrences.

Key Features and Capabilities
-----------------------------

**Multi-Format Data Support**
   - Import EEG data from various formats including EDF, BIDS (Brain Imaging Data Structure), and raw MNE formats
   - Export processed data to standard formats for downstream analysis
   - Full BIDS compatibility for modern neuroimaging workflows

**Comprehensive Artifact Correction Pipeline**
   - Automated trigger detection and missing trigger interpolation
   - Sub-sample precision trigger alignment using cross-correlation
   - Motion-informed artifact correction using head motion parameters
   - Ballistocardiogram (BCG) artifact detection and correction
   - Adaptive Noise Cancellation (ANC) for residual artifact removal

**Robust Evaluation Framework**
   - Multiple quality metrics including Signal-to-Noise Ratio (SNR), Root Mean Square (RMS) ratios
   - Automated before/after correction comparisons
   - Visualization tools for quality assessment
   - Statistical evaluation of correction effectiveness

**Flexible Processing Pipeline**
   - Modular architecture allowing customization of correction steps
   - Pre- and post-processing options including filtering, resampling, and channel selection
   - Integration with MNE-Python ecosystem for additional EEG analysis capabilities

Use Cases and Applications
--------------------------

**Simultaneous EEG-fMRI Research**
   Enable researchers to collect high-quality EEG data during fMRI scanning, opening possibilities for multimodal brain imaging studies that combine the temporal precision of EEG with the spatial resolution of fMRI.

**Clinical Neuroimaging**
   Support clinical applications where both EEG and fMRI data are needed simultaneously, such as pre-surgical epilepsy evaluation or studies of brain disorders requiring both modalities.

**Neuroscience Research**
   Facilitate studies of brain networks, cognitive processes, and neurological conditions that benefit from simultaneous measurement of electrical brain activity and blood flow changes.

**Method Development**
   Provide a platform for researchers developing new artifact correction methods, with comprehensive evaluation tools and flexible framework architecture.

Technical Architecture
----------------------

FACETpy is built with a modular, pipeline-based architecture. Every processing step is
a **Processor** that receives a ``ProcessingContext``, performs a single operation, and
returns a new context. Processors are composed into **Pipelines** that execute them in
sequence and collect results in a ``PipelineResult``.

The main modules are:

- ``facet.core`` — ``Pipeline``, ``Processor``, ``ProcessingContext``, ``PipelineResult``, registry
- ``facet.io`` — ``Loader``, ``BIDSLoader``, ``EDFExporter``, ``BIDSExporter``
- ``facet.preprocessing`` — Filters, resampling, trigger detection, alignment
- ``facet.correction`` — ``AASCorrection``, ``ANCCorrection``, ``PCACorrection``
- ``facet.evaluation`` — ``SNRCalculator``, ``RMSCalculator``, ``MetricsReport``, and more

The toolbox is built on top of established scientific Python libraries:
   - **MNE-Python**: For EEG data handling and basic processing
   - **NumPy/SciPy**: For numerical computations and signal processing
   - **Scikit-learn**: For PCA-based correction
   - **Matplotlib**: For visualization and plotting
   - **rich**: For terminal progress display and formatted output

Integration and Workflow
------------------------

FACETpy integrates seamlessly into standard neuroimaging workflows:

1. **Data Import**: Load EEG data from EDF, GDF, or BIDS formats
2. **Preprocessing**: Apply filtering, resampling, trigger detection, and alignment
3. **Artifact Correction**: Apply AAS, ANC, or PCA correction
4. **Quality Assessment**: Evaluate correction effectiveness using built-in metrics
5. **Export**: Save corrected data in standard formats for further analysis

The toolbox maintains compatibility with the broader MNE-Python ecosystem, allowing users to apply additional EEG analysis methods after artifact correction.

Performance and Validation
--------------------------

FACETpy has been designed with performance and accuracy in mind:

- **Efficient algorithms**: Optimized implementations for processing large datasets
- **Memory management**: Smart data handling to work with limited system resources  
- **Validation tools**: Built-in quality metrics to assess correction performance
- **Flexible evaluation**: Multiple assessment methods to suit different research needs

Future Directions
-----------------

FACETpy continues to evolve with new developments in:

- Advanced machine learning techniques for artifact correction
- Real-time processing capabilities for online applications  
- Enhanced integration with other neuroimaging analysis tools
- Extended support for different EEG acquisition systems and formats

The toolbox represents a comprehensive solution for one of the most challenging problems in multimodal neuroimaging, making simultaneous EEG-fMRI studies more accessible and reliable for the research community.
