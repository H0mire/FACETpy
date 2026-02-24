Preprocessing API
=================

Preprocessing processors for preparing EEG data before correction.

.. currentmodule:: facet.preprocessing

Filtering
---------

Filter
~~~~~~

.. autoclass:: Filter
   :members:
   :undoc-members:
   :show-inheritance:

HighPassFilter
~~~~~~~~~~~~~~

.. autoclass:: HighPassFilter
   :members:
   :undoc-members:
   :show-inheritance:

LowPassFilter
~~~~~~~~~~~~~

.. autoclass:: LowPassFilter
   :members:
   :undoc-members:
   :show-inheritance:

BandPassFilter
~~~~~~~~~~~~~~

.. autoclass:: BandPassFilter
   :members:
   :undoc-members:
   :show-inheritance:

NotchFilter
~~~~~~~~~~~

.. autoclass:: NotchFilter
   :members:
   :undoc-members:
   :show-inheritance:

MATLABPreFilter
~~~~~~~~~~~~~~~

.. autoclass:: MATLABPreFilter
   :members:
   :undoc-members:
   :show-inheritance:

Resampling
----------

Resample
~~~~~~~~

.. autoclass:: Resample
   :members:
   :undoc-members:
   :show-inheritance:

UpSample
~~~~~~~~

.. autoclass:: UpSample
   :members:
   :undoc-members:
   :show-inheritance:

DownSample
~~~~~~~~~~

.. autoclass:: DownSample
   :members:
   :undoc-members:
   :show-inheritance:

Trigger Detection
-----------------

TriggerDetector
~~~~~~~~~~~~~~~

.. autoclass:: TriggerDetector
   :members:
   :undoc-members:
   :show-inheritance:

QRSTriggerDetector
~~~~~~~~~~~~~~~~~~

.. autoclass:: QRSTriggerDetector
   :members:
   :undoc-members:
   :show-inheritance:

MissingTriggerDetector
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MissingTriggerDetector
   :members:
   :undoc-members:
   :show-inheritance:

MissingTriggerCompleter
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MissingTriggerCompleter
   :members:
   :undoc-members:
   :show-inheritance:

SliceTriggerGenerator
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SliceTriggerGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Alignment
---------

TriggerAligner
~~~~~~~~~~~~~~

.. autoclass:: TriggerAligner
   :members:
   :undoc-members:
   :show-inheritance:

SubsampleAligner
~~~~~~~~~~~~~~~~

.. autoclass:: SubsampleAligner
   :members:
   :undoc-members:
   :show-inheritance:

Transforms
----------

Crop
~~~~

.. autoclass:: Crop
   :members:
   :undoc-members:
   :show-inheritance:

MagicErasor
~~~~~~~~~~~

.. autoclass:: MagicErasor
   :members:
   :undoc-members:
   :show-inheritance:

RawTransform
~~~~~~~~~~~~

.. autoclass:: RawTransform
   :members:
   :undoc-members:
   :show-inheritance:

PickChannels
~~~~~~~~~~~~

.. autoclass:: PickChannels
   :members:
   :undoc-members:
   :show-inheritance:

DropChannels
~~~~~~~~~~~~

.. autoclass:: DropChannels
   :members:
   :undoc-members:
   :show-inheritance:

ChannelStandardizer
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ChannelStandardizer
   :members:
   :undoc-members:
   :show-inheritance:

PrintMetric
~~~~~~~~~~~

.. autoclass:: PrintMetric
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
-----------

AnalyzeDataReport
~~~~~~~~~~~~~~~~~

.. autoclass:: AnalyzeDataReport
   :members:
   :undoc-members:
   :show-inheritance:

CheckDataReport
~~~~~~~~~~~~~~~

.. autoclass:: CheckDataReport
   :members:
   :undoc-members:
   :show-inheritance:
