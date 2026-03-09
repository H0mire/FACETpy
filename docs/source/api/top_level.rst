Top-Level API
=============

The ``facet`` package re-exports the most commonly used classes and helper
functions.

.. currentmodule:: facet

Convenience Functions
---------------------

.. autofunction:: load

.. autofunction:: export

.. autofunction:: create_standard_pipeline

Configuration
-------------

.. autofunction:: get_config

.. autofunction:: set_config

.. autofunction:: reset_config

Commonly Used Re-Exports
------------------------

Core
~~~~

.. autoclass:: Pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: ProcessingContext
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: BatchResult
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Errors
~~~~~~

.. autoclass:: ProcessorError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: PipelineError
   :members:
   :show-inheritance:
   :no-index:
