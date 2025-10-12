Core API
========

The core module provides the fundamental building blocks of FACETpy's architecture.

.. currentmodule:: facet.core

Base Classes
------------

Processor
~~~~~~~~~

.. autoclass:: Processor
   :members:
   :undoc-members:
   :show-inheritance:

ProcessingContext
~~~~~~~~~~~~~~~~~

.. autoclass:: ProcessingContext
   :members:
   :undoc-members:
   :show-inheritance:

ProcessingMetadata
~~~~~~~~~~~~~~~~~~

.. autoclass:: ProcessingMetadata
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
--------

Pipeline
~~~~~~~~

.. autoclass:: Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

PipelineResult
~~~~~~~~~~~~~~

.. autoclass:: PipelineResult
   :members:
   :undoc-members:
   :show-inheritance:

Composite Processors
--------------------

SequenceProcessor
~~~~~~~~~~~~~~~~~

.. autoclass:: SequenceProcessor
   :members:
   :undoc-members:
   :show-inheritance:

ConditionalProcessor
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConditionalProcessor
   :members:
   :undoc-members:
   :show-inheritance:

SwitchProcessor
~~~~~~~~~~~~~~~

.. autoclass:: SwitchProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Registry
--------

.. autofunction:: register_processor

.. autofunction:: get_processor

.. autofunction:: list_processors

Parallel Execution
------------------

ParallelExecutor
~~~~~~~~~~~~~~~~

.. autoclass:: ParallelExecutor
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. autoclass:: ProcessorError
   :members:
   :show-inheritance:

.. autoclass:: ProcessorValidationError
   :members:
   :show-inheritance:

.. autoclass:: PipelineError
   :members:
   :show-inheritance:
