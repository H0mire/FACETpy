Configuration
=============

FACETpy provides a central configuration layer with explicit precedence.
This lets you keep project-wide defaults while still overriding settings for a
specific run or notebook session.

Precedence
----------

Configuration sources are applied in this order (highest to lowest):

1. Runtime overrides via ``facet.set_config(...)``
2. Environment variables
3. Global config file (TOML)
4. Built-in defaults

In short: runtime values always win over environment values, and environment
values win over file values.


Available Keys
--------------

``console_mode``
~~~~~~~~~~~~~~~~

Controls how FACETpy renders console output.

- Allowed values: ``"classic"``, ``"modern"``
- Aliases accepted by runtime/env/file parsing: ``"legacy"``, ``"loguru"`` (both map to ``"classic"``)
- Default: ``"classic"``

Use ``"modern"`` when running in an interactive terminal where the Rich-based
live dashboard is desired. If modern mode is unavailable at runtime, FACETpy
falls back to classic output automatically.


``log_level``
~~~~~~~~~~~~~

Controls the minimum level shown in the console sink.

- Allowed values: ``TRACE``, ``DEBUG``, ``INFO``, ``SUCCESS``, ``WARNING``, ``ERROR``, ``CRITICAL``
- Default: ``"INFO"``

Typical choices:

- ``INFO`` for day-to-day use
- ``DEBUG`` for processor-level diagnostics
- ``WARNING`` for minimal operational output


``log_file``
~~~~~~~~~~~~

Enables or disables per-run log file output.

- Allowed values: boolean
- Default: ``False``

When enabled, FACETpy creates a timestamped log file for each run.


``log_file_level``
~~~~~~~~~~~~~~~~~~

Controls the minimum level written to the log file sink.

- Allowed values: ``TRACE``, ``DEBUG``, ``INFO``, ``SUCCESS``, ``WARNING``, ``ERROR``, ``CRITICAL``
- Default: ``"DEBUG"``

This is independent from ``log_level``. For example, you can keep a quiet
console (``INFO``) while writing detailed file logs (``DEBUG``).


``log_dir``
~~~~~~~~~~~

Directory where per-run log files are written when ``log_file=True``.

- Allowed values: path string or ``None``
- Default: ``None`` (resolves to ``./logs`` relative to current working directory)

FACETpy creates the directory automatically if it does not exist.


``auto_logging``
~~~~~~~~~~~~~~~~

Controls whether FACETpy auto-configures logging during import.

- Allowed values: boolean
- Default: ``True``

Set this to ``False`` only when you want to take full manual control of logging
setup in your host application.


Python API Syntax
-----------------

.. code-block:: python

   import facet

   # Read full resolved config
   cfg = facet.get_config()

   # Read one key
   mode = facet.get_config("console_mode")

   # Set runtime overrides (highest precedence)
   facet.set_config(
       console_mode="modern",
       log_level="DEBUG",
       log_file=True,
       log_file_level="DEBUG",
       log_dir="./logs",
   )

   # Reset runtime overrides
   facet.reset_config()

Notes:

- ``set_config()`` validates keys and values.
- Unknown keys raise ``KeyError``.
- Invalid values raise ``ValueError``.
- By default, ``set_config()`` and ``reset_config()`` reconfigure logging immediately.


Environment Variable Syntax
---------------------------

Supported environment variables:

- ``FACET_CONSOLE_MODE`` -> ``console_mode``
- ``FACET_LOG_CONSOLE_LEVEL`` -> ``log_level``
- ``FACET_LOG_FILE`` -> ``log_file``
- ``FACET_LOG_FILE_LEVEL`` -> ``log_file_level``
- ``FACET_LOG_DIR`` -> ``log_dir``
- ``FACET_DISABLE_AUTO_LOGGING`` -> inverse of ``auto_logging``
- ``FACET_CONFIG_FILE`` -> explicit path to the TOML config file

Example:

.. code-block:: bash

   export FACET_CONSOLE_MODE=modern
   export FACET_LOG_CONSOLE_LEVEL=INFO
   export FACET_LOG_FILE=1
   export FACET_LOG_FILE_LEVEL=DEBUG
   export FACET_LOG_DIR=/tmp/facet-logs

For booleans, FACETpy accepts: ``1/0``, ``true/false``, ``yes/no``, ``on/off``
(case-insensitive).


TOML File Syntax
----------------

Default config file path:

- ``~/.config/facetpy/config.toml``

You can override the path with ``FACET_CONFIG_FILE``.

Simple form:

.. code-block:: toml

   [facet]
   console_mode = "classic"
   log_level = "INFO"
   log_file = false
   log_file_level = "DEBUG"
   log_dir = "/tmp/facet-logs"
   auto_logging = true

Optional nested logging form:

.. code-block:: toml

   [facet.logging]
   console_mode = "modern"
   level = "INFO"
   file_enabled = true
   file_level = "DEBUG"
   dir = "/tmp/facet-logs"


Practical Recommendation
------------------------

For research projects, keep stable team defaults in the TOML file, allow
cluster-/CI-specific overrides via environment variables, and use
``facet.set_config(...)`` only for explicit per-run deviations in scripts or
notebooks.
