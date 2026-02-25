KI-Agenten fuer Prozessor-Entwicklung
=====================================

Diese Seite beschreibt einen pragmatischen Workflow, um Prozessoren in FACETpy mit KI-Agenten
zu entwickeln, zu refaktorieren und zu reviewen.

Pflicht-Kontext fuer Prompts
----------------------------

Wenn du einen KI-Agenten fuer Prozessor-Code nutzt, gib immer den Inhalt von
``docs/PROCESSOR_GUIDELINES.md`` im Prompt mit (oder als zusaetzliche Kontextdatei).
Ohne diese Guidelines sind Stil- und Architekturabweichungen sehr wahrscheinlich.

Empfohlener Minimal-Kontext pro Prompt:

- Ziel des Prozessors in einem Satz
- Betroffene Dateien (z. B. ``src/facet/correction/...`` und ``tests/...``)
- Akzeptanzkriterien (funktional + technisch)
- ``docs/PROCESSOR_GUIDELINES.md`` als verpflichtende Richtlinie

Prompt-Template (kurz)
----------------------

.. code-block:: text

   Aufgabe: [kurze Beschreibung]
   Scope: [Dateien/Module]
   Anforderungen:
   - Folge strikt den Regeln aus docs/PROCESSOR_GUIDELINES.md.
   - Schreibe/aktualisiere passende Tests.
   - Halte dich an bestehende APIs und Namenskonventionen.
   Definition of Done:
   - Tests laufen lokal durch.
   - Doku ist bei Verhaltensaenderungen aktualisiert.

Tipps und Tricks
----------------

- Arbeite in kleinen, klaren Schritten statt in grossen Sammel-Prompts.
- Fordere immer einen konkreten Diff-Plan an, bevor der Agent mehrere Dateien aendert.
- Lass den Agenten zuerst Tests formulieren oder bestehende Tests erweitern, dann implementieren.
- Gib Randfaelle explizit vor (z. B. fehlende Trigger, ungueltige Parameter, leere Daten).
- Verlange bei Performance-kritischen Aenderungen eine kurze Begruendung zur Laufzeit-/Speicherwirkung.

Review-Workflow mit Codex
-------------------------

Wenn Codex installiert ist, kannst du vor dem manuellen Review direkt den VSCode-Task
``Review: Uncommitted Changes (Codex)`` ausfuehren (``Tasks: Run Task`` in VSCode).
Fuer Branch-Vergleiche steht zusaetzlich ``Review: Against Branch (Codex)`` zur Verfuegung.

Sinnvolle zusaetzliche Kontexte
-------------------------------

Je nach Aenderung sind diese Dateien als Zusatzkontext oft hilfreich:

- ``AGENTS.md`` (Repo-spezifische Arbeitsregeln)
- ``docs/source/development/contributing.rst`` (lokaler Workflow und Checks)
- ``.vscode/tasks.json`` (standardisierte Build-/Test-/Review-Tasks)
- Betroffene API- oder User-Guide-Seiten unter ``docs/source/``

Wenn ihr haeufig mit Agenten arbeitet, lohnt sich ein eigenes Playbook wie
``AI_AGENT_PLAYBOOK.md`` mit team-spezifischen Prompt-Vorlagen und Do/Don'ts.

Optionale Skill-Ideen
---------------------

Fuer wiederkehrende Aufgaben koennt ihr zusaetzlich eigene Agent-Skills aufsetzen, z. B.:

- ``processor-implementation``: Erstellt neuen Prozessor + Grundtests nach Guidelines.
- ``processor-review-check``: Prueft Diffs gegen ``docs/PROCESSOR_GUIDELINES.md``.
- ``processor-doc-sync``: Prueft, ob API-/User-Guide-Doku bei Codeaenderungen nachgezogen wurde.

Starter Kit im Repository
-------------------------

Im Repository gibt es jetzt ein kleines Starter-Kit fuer Agent-Workflows:

- ``AI_AGENT_PLAYBOOK.md`` (standardisierter Ablauf)
- ``AI_PROMPT_TEMPLATES.md`` (Prompt-Vorlagen fuer Implementierung/Refactor/Review)
- ``scripts/agent_context.sh`` (Kontext-Bundle fuer Prompts erzeugen)
- ``.github/pull_request_template.md`` (PR-Checklist inkl. Agent-Checks)
