"""Wie jedes Artefakt des Evidenzpakets im Kapitel zu verwenden ist.

Vier getrennte Angaben je Artefakt, bewusst nicht zusammengezogen:

``shows``
    Was physisch auf der Seite steht — Zeilen, Achsen, Einheiten, Stichprobe.
``says``
    Die Aussage, die es trägt, und wie stark.
``use_for``
    Wo es in der Argumentation hingehört.
``not_for``
    Die Fehllesart, die es einlädt. **Dieses Feld ist der Grund für die Datei.**
    Eine Bildunterschrift sagt, was zu sehen ist, aber nicht, was das Artefakt
    nicht tragen kann — und genau dort geht ein Ergebniskapitel schief.

Getrennt von :mod:`build_pack`, weil das Register Fließtext ist und der Builder
Messcode. Zusammen in einer Datei hätte das eine das andere unlesbar gemacht.

Die Schlüssel sind Artefakt-*Stems* ohne Endung, also ``table_5_4_...`` für
``table_5_4_....csv`` und ``.md``. Ein Stem ohne Eintrag taucht in der
Abdeckungsliste am Ende jeder ``usage_guide.md`` auf — Lücken sind sichtbar,
nicht stillschweigend.
"""

from __future__ import annotations

USAGE: dict[str, dict[str, str]] = {

    # ---------------------------------------------- Gemeinsames Protokoll (01_)
    "dataset_split_register": dict(
        shows="Je Datensatzversion Pfad und Größe, Beispiel-, Trainings- und "
              "Validierungszahlen, die Kontextform, die Kernlänge, die Abtastrate, die "
              "Herkunft des Clean-Signals, ob Spikes injiziert wurden, die Splitregel und "
              "ein Hash der Referenzarrays.",
        says="Jede im Kapitel genannte Datensatzversion ist eindeutig identifiziert und über "
             "den Hash wiedererkennbar.",
        use_for="Der gemeinsame Protokollteil vor 5.4. Bei jeder Versionsnennung im Text auf "
                "diese Zeile verweisen, statt die Version zu beschreiben.",
        not_for="Als Ergebnis. Und die Beispielzahlen nicht als Stichprobengröße für "
                "Spike-Tests lesen — dafür zählen die Ereignisse."),
    "model_identity_register": dict(
        shows="Je verglichenem Arm die Beschreibung, das Auswertungsverzeichnis, der "
              "Datensatz, der Checkpoint mit Hash und ob er lokal vorliegt, die "
              "Modellfabrik, die Kanalzahl, ob im Residualmodus ausgewertet wurde, die "
              "Formel des Arms sowie die Zahl der Validierungs- und Spike-Beispiele.",
        says="Jede Zahl im Kapitel lässt sich auf einen benannten Checkpoint und eine "
             "explizite Armformel zurückführen.",
        use_for="Die Nachvollziehbarkeitszusage des Kapitels. Die Spalte mit der Armformel ist "
                "der Ort, an dem Kaskade und direkte Formulierung unterscheidbar werden.",
        not_for="Als Ergebnis oder als Rangliste."),
    "hardware_runtime_register": dict(
        shows="Je Lauf der Inferenzhost, die Messdefinition, eine Anmerkung und der "
              "Datensatz.",
        says="Auf welcher Hardware die Spike-Evaluationen liefen — ausdrücklich als Kontext, "
             "nicht als Kostenvergleich.",
        use_for="Fußnote zum Protokollteil.",
        not_for="Ein Kostenvergleich der Modelle. Dafür ist table_5_2c da, das eigens unter "
                "einem festen Protokoll auf einer Maschine gemessen wurde."),

    # ----------------------------------------------------------- 5.1 Refactoring
    "table_5_1_engineering_indicators": dict(
        claims="R5.1.1-C1, R5.1.1-C2, R5.1.1-C3",
        shows="26 vorab definierte Indikatoren in drei Spalten: FACETpy 0.1.0 (Branch "
              "`bachelor`), 2.0.0 im vollen Umfang und 2.0.0 beschränkt auf den klassischen "
              "Kern. Je Zeile die bevorzugte Richtung und wofür der Indikator Beleg ist.",
        says="Das Refactoring hat Typisierung, Dokumentation und Prüfdichte deutlich "
             "verbessert und zwei Fehlerklassen beseitigt; bei Funktionsgröße und "
             "Komplexität ist es schlechter geworden. Ein gemischtes Ergebnis, kein Erfolg "
             "auf ganzer Linie.",
        use_for="Die Kernaussage von 5.1.1. Im Text die größenbereinigten Zeilen zitieren "
                "(Anteile, Dichten) und die Verschlechterungen mitnennen — sie machen die "
                "Tabelle glaubwürdig.",
        not_for="Die absoluten Zählungen der mittleren Spalte als Qualitätsvergleich. 2.0.0 "
                "enthält zwölf Modellpakete, die 0.1.0 nicht hatte; ein Vergleich der "
                "Codezeilen misst Funktionsumfang, nicht Güte."),
    "table_5_1b_indicator_scope": dict(
        shows="Der Zählbereich als Text: was gezählt wird, was ausgeschlossen ist, wie eine "
              "Codezeile definiert ist und warum der Bereich vor der Messung festgelegt wurde.",
        says="Die Indikatoren sind nicht nachträglich passend gewählt.",
        use_for="Als Methodenabsatz vor Tabelle 5.1 oder als Fußnote dazu.",
        not_for="Als eigenständiges Ergebnis — die Tabelle enthält keine Messwerte."),
    "table_5_1c_api_walkthrough": dict(
        claims="R5.1.1-C1",
        shows="Acht fachliche Schritte derselben Korrektur, links der 0.1.0-Aufruf, rechts "
              "die v2-Prozessoren, dazu je eine Bemerkung zum Unterschied.",
        says="Drei Unterschiede sind mehr als Geschmack: `pre_processing()` führt zwei "
             "ungenannte Schritte aus, `find_triggers` muss nach dem Upsampling erneut "
             "laufen, und Schätzen und Subtrahieren sind über verstecktem Zustand getrennt.",
        use_for="Der qualitative Teil von 5.1.1, der die Zahlen der Tabelle 5.1 erklärt. "
                "Die drei Bemerkungen sind zitierfähige Beispiele für die Fehleranfälligkeit "
                "der alten API.",
        not_for="Ein Beleg für bessere Laufzeit oder Korrektheit — dafür ist 5.1.2 zuständig."),
    "figure_5_1_engineering_indicators": dict(
        claims="R5.1.1-C2, R5.1.1-C3",
        shows="Balkenpaare für diejenigen Indikatoren, deren Wert **nicht** am Umfang hängt: "
              "Anteile, Dichten und mittlere Längen, 0.1.0 gegen den klassischen Kern.",
        says="Dasselbe wie Tabelle 5.1, aber ohne die Möglichkeit, absolute Zählungen "
             "misszuverstehen.",
        use_for="Die Abbildung zu 5.1.1. Sie ersetzt die Tabelle im Fließtext; die Tabelle "
                "gehört in den Anhang.",
        not_for="Vollständigkeit — die umfangsabhängigen Indikatoren fehlen hier bewusst und "
                "stehen nur in der Tabelle."),
    "table_5_2_verification_runtime_memory": dict(
        claims="R5.1.2-C1, R5.1.2-C2",
        shows="Drei Arme (0.1.0, 2.0.0 mit identischer Parametrierung, 2.0.0 mit "
              "Standardparametern) mit Laufzeit über mehrere Wiederholungen, Streuung, "
              "Spitzenspeicher — plus die Definitionen, wie Zeit und Speicher gemessen wurden.",
        says="Die neue Implementierung liefert dasselbe Ergebnis; Laufzeit und Speicher sind "
             "der Preis der Modularisierung und in der Tabelle beziffert.",
        use_for="5.1.2, direkt neben Tabelle 5.2b. Die Definitionsspalten mitzitieren — ohne "
                "sie ist eine Speicherzahl bedeutungslos.",
        not_for="Ein Vergleich mit anderen Toolboxen: gemessen wurde eine Maschine, ein "
                "Datensatz, eine Kette."),
    "table_5_2b_parity_per_channel": dict(
        claims="R5.1.2-C1",
        shows="Je EEG-Kanal der RMS beider Implementierungen, die Differenz absolut und "
              "relativ zum unkorrigierten Signal, die Pearson-Korrelation und ob der Kanal "
              "innerhalb der vorab festgelegten Toleranz liegt.",
        says="Äquivalenz ist kanalweise geprüft, nicht über einen Mittelwert behauptet.",
        use_for="Der Nachweis, dass 5.1 überhaupt einen Vergleich anstellen darf. Die "
                "Toleranz stand vor der Messung fest — das gehört in den Satz.",
        not_for="Eine Aussage über Korrektheit gegen Grundwahrheit. Geprüft ist "
                "Übereinstimmung zweier Implementierungen, nicht deren Richtigkeit."),
    "figure_5_2_runtime_memory": dict(
        claims="R5.1.2-C2",
        shows="Laufzeit und Spitzenspeicher je Arm als Balken, die Streuung der "
              "Wiederholungen als Fehlerbalken.",
        says="Der Mehraufwand der v2-Architektur ist sichtbar und klein gegenüber der "
             "Streuung zwischen Läufen.",
        use_for="Die Kostenseite von 5.1.2 in einem Bild.",
        not_for="Skalierungsaussagen — gemessen ist genau eine Aufnahmelänge."),

    # ------------------------------------------- 5.2 Explorativer Modellvergleich
    "table_5_3_aas_baseline_unified_holdout": dict(
        claims="R5.2.1-C1",
        shows="Eine Zeile Protokoll: Datensatz, Split, Zahl der Holdout-Fenster und "
              "Kanalfenster, wie viele Modelle Qualitätskennzahlen haben, welche warum "
              "ausgeschlossen sind, welche Baselines mitlaufen und welcher Vergleich "
              "überhaupt zulässig ist.",
        says="Alle Modelle wurden auf demselben Holdout mit derselben Zieldefinition "
             "bewertet — die Voraussetzung dafür, dass 5.2 mehr ist als eine Sammlung "
             "einzelner Trainingsberichte.",
        use_for="Der einleitende Methodenabsatz von 5.2. Die Zeile `allowed_comparison` "
                "wörtlich übernehmen.",
        not_for="Als Ergebnis. Hier steht nur, was verglichen werden darf."),
    "table_5_3b_holdout_exclusions": dict(
        shows="Je ausgeschlossenem Modell der Grund, welche Kennzahlen vorliegen und wo.",
        says="Die Ausschlüsse sind benannt und begründet, nicht weggelassen.",
        use_for="Fußnote zur Rangliste. Ein Ranking ohne sichtbare Ausschlussliste ist "
                "nicht prüfbar.",
        not_for="Eine Qualitätsaussage über die ausgeschlossenen Modelle."),
    "table_5_3c_holdout_split_verification": dict(
        claims="R5.2.1-C1",
        shows="Der aus Seed 42 neu berechnete Split-Hash gegen die in den Modellläufen "
              "gespeicherten Hashes, plus wie Läufe ohne gespeicherten Hash abgedeckt sind.",
        says="Alle Modelle sahen denselben Holdout. Das ist nachgerechnet, nicht angenommen.",
        use_for="Ein Satz im Methodenteil von 5.2 mit der Zahl der übereinstimmenden Läufe.",
        not_for="Ein Beleg gegen Selektionsoptimismus — dafür ist der gesperrte Holdout in "
                "5.6 zuständig (table_5_17b)."),
    "table_5_4_family_level_ranking": dict(
        claims="R5.2.2-C1",
        shows="Je Architekturfamilie Median, Minimum und Maximum der SNR-Verbesserung über "
              "die Modelle der Familie, dazu Artefaktkorrelation und Rest-RMS-Verhältnis, "
              "die Zahl der Holdout-Fenster und die verwendete Rangregel.",
        says="Auf Familienebene trennen sich die Ansätze deutlich; innerhalb einer Familie "
             "ist die Spanne oft größer als der Abstand zur Nachbarfamilie.",
        use_for="Der Einstieg in 5.2.2 — erst Familien, dann Einzelmodelle.",
        not_for="Eine Rangfolge einzelner Modelle. Familien mit einem einzigen Modell haben "
                "keine Spanne und sind deshalb nicht gleichwertig gereiht."),
    "figure_5_4_family_level_ranking": dict(
        claims="R5.2.2-C1",
        shows="Waagerechte Ranglistenbalken je Familie, mit der Stichprobengröße neben "
              "jedem Balken.",
        says="Dasselbe wie Tabelle 5.4, mit der Stichprobe im Bild statt in einer Fußnote.",
        use_for="Die Übersichtsabbildung von 5.2.2.",
        not_for="Signifikanzaussagen — die Balken tragen keine Intervalle."),
    "table_5_5_leading_architectures": dict(
        claims="R5.2.2-C2",
        shows="Die führenden Einzelmodelle mit Checkpoint, SNR-Verbesserung, Clean-SNR nach "
              "der Korrektur, Artefaktkorrelation, Rest-RMS-Verhältnis, Inferenzzeit — plus "
              "je Zeile die Auswahlregel, die Unsicherheit und die Einschränkung.",
        says="Welche Architekturen vorn liegen, und zu welchem Preis.",
        use_for="Die Detailtabelle von 5.2.2. Die Spalte `limitation` gehört in den Text, "
                "nicht in eine Fußnote.",
        not_for="Eine Aussage, dass der Erste besser ist als der Zweite — dafür ist "
                "table_5_5c da."),
    "figure_5_5_leading_architectures": dict(
        claims="R5.2.2-C2",
        shows="Ranglistenbalken der führenden Modelle mit sichtbarer Stichprobengröße.",
        says="Die Reihenfolge der Spitzenmodelle.",
        use_for="Neben Tabelle 5.5.",
        not_for="Unterschiede zwischen benachbarten Rängen — dazu fehlen Intervalle."),
    "table_5_5b_holdout_intervals": dict(
        claims="R5.2.2-C3",
        shows="Je Modell Mittelwert und Median der SNR-Verbesserung über die Holdout-Fenster, "
              "Standardabweichung und Bootstrap-Konfidenzintervall, daneben der gespeicherte "
              "Punktschätzer des ursprünglichen Laufs.",
        says="Die Punktschätzer der Trainingsberichte sind reproduzierbar, und die "
             "Rangunterschiede sind gegen die Streuung zwischen Fenstern zu lesen.",
        use_for="Der Absatz in 5.2.2, der die Rangliste relativiert.",
        not_for="Paarweise Vergleiche. Überlappende Intervalle sind kein Test — dafür "
                "table_5_5c."),
    "figure_5_5b_holdout_intervals": dict(
        claims="R5.2.2-C3",
        shows="Forest-Plot: je Modell der Effekt mit Konfidenzintervall.",
        says="Wie groß die Unsicherheit je Modell gegenüber dem Rangabstand ist.",
        use_for="Die Abbildung zur Unsicherheit in 5.2.2.",
        not_for="Signifikanz zwischen zwei Modellen aus überlappenden Intervallen ableiten."),
    "table_5_5c_holdout_paired_vs_leader": dict(
        claims="R5.2.2-C3",
        shows="Gepaarte Differenz jedes Modells gegen das Spitzenmodell auf denselben "
              "Fenstern, mit Konfidenzintervall, Holm-korrigiertem p und Signifikanzflag.",
        says="Welche Rangunterschiede den Test überstehen und welche nicht.",
        use_for="Der einzige Ort, an dem 5.2 sagen darf, ein Modell sei besser als ein "
                "anderes.",
        not_for="Aussagen über Modelle, die nicht in der Tabelle stehen."),
    "table_5_6_training_runtime_memory": dict(
        shows="Je Modell Laufkennung, Epochenzahl, beste Epoche und Metrik, Abschlussstatus, "
              "Trainingszeit, Hardware, Präzision, Spitzenspeicher und die Messdefinition.",
        says="Die Trainingsläufe sind unterschiedlich weit gelaufen und auf "
             "unterschiedlicher Hardware — Trainingskosten sind zwischen Modellen nicht "
             "direkt vergleichbar.",
        use_for="Die Fairness-Diskussion in 5.3.3 und die Einordnung der Trainingskurven.",
        not_for="Ein Kostenvergleich der Architekturen. Für vergleichbare Kosten ist "
                "table_5_2c da, das auf einer Maschine gemessen wurde."),
    "figure_5_6_training_dynamics": dict(
        shows="Verlustkurven auf einer einheitlichen Achsendefinition; unvollständige Läufe "
              "sind als solche markiert.",
        says="Manche Läufe waren nicht auskonvergiert — das begrenzt, was aus dem "
             "Endergebnis über die Architektur folgt.",
        use_for="Die Fairness-Einschränkung in 5.3.3 belegen.",
        not_for="Architekturvergleich anhand der Kurvenhöhe: die Verlustfunktionen sind "
                "nicht überall dieselbe Größe."),
    "table_5_2c_inference_cost": dict(
        claims="R5.2.4-C1",
        shows="Je Modell Inferenzzeit und Spitzenspeicher, auf einem Gerät, mit "
              "Wiederholungen, Streuung und Millisekunden pro Fenster, plus die Definitionen "
              "von Zeit, Speicher und Präzision.",
        says="Die Inferenzkosten unterscheiden sich um Größenordnungen — und zwar unter "
             "identischen Bedingungen gemessen, anders als die Trainingskosten.",
        use_for="Die Kostenachse von 5.2.4 und die Pareto-Abbildung.",
        not_for="Kosten auf Zielhardware. Gemessen ist eine Maschine in einer Präzision."),
    "table_5_2d_cost_protocol": dict(
        shows="Das Messprotokoll als Schlüssel-Wert-Liste: Wiederholungen, Warmlauf, "
              "Prozessisolierung, Zeitlimit, Fensterzahl.",
        says="Die Kostenzahlen sind unter einem festgelegten Protokoll entstanden.",
        use_for="Methodenfußnote zu Tabelle 5.2c.",
        not_for="Als Ergebnis."),
    "table_5_2e_quality_cost": dict(
        claims="R5.2.4-C2",
        shows="Je Modell die SNR-Verbesserung neben Millisekunden pro Fenster und "
              "Spitzenspeicher — Qualität und Kosten in einer Zeile.",
        says="Die teuersten Modelle sind nicht die besten; es gibt eine nutzbare "
             "Pareto-Front.",
        use_for="Die Datengrundlage der Pareto-Abbildung; im Text die dominierten Modelle "
                "benennen.",
        not_for="Eine Empfehlung. Welcher Punkt der Front richtig ist, hängt vom "
                "Einsatzfall ab, und den entscheidet die Arbeit nicht."),
    "figure_5_6b_quality_cost_pareto": dict(
        claims="R5.2.4-C2",
        shows="Qualität gegen Kosten als Streudiagramm mit angegebener Vorzugsrichtung.",
        says="Welche Modelle auf der Pareto-Front liegen und welche dominiert werden.",
        use_for="Die Abbildung zu 5.2.4.",
        not_for="Extrapolation auf andere Hardware oder Batchgrößen."),

    # -------------------------------- 5.3 Architekturtreue und Eingangsvertrag
    "table_5_7_architecture_fidelity": dict(
        claims="R5.3.1-C1, R5.3.1-C2",
        shows="Das vollständige Treueregister: je Paper-Anforderung der geforderte Aspekt, "
              "der Zustand der Originalimplementierung, der Status der "
              "Paper-Accurate-Edition, Schweregrad und Disposition — und dazu die geprüften "
              "Codesymbole: welche im Paket gefunden wurden, welche nicht, welche von Hand "
              "verifiziert sind, plus Testdateien und Zahl der Testfunktionen.",
        says="Die Treuebehauptungen der Modellpakete sind gegen den tatsächlichen Code "
             "geprüft und nicht nur aus deren eigener Dokumentation übernommen.",
        use_for="Der Kern von 5.3.1. Im Text die Symbolprüfung erwähnen — sie ist der "
                "Unterschied zwischen einem Register und einer Selbstauskunft.",
        not_for="Eine Aussage, dass ein Paket das Paper korrekt *implementiert*. Geprüft ist, "
                "ob die genannten Symbole existieren, nicht ob sie das Richtige tun."),
    "table_5_7b_fidelity_per_model": dict(
        claims="R5.3.1-C2",
        shows="Je Modellpaket die Zahl der Anforderungen, der symbolverifizierten, der nicht "
              "gefundenen und der von Hand verifizierten Symbole, die Zahl der Testfunktionen "
              "und der Parsestatus der Registertabelle.",
        says="Die Abdeckung der Treueprüfung ist je Paket unterschiedlich, und das ist "
             "beziffert.",
        use_for="Die Übersichtszeile von 5.3.1; Pakete mit vielen nicht gefundenen Symbolen "
                "im Text benennen.",
        not_for="Eine Rangliste der Modellqualität."),
    "figure_5_7_fidelity_register": dict(
        claims="R5.3.1-C1",
        shows="Gestapelte Balken: Anforderungen je Modellpaket, aufgeteilt nach Disposition.",
        says="Wie vollständig die Pakete ihre Paper-Anforderungen umsetzen.",
        use_for="Die Abbildung zu 5.3.1.",
        not_for="Die Symbolprüfung — die steht nur in der Tabelle."),
    "table_5_8_effective_context": dict(
        claims="R5.3.2-C1",
        shows="Je Modell die nominelle Eingangsform gegen den tatsächlich erreichten Kontext: "
              "gelieferte Epochen und Kanäle, wie viele Kanäle ein Ausgabesample überhaupt "
              "erreichen, der Gradientenanteil der Mittelepoche gegen den der Randepochen, "
              "das rezeptive Feld in Samples, Anteil und Millisekunden, dazu die Parameterzahl.",
        says="Mehrere Architekturen bekommen einen Kontext geliefert, den sie rechnerisch "
             "nicht nutzen können — der Eingangsvertrag ist nicht dasselbe wie die "
             "tatsächliche Reichweite.",
        use_for="Der Kern von 5.3.2 und die wichtigste Einschränkung des Modellvergleichs: "
                "Kontextmodelle sind teilweise gar keine.",
        not_for="Eine Erklärung der Leistungsunterschiede. Der Zusammenhang zwischen "
                "rezeptivem Feld und Ergebnis ist hier nicht getestet."),
    "figure_5_9_context_utilization": dict(
        claims="R5.3.2-C1",
        shows="Je Edition der Gradientenanteil pro Epoche und der Anteil des rezeptiven "
              "Feldes am Fenster.",
        says="Wie viel des gelieferten Kontexts ein Modell erreicht.",
        use_for="Die Abbildung zu 5.3.2.",
        not_for="Kausale Aussagen über die Ergebnisqualität."),
    "table_5_9_fairness_limitations": dict(
        shows="Je Fairnessachse die betroffenen Modelle, der Beleg, der Schweregrad, die "
              "Auswirkung auf die Schlussfolgerung und die Abhilfe.",
        says="Der Modellvergleich in 5.2 ist in benannten Punkten unfair, und jede "
             "Unfairness hat eine bezifferte Auswirkung.",
        use_for="Der Einschränkungsabsatz zu 5.2 und 5.3. Diese Tabelle vorwegnehmen, statt "
                "sie in die Diskussion zu schieben.",
        not_for="Ein Grund, die Ergebnisse zu verwerfen — die Spalte `impact` sagt, wie weit "
                "die jeweilige Unfairness trägt."),

    # --------------------------------------- 5.4 Ausfallmodi und Zielfunktionen
    "table_5_10_model_failure_modes": dict(
        claims="R5.4-C1",
        shows="Je Ausfallmodus die Definition, die Erkennungsregel, die betroffenen Modelle, "
              "die Abdeckung, das Evidenzniveau und die Quelle.",
        says="Die beobachteten Fehlschläge sind nicht Einzelfälle, sondern lassen sich auf "
             "wenige Modi mit prüfbaren Erkennungsregeln zurückführen.",
        use_for="Die Gliederung von 5.4. Die Erkennungsregel je Modus im Text nennen — sie "
                "macht den Modus falsifizierbar.",
        not_for="Vollständigkeit. Die Tabelle listet die Modi, die gemessen wurden, nicht "
                "alle möglichen."),
    "table_5_10d_failure_mode_coverage": dict(
        shows="Dieselben Modi mit der Zahl der betroffenen Modelle und der Grundlage, auf der "
              "die Häufigkeit bestimmt wurde.",
        says="Wie breit jeder Modus belegt ist — ein Modus mit einem einzigen Fall ist als "
             "solcher erkennbar.",
        use_for="Fußnote zu Tabelle 5.10; im Text die dünn belegten Modi entsprechend "
                "vorsichtig formulieren.",
        not_for="Häufigkeitsaussagen über Architekturen allgemein."),
    "figure_5_10_failure_mode_examples": dict(
        claims="R5.4-C1",
        shows="Beispielspuren, eine Zeile je Beispiel, eine Spalte je Arm, gemeinsame "
              "y-Grenzen.",
        says="Wie ein Ausfall im Signal aussieht — die Modi sind sichtbar, nicht nur "
             "kennzahlig.",
        use_for="Die Abbildung, die 5.4 eröffnet.",
        not_for="Repräsentativität. Es sind ausgewählte Beispiele; die Häufigkeit steht in "
                "table_5_10d."),
    "table_5_10b_positional_sensitivity": dict(
        claims="R5.4.1-C1, R5.4.1-C2",
        shows="Je Lauf der Fensterversatz und die Trigger-Fehlausrichtung in Samples und "
              "Millisekunden, der Modell-RMSE, der RMSE eines idealen FARM-Templates und der "
              "der Nullausgabe, ob das Modell schlechter als die Nullausgabe ist, sowie die "
              "Spike-Morphologiekorrelation.",
        says="Zwei verschiedene Dinge: ein reiner Fensterversatz kostet nichts (das Modell "
             "ist positionsinvariant), eine Trigger-Fehlausrichtung kostet massiv — und zwar "
             "**jedes** templatebasierte Verfahren gleichermaßen.",
        use_for="R5.4.1: Ausrichtung ist eine Voraussetzung der Templatesubtraktion, keine "
                "Verfeinerung. Die Invarianz gegen Fensterversatz ist das eigentliche "
                "Ergebnis und gehört zuerst genannt.",
        not_for="Die Behauptung, direkte Modelle seien gegen Fehlausrichtung immun und die "
                "Kaskade nicht. Ein verschobenes Template ist bei jedem Verfahren falsch; "
                "die Zeilen mit Fehlausrichtung zeigen den Preis der Voraussetzung, nicht "
                "einen Nachteil der Kaskade."),
    "figure_5_10b_positional_sensitivity": dict(
        claims="R5.4.1-C1, R5.4.1-C2",
        shows="Zwei Felder: links der harmlose Fensterversatz, rechts die schädliche "
              "Fehlausrichtung.",
        says="Die beiden Positionsstörungen sind qualitativ verschieden — das ist der ganze "
             "Punkt der Abbildung.",
        use_for="Die Abbildung zu 5.4.1. Beide Felder gemeinsam zitieren; einzeln ist jedes "
                "irreführend.",
        not_for="Übertragung auf reale Triggerfehler ohne table_5_10e und table_5_10f — die "
                "hier gezeigte Fehlausrichtung ist konstruiert."),
    "table_5_10c_fm5_paired": dict(
        shows="Je Bedingung der Versatz in Samples und Millisekunden, die Epochenzahl, der "
              "Median-RMSE, die Hodges-Lehmann-Differenz gegen den ausgerichteten Fall mit "
              "Konfidenzintervall, Holm-korrigiertes p und Signifikanz.",
        says="Der Effekt der Fehlausrichtung ist gepaart getestet, nicht aus zwei Mittelwerten "
             "abgeleitet.",
        use_for="Der statistische Beleg hinter table_5_10b.",
        not_for="Als eigenständiges Ergebnis ohne die Einordnung aus 5.4.1 — die Bedingungen "
                "sind konstruiert."),
    "table_5_10e_estimation_vs_application_jitter": dict(
        claims="R5.4.1-C4",
        shows="Zwei Arme mit erklärter Bedeutung: Jitter bei der **Schätzung** (jede Epoche "
              "einzeln verschoben) gegen Jitter bei der **Anwendung** (ein gemeinsamer "
              "Versatz). Je Stufe Fehler, HL-Differenz mit Intervall, ob das Ergebnis "
              "schlechter als die Nullausgabe ist, plus RMS-Verhältnis und Korrelation des "
              "resultierenden Templates zum ausgerichteten.",
        says="Die beiden Ausfälle sind mechanisch verschieden: Schätzjitter **verwischt** das "
             "Template (RMS sinkt, der Fehler sättigt bei „keine Korrektur“), ein "
             "Anwendungsversatz lässt das Template intakt (RMS-Verhältnis 1,0) und fügt "
             "Artefaktenergie hinzu, kann also über „keine Korrektur“ hinaus schaden.",
        use_for="Der Kern von R5.4.1-C4. Die Template-RMS-Spalte ist der Beleg für die "
                "Unterscheidung und gehört in den Text.",
        not_for="Die Aussage, der Anwendungsfall trete real auf. Ein globaler Versatz "
                "verschiebt Mittelung und Subtraktion gemeinsam und ist deshalb aus einem "
                "Triggerfehler heraus gar nicht erreichbar."),
    "table_5_10f_pipeline_trigger_jitter": dict(
        claims="R5.4.1-C5",
        shows="Derselbe Jitter in der ausgelieferten Kette: je Arm, "
              "Realign-Einstellung und Jitterstärke (in ms und nativen Samples) der Rest-RMS, "
              "der entfernte Leistungsanteil und die spektralen Anteile ober- und innerhalb "
              "des EEG-Bands.",
        says="Mit FARMs 30-Epochen-Fenster ist der realistische Ausfall klein: eine "
             "Verschiebung um ein natives Sample kostet Bruchteile eines Prozentpunkts "
             "entfernter Leistung. Der kontrollierte Sweep überschätzt den Effekt, weil er "
             "über sechs Nachbarepochen mittelt.",
        use_for="Die Einordnung von 5.4.1 in die Praxis — der Satz, der die konstruierten "
                "Zahlen wieder erdet.",
        not_for="Eine Genauigkeitsaussage. Auf einer echten Aufnahme gibt es kein sauberes "
                "Referenzsignal; die Spalten sagen, **wie viel** entfernt wurde, nicht ob das "
                "Richtige."),
    "table_5_11_target_degeneracy_signal_deletion": dict(
        claims="R5.4.2-C1",
        shows="Je Modell die gepaarte Gegenüberstellung mit der Nullausgabe: Median-RMSE "
              "beider, HL-Differenz mit Intervall und Holm-p, dasselbe für den SNR, und ein "
              "Flag, ob das Modell schlechter als die Nullausgabe ist.",
        says="Mehrere Modelle sind nicht besser als ein Korrektor, der konstant Null "
             "ausgibt — sie löschen das Signal, statt das Artefakt zu entfernen.",
        use_for="Das schärfste Ergebnis von 5.4. Die Nullausgabe als Pflichtarm im Text "
                "begründen: das Artefakt ist ein Vielfaches des EEG, deshalb ist „gib Null "
                "aus“ eine starke Baseline.",
        not_for="Ein Beleg, dass die betroffenen Architekturen grundsätzlich untauglich sind "
                "— getestet ist eine Zielfunktion auf einem Datensatz."),
    "figure_5_11_signal_deletion": dict(
        claims="R5.4.2-C1",
        shows="Roh, Clean, Ziel, Vorhersage und rekonstruiertes Clean untereinander, alle bei "
              "**unveränderter** Skala.",
        says="Die Signallöschung ist direkt sichtbar: die Vorhersage folgt dem Artefakt, das "
             "rekonstruierte Clean ist nahezu flach.",
        use_for="Die Abbildung zu 5.4.2. Die unveränderte Skala im Text erwähnen — eine "
                "automatisch skalierte Achse würde genau diesen Befund verstecken.",
        not_for="Repräsentativität einzelner Epochen."),
    "figure_5_11b_rmse_vs_null": dict(
        claims="R5.4.2-C1",
        shows="Forest-Plot der RMSE-Differenzen gegen die Nullausgabe, je Modell eine Zeile "
              "mit Intervall.",
        says="Welche Modelle die Nullausgabe schlagen und welche nicht.",
        use_for="Die kompakte Form von Tabelle 5.11.",
        not_for="Vergleiche der Modelle untereinander — die Referenz ist überall die "
                "Nullausgabe, nicht das jeweils andere Modell."),
    "table_5_12_recovered_clean_objective": dict(
        claims="R5.4.3-C1",
        shows="Je Ebene (Bulk/Spike) und Kennzahl die Einheit, die Vorzugsrichtung, die Zahl "
              "unabhängiger Einheiten, die Mediane, die HL-Differenz mit Intervall und "
              "Holm-p, ob der Test überhaupt aussagekräftig ist und worauf das kontrollierte "
              "Paar beruht.",
        says="Die Zielfunktion `recovered_clean` verhindert die Signallöschung, die eine reine "
             "MSE auf das Artefakt erzeugt.",
        use_for="Der Kern von 5.4.3.",
        not_for="Als Wirksamkeitsnachweis ohne das kontrollierte Paar — erst table_5_12b "
                "isoliert die Zielfunktion als Ursache."),
    "table_5_12b_controlled_objective_pair": dict(
        claims="R5.4.3-C2",
        shows="Zwei Trainingsläufe, die sich in genau einer Konfigurationszeile "
              "unterscheiden: die Zielfunktion. Je Lauf Median-RMSE, HL gegen die Nullausgabe "
              "mit Intervall und p, ob EEG rekonstruiert wird, der Clean-SNR gegen die "
              "Nullausgabe und die Trainingskennzahlen.",
        says="Die Zielfunktion ist die Ursache, nicht ein Begleitumstand: derselbe Datensatz, "
             "dasselbe Modell, dieselbe Dauer — nur die Zielfunktion unterscheidet sich, und "
             "der reine MSE-Lauf ist schlechter als gar nichts zu tun.",
        use_for="Das stärkste kausale Argument in 5.4. Wörtlich als kontrolliertes Paar "
                "bezeichnen.",
        not_for="Übertragung auf andere Architekturen ohne den direkten Gegenpart "
                "(table_5_12c)."),
    "table_5_12c_objective_pair_direct": dict(
        claims="R5.4.3-C2",
        shows="Dasselbe kontrollierte Paar für die direkte Formulierung, je Kennzahl HL, "
              "Intervall, Holm-p und Signifikanz.",
        says="Der Effekt der Zielfunktion ist nicht an die Kaskadenformulierung gebunden.",
        use_for="Der Generalisierungssatz zu 5.4.3.",
        not_for="Eine Aussage über die Rangfolge direkter gegen kaskadierter Formulierung."),
    "figure_5_12_objective_comparison": dict(
        claims="R5.4.3-C1, R5.4.3-C2",
        shows="Gittereffekte der Zielfunktionsvarianten, mit der Nullausgabe und FARM als "
              "eingezeichneten Bezugslinien.",
        says="Wo die Varianten gegenüber den beiden Bezugsgrößen liegen.",
        use_for="Die Abbildung zu 5.4.3.",
        not_for="Signifikanz — die Balken tragen keine Tests."),

    # ---------------------------------------- 5.5 Spike-Erhalt gegen FARM
    "table_5_13_spike_dataset_models": dict(
        shows="Je Modell der Datensatz und Split, die Zahl der Validierungsbeispiele, der "
              "Spike-Beispiele und der **unabhängigen Spike-Ereignisse**, die Art der "
              "Annotation, Eingangsvertrag und Eingangssignal, Checkpoint, Artefaktquelle "
              "sowie ob das Modell vergleichsberechtigt ist und warum.",
        says="Welche Modelle auf welcher Datengrundlage überhaupt gegeneinander antreten "
             "dürfen.",
        use_for="Der Methodenabsatz von 5.5. Die Spalte mit den unabhängigen Ereignissen ist "
                "die wichtigste — sie ist die Stichprobengröße aller Spike-Tests.",
        not_for="Die Beispielzahl als Stichprobengröße lesen. Der Builder schreibt ein "
                "Beispiel je Zielelektrode, also ist die unabhängige Einheit das Ereignis, "
                "nicht das Beispiel."),
    "table_5_13b_spike_inventory": dict(
        shows="Das Spike-Inventar als Größen-Wert-Liste mit Anmerkungen.",
        says="Wie viele Spikes es gibt, wie sie verteilt sind und was daraus für die "
             "Teststärke folgt.",
        use_for="Fußnote zu Tabelle 5.13 und Begründung der Datensatzversionen in 5.6.",
        not_for="Als Ergebnis."),
    "figure_5_13b_clean_and_artifact": dict(
        shows="Roh, Artefakt und Clean, jeweils in eigener Skala.",
        says="Das Größenverhältnis: das Artefakt ist ein Vielfaches des EEG.",
        use_for="Der Absatz, der begründet, warum die Nullausgabe ein Pflichtarm ist.",
        not_for="Eine Aussage über die Korrekturqualität."),
    "table_5_14_spike_metric_dictionary": dict(
        shows="Je Kennzahl Bezeichnung, Einheit, Vorzugsrichtung, Ebene, die unabhängige "
              "Einheit, n, Parameter, ob sie primär ist, der bekannte Randfall, die "
              "Implementierung und der Testbeleg.",
        says="Die Spike-Kennzahlen sind definiert, parametrisiert und getestet, bevor sie "
              "verwendet werden.",
        use_for="Der Methodenteil von 5.5 und die Referenz für jede Kennzahlnennung im Text.",
        not_for="Als Ergebnis. Die Spalte `edge_case` ist trotzdem zitierpflichtig, wo die "
                "betroffene Kennzahl berichtet wird."),
    "table_5_14b_closed_form_validation": dict(
        shows="Je Testfunktion die Zahl parametrisierter Fälle, die geprüfte Eigenschaft, "
              "Datei und Zeile.",
        says="Die Kennzahlen sind gegen von Hand ableitbare Erwartungswerte geprüft, nicht "
             "nur gegen sich selbst.",
        use_for="Ein Satz im Methodenteil von 5.5, der die Kennzahlen absichert.",
        not_for="Ein Beleg für die Richtigkeit der Ergebnisse — geprüft ist die "
                "Kennzahlimplementierung."),
    "table_5_15_direct_dl_vs_farm": dict(
        claims="R5.5-C1, R5.5-C2",
        shows="Je Ebene, Modell und Kennzahl die Mediane von Modell, idealem FARM und "
              "Nullausgabe, dazu **zwei** gepaarte Vergleiche — gegen FARM und gegen die "
              "Nullausgabe — jeweils mit HL, Intervall, Holm-p und Signifikanz, plus ein "
              "Flag, ob der Test aussagekräftig ist.",
        says="Die direkten Modelle entfernen mehr Artefakt als FARM und schlagen die "
             "Nullausgabe; beim Spike-Erhalt ist das Bild differenzierter.",
        use_for="Die Ergebnistabelle von 5.5. Beide Vergleiche zitieren — gegen FARM allein "
                "wäre die Signallöschung unsichtbar.",
        not_for="Zeilen mit `testable = false` als Ergebnis lesen. Bei kleiner Ereigniszahl "
                "ist das kleinste erreichbare p durch die Zahl der Paare begrenzt."),
    "figure_5_14_direct_dl_vs_farm": dict(
        claims="R5.5-C1",
        shows="Je Kennzahl ein Feld mit jedem Beispiel als verbundenem Paar, daneben der "
              "Effekt.",
        says="Die Vergleiche sind gepaart, und die Streuung der Paare ist sichtbar.",
        use_for="Die Hauptabbildung von 5.5.",
        not_for="Ablesen einzelner Werte — dafür ist die Tabelle da."),
    "figure_5_14b_direct_effects": dict(
        claims="R5.5-C1, R5.5-C2",
        shows="Forest-Plot der Effekte je Modell und Kennzahl, eingefärbt nach "
              "Holm-korrigierter Signifikanz.",
        says="Welche Effekte den Test überstehen.",
        use_for="Die kompakte Ergebnisübersicht von 5.5.",
        not_for="Signifikanz am Intervall ablesen — eingefärbt ist nach p_holm, und das ist "
                "der maßgebliche Wert."),
    "figure_5_15_spike_examples": dict(
        shows="Spike-Beispiele, eine Zeile je Beispiel, eine Spalte je Arm, gemeinsame "
              "y-Grenzen.",
        says="Wie die Verfahren mit einem echten Spike umgehen.",
        use_for="Die qualitative Ergänzung zu den Spike-Kennzahlen.",
        not_for="Repräsentativität."),
    "table_5_15c_metric_sensitivity": dict(
        claims="R5.5.3-C1",
        shows="Derselbe Vergleich unter mehreren Auswertungskonfigurationen: je "
              "Datensatzversion, Nachbarschaftsfenster und Spike-Verbreiterung, je Modell und "
              "Kennzahl die HL-Differenz gegen FARM mit Intervall, Holm-p und Signifikanz.",
        says="Die **Rangfolge** der Formulierungen ist stabil, der **Nullpunkt** nicht: mit "
             "der schmalen Label-Breite liegt die Kaskade unter FARM, mit einer realistischen "
              "Spike-Dauer darüber.",
        use_for="Die Sensitivitätsanalyse von 5.5.3 und die Begründung, warum jede "
                "Morphologieaussage die Label-Breite mitnennen muss.",
        not_for="Sich eine Konfiguration herauszusuchen. Die Tabelle existiert, damit alle "
                "berichtet werden."),
    "figure_5_15c_metric_sensitivity": dict(
        claims="R5.5.3-C1",
        shows="Je Kennzahl ein Feld, gruppierte Balken je Konfiguration, eine Gruppe je Arm.",
        says="Wie stark die Auswertungsparameter das Ergebnis verschieben.",
        use_for="Die Abbildung zu 5.5.3.",
        not_for="Eine einzelne Konfiguration als die richtige darzustellen."),
    "table_5_15d_ordering_stability": dict(
        claims="R5.5.3-C2",
        shows="Je Datensatzversion und Kennzahl die Referenzreihenfolge gegen die Reihenfolge "
              "unter der jeweiligen Konfiguration, plus ob die Vertauschungen "
              "Formulierungsgruppen überschreiten und wie groß der größte betroffene Abstand "
              "war.",
        says="Vertauschungen treten auf, aber innerhalb der Gruppen — die Trennung nach "
             "Eingangsformulierung überlebt jede geprüfte Konfiguration.",
        use_for="Der Satz, der 5.5.3 abschließt und die Kernaussage rettet.",
        not_for="Eine Aussage über einzelne Modellpaare."),

    # ------------------------------- 5.6 FARM-DL-Residualkaskade
    "table_5_16_cascade_ablation_matrix": dict(
        claims="R5.6.1-C1",
        shows="Die vollständige Ablationsmatrix: je Konfiguration FARM-Einstellung, "
              "Gewichte, Lernrate, Kanalzahl, Kontext, Fenster-Jitter, Seed, Laufstatus, die "
              "Gitterkennzahlen, ob die Nullausgabe geschlagen wurde, ob ein Checkpoint "
              "vorliegt, ob die Konfiguration holdout-berechtigt ist und auf welcher "
              "Grundlage sie ausgewählt wurde.",
        says="Die eingesetzte Konfiguration ist aus einem vollständig berichteten Gitter "
             "ausgewählt, nicht nachträglich gefunden.",
        use_for="Der Methodenteil von 5.6.1. Die Auswahlgrundlage im Text nennen.",
        not_for="Die Gitterkennzahlen als Ergebnis. Sie stammen aus dem Auswahlsplit; das "
                "Ergebnis steht in table_5_17."),
    "figure_5_17_cascade_ablation_effects": dict(
        claims="R5.6.1-C1",
        shows="Gittereffekte mit eingezeichneter Nullausgabe und FARM.",
        says="Welche Konfigurationen überhaupt über die Bezugslinien kommen.",
        use_for="Die Abbildung zu 5.6.1.",
        not_for="Signifikanz."),
    "table_5_17_cascade_artifact_spike_results": dict(
        claims="R5.6.2-C1, R5.6.2-C2",
        shows="Je Ebene und Kennzahl Kaskade, ideales FARM und Nullausgabe, die gepaarten "
              "Vergleiche gegen FARM und gegen die Nullausgabe mit HL, Intervall und Holm-p, "
              "ob der Test aussagekräftig ist, sowie die Vergleiche gegen die beiden "
              "stärksten direkten Modelle.",
        says="Die Kaskade schlägt FARM auf beiden Kopfkennzahlen und die Nullausgabe "
             "ebenfalls; gegen die direkten Modelle ist das Bild kennzahlabhängig.",
        use_for="Die Ergebnistabelle von 5.6.",
        not_for="Zeilen mit `testable = false` als Ergebnis. Und: die Morphologiezeile ohne "
                "die Label-Breite aus table_5_18b zu zitieren."),
    "figure_5_14c_cascade_vs_farm": dict(
        claims="R5.6.2-C1",
        shows="Je Kennzahl ein Feld mit jedem Beispiel als verbundenem Paar Kaskade gegen "
              "ideales FARM, daneben der Effekt.",
        says="Der Vorteil der Kaskade ist gepaart und über die Beispiele hinweg konsistent.",
        use_for="Die Hauptabbildung von 5.6.",
        not_for="Ablesen einzelner Werte."),
    "table_5_17b_locked_holdout": dict(
        claims="R5.6.3-C1",
        shows="Der Auswahlsplit gegen den **gesperrten** Split, je Kennzahl mit Epochenzahl, "
              "HL gegen die Nullausgabe mit Intervall und p sowie HL gegen FARM.",
        says="Auf dem nie zur Auswahl benutzten Split ist das Ergebnis nicht schlechter — "
             "die Schlussfolgerung ist kein Selektionsoptimismus.",
        use_for="Der Validitätsabsatz von 5.6.3.",
        not_for="Ein Vergleich der absoluten Zahlen mit table_5_17: die Modelle des "
                "gesperrten Laufs sind auf weniger Beispielen trainiert und deshalb "
                "insgesamt schwächer. Zu vergleichen ist der **Abstand** zwischen den beiden "
                "Splits, nicht das Niveau."),
    "table_5_17c_locked_split_definition": dict(
        shows="Wie der gesperrte Split geschnitten wurde: Kontextepochen, Epochenzahlen vor "
              "und nach der Teilung, die als Schutzband verworfenen Epochen und die "
              "resultierenden Beispielzahlen.",
        says="Der gesperrte Split ist leckfrei geschnitten — nach Mittelepoche, mit "
             "Schutzband.",
        use_for="Methodenfußnote zu table_5_17b.",
        not_for="Als Ergebnis."),
    "table_5_18_remaining_spike_morphology": dict(
        claims="R5.6.4-C1",
        shows="Je Kennzahl und Verfahren die Beschreibung, die Referenz, die Zahl "
              "**unabhängiger Ereignisse** und die Zahl der Kanalfenster, die Mediane, die "
              "Ereignisdifferenz und die HL-Differenz, ob der Test aussagekräftig ist, das "
              "Holm-p — und getrennt davon die deskriptive Fensterdifferenz mit "
              "Gültigkeitsbereich.",
        says="Beim Spike-Erhalt bleibt ein Rest; ob er signifikant ist, hängt an der Zahl der "
             "unabhängigen Ereignisse und an der Label-Breite.",
        use_for="Der ehrliche Abschluss von 5.6. Ereignis- und Fensterspalten getrennt "
                "zitieren.",
        not_for="Die Fensterdifferenz als Test verwenden — sie ist ausdrücklich deskriptiv, "
                "weil Kanalfenster desselben Ereignisses nicht unabhängig sind."),
    "table_5_18b_label_width": dict(
        claims="R5.6.4-C2",
        shows="Je Datensatzversion, bewerteter Spike-Ausdehnung und Modell die "
              "Morphologiedifferenz gegen FARM mit Intervall, Holm-p, Signifikanz und ein "
              "Flag, ob das Modell FARM schlägt.",
        says="Die Breite des bewerteten Fensters entscheidet das Vorzeichen des "
             "Morphologiebefunds — mit der gebauten Labelbreite liegt die Kaskade unter FARM, "
             "mit der realen Spike-Dauer darüber.",
        use_for="Der Satz, der jede Morphologieaussage im Kapitel qualifiziert. Ohne diese "
                "Tabelle ist keine Morphologiezahl zitierfähig.",
        not_for="Sich die günstige Breite auszusuchen. Beide Breiten berichten."),
    "figure_5_19_spike_morphology_examples": dict(
        shows="Spike-Beispiele je Arm mit gemeinsamen y-Grenzen.",
        says="Wie der verbleibende Morphologieunterschied aussieht.",
        use_for="Die qualitative Ergänzung zu table_5_18.",
        not_for="Repräsentativität."),
    "figure_5_18_artifact_spike_tradeoff": dict(
        claims="R5.6.2-C2",
        shows="Artefaktentfernung gegen Spike-Erhalt als zweidimensionaler Kompromiss, mit "
              "angegebener Vorzugsrichtung.",
        says="Die beiden Ziele stehen in Spannung, und die Verfahren belegen verschiedene "
             "Punkte.",
        use_for="Der Kompromissabsatz von 5.6.",
        not_for="Eine Empfehlung ohne Angabe der Label-Breite, mit der die y-Achse gebildet "
                "wurde."),
    "table_5_19_replication_dataset_register": dict(
        claims="R5.6.5-C1",
        shows="Je Datensatzversion die Herkunft des Clean-Signals, die IED-Rate, die "
              "mittleren Amplituden von Clean und Artefakt, die Zahl der Spike-Ereignisse in "
              "Training und Validierung, das **kleinste erreichbare Holm-p** bei sechs "
              "Kennzahlen und daraus abgeleitet, ob die Spike- und die Bulk-Ebene überhaupt "
              "testbar sind.",
        says="Die Teststärke ist vorab aus der Ereigniszahl bestimmt, nicht nachträglich "
             "entschuldigt: bei sechs Kennzahlen braucht es mindestens acht Paare, sonst kann "
             "kein Ergebnis signifikant werden.",
        use_for="Die Begründung, warum es die Datensatzversionen gibt, und die Vorwegnahme "
                "jedes Einwands zur Stichprobengröße.",
        not_for="Als Ergebnis."),
    "table_5_20_replication_results": dict(
        claims="R5.6.5-C1, R5.6.5-C2",
        shows="Je Datensatzversion und Modell die IED-Rate, die Epochenzahl, das "
              "Bulk-Ergebnis gegen die Nullausgabe, ob EEG rekonstruiert wird, die Zahl der "
              "Spike-Ereignisse, die Morphologiedifferenz mit p — und getrennt, ob der Test "
              "**berechenbar** und ob er **teststark** war.",
        says="Die Bulk-Aussage repliziert über die Datensatzversionen; die "
             "Morphologie-Signifikanz tut das nicht überall, und die Trennung von "
             "„berechenbar“ und „teststark“ sagt warum.",
        use_for="Der Replikationsabsatz von 5.6.5.",
        not_for="Ein nicht signifikantes Morphologieergebnis als Gleichstand lesen, wenn die "
                "Spalte `morphology_power_sufficient` falsch ist."),
    "figure_5_20_morphology_replication": dict(
        claims="R5.6.5-C2",
        shows="Forest-Plot der Morphologieeffekte über die Datensatzversionen, eingefärbt "
              "nach Holm-korrigierter Signifikanz.",
        says="Wie konsistent der Morphologiebefund über die Versionen ist.",
        use_for="Die Abbildung zu 5.6.5.",
        not_for="Signifikanz am Intervall ablesen — maßgeblich ist die Einfärbung nach p_holm."),
    "table_5_22_gpu_runs": dict(
        shows="Je GPU-Lauf Pod, Familie, Seed, Lernrate, Verlustgewichte, Epochenzahl, beste "
              "Epoche und bester Validierungsverlust, ob dieser Verlust überhaupt "
              "vergleichbar ist, der Ausschlussgrund und die Trainingsdauer.",
        says="Welche Läufe stattgefunden haben und welche davon vergleichbar sind.",
        use_for="Die Nachvollziehbarkeit der Trainingsläufe in 5.6.",
        not_for="Validierungsverluste über Familien hinweg vergleichen, wo "
                "`val_loss_comparable` falsch ist — verschiedene Zielfunktionen sind "
                "verschiedene Größen."),
    "table_5_22b_ablations_against_seed_noise": dict(
        claims="R5.6.6-C1",
        shows="Je Lauf der beste Validierungsverlust gegen die aus mehreren Seeds bestimmte "
              "Spanne, plus ob der Lauf innerhalb dieser Spanne liegt und das daraus folgende "
              "Urteil.",
        says="Mehrere Ablationseffekte sind kleiner als der Unterschied zwischen zwei Seeds "
             "desselben Aufbaus — sie sind damit keine Effekte.",
        use_for="Der Absatz, der die Ablationstabelle relativiert. Dieser Maßstab gehört vor "
                "jede Ablationsaussage.",
        not_for="Eine Aussage über die Endergebnisse — verglichen sind Validierungsverluste."),
    "table_5_23_seed_spread_on_conclusions": dict(
        claims="R5.6.6-C2",
        shows="Je Seed dieselben Kopfkennzahlen: Bulk gegen die Nullausgabe mit p, ob EEG "
              "rekonstruiert wird, die Morphologiedifferenz mit p und Signifikanz, und die "
              "Zahl der Spike-Ereignisse.",
        says="Die Bulk-Schlussfolgerung ist seed-robust; die **Signifikanz** des "
             "Morphologiebefunds ist es nicht.",
        use_for="Die wichtigste Selbstbeschränkung von 5.6. Beide Hälften nennen — die "
                "robuste und die nicht robuste.",
        not_for="Den Validierungsverlust als Auswahlkriterium rechtfertigen: ein Seed mit "
                "deutlich besserem Validierungsverlust liefert dieselbe Bulk-Kennzahl."),
    "figure_5_22_seed_and_ablations": dict(
        claims="R5.6.6-C1",
        shows="Bester Validierungsverlust je Lauf, mit der Seed-Spanne als eingezeichnetem "
              "Maßstab.",
        says="Welche Läufe sich vom Seed-Rauschen abheben und welche nicht.",
        use_for="Die Abbildung zu 5.6.6.",
        not_for="Vergleiche über Zielfunktionen hinweg."),

    # ----------------------------------------------- 5.7 Einsatz in der Pipeline
    "table_5_21_pipeline_end_to_end": dict(
        claims="R5.7-C1, R5.7-C2",
        shows="Dieselbe EDF durch vier Ketten — ohne jede Korrektur (auch ohne die "
              "Aufräum-PCA, die selbst 16,7 % der Rohleistung entfernt), mit der FARM-Referenzkette, "
              "mit der Primärkorrektur des Trainings-Bundles (FARM + PCA/OBS(4, 300 Hz)) und "
              "mit dieser plus Kaskade. Je Arm die Schrittfolge, die Laufzeit, Kanalzahl und "
              "Abtastrate, der Rest-RMS, der entfernte Anteil, die spektralen Anteile und der "
              "Beitrag der Kaskade gegenüber **ihrem eigenen Eingang**.",
        says="Die Kaskade läuft in der ausgelieferten Kette auf einer echten Aufnahme, und "
             "ihr Beitrag über die Primärkorrektur hinaus ist beziffert.",
        use_for="Der Einsatznachweis von 5.7. Den dritten Arm im Text erklären: ohne ihn "
                "würde der Kaskade die zusätzliche PCA-Stufe zugerechnet.",
        not_for="Eine Genauigkeitsaussage. Auf einer echten Aufnahme gibt es kein sauberes "
                "Referenzsignal; alle Spalten sind deskriptiv. Und: den steigenden Rest-RMS "
                "des Kaskadenarms als Verschlechterung lesen — was sich ändert, ist die "
                "Zusammensetzung des Rests (R5.7-C2), nicht nachweislich seine Güte."),
    "table_5_21c_family_pipelines": dict(
        claims="R5.7-C3",
        shows="Alle vierzehn Modellfamilien plus die drei Referenzarme als Korrektor in "
              "derselben Kette, sortiert nach Rest-RMS: entfernte Leistung gegen den Arm "
              "ohne jede Korrektur, Spitze-Spitze, die Stufe an der Epochennaht absolut und "
              "im Verhältnis zum gewöhnlichen Sprung zwischen zwei Samples, die Verzerrung "
              "vor dem Scanbeginn und die Laufzeit.",
        says="Die Rangfolge des vereinheitlichten Holdouts überträgt sich nicht auf eine "
             "echte Aufnahme: drei Familien machen das Signal schlechter als gar keine "
             "Korrektur, zwei weitere korrigieren kaum. Dazu ein Defekt, den eine Auswertung "
             "je Epoche grundsätzlich nicht sehen kann — Stufen an den Epochengrenzen, weil "
             "jedes Segment seinen eigenen Mittelwert verliert.",
        use_for="Der Einsatzteil von 5.7 und der Beleg für die Fairness-Einschränkung in "
                "5.3.3. Im Text erwähnen, dass die Adapter bit-identisch zur "
                "Holdout-Inferenz verifiziert sind — sonst liest sich der Befund als "
                "Werkzeugfehler.",
        not_for="Als Rangliste der Architekturen. Es gibt kein Referenzsignal: der niedrigste "
                "Rest-RMS kann auch heißen, dass das EEG mitgelöscht wurde. Und die Spalte "
                "vor dem Scanbeginn ist ein Nullbefund — alle Korrektoren arbeiten nur auf "
                "Triggerepochen, sie unterscheidet die Arme nicht."),
    "table_5_21d_prediction_dc_bias": dict(
        claims="R5.7-C4",
        shows="Drei Arme zweimal: mit der ausgelieferten Spezifikation und mit abgezogenem "
              "Gleichanteil der Vorhersage, je Rest-RMS und entfernte Leistung, dazu ein "
              "Flag, ob der Versatz den Ausfall erklärt.",
        says="Zwei der drei schlechtesten Arme scheitern an einem konstanten Versatz in der "
             "Vorhersage, nicht an der Architektur — nested_gan springt von 44 % auf 98 % "
             "entfernter Leistung. Der dritte (dhct_gan) ändert sich nicht: sein Ausfall ist "
             "echte Verstärkung. Ob der Gleichanteil abgezogen wird, ist eine "
             "Auswertungsentscheidung der ursprünglichen Einzelevaluationen, keine "
             "Modelleigenschaft.",
        use_for="Der Diagnoseabsatz nach Tabelle 5.21c, und der Beleg für eine bisher nicht "
                "benannte Fairnessachse in 5.3.3: vier von vierzehn Referenzimplementierungen "
                "ziehen den Gleichanteil ab, zehn nicht.",
        not_for="Als Korrektur der Rangfolge. Die ausgelieferte Spezifikation bleibt "
                "unverändert, weil sie bit-identisch zur Holdout-Inferenz verifiziert ist; "
                "Tabelle 5.21c bleibt maßgeblich. Und dhct_gan_v2 ist auch mit DC-Abzug noch "
                "weit von der FARM-Referenz entfernt."),
    "table_5_21b_pipeline_chain": dict(
        shows="Die Kette in Reihenfolge, je Schritt der Prozessor und der **gemessene** Grund, "
              "warum er darin steht.",
        says="Kein Schritt der Kette ist Gewohnheit; für jeden gibt es eine Messung im "
             "Kapitel.",
        use_for="Der Aufbauabsatz von 5.7 und die Verbindung zurück zu 5.4.1.",
        not_for="Als Ergebnis."),
}
