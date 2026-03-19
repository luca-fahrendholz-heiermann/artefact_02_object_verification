# artefact_02_object_verification
AI Model for Classification of Objekt Verification based on Descriptor Differences

## Validation Workflow (`artefact_02_validation`)

### 1) Batch-Validierung laufen lassen
Erzeugt Vorhersagen pro Projekt und aggregiert pro Domain im Ordner `artefact_02_validation/results/...`.

```bash
python artefact_02_validation/run_validation.py --domain indoor
python artefact_02_validation/run_validation.py --domain outdoor
```

Optional wichtige Parameter:
- `--checkpoint <path_to_model.pth>`
- `--esf-exe <path_to_esf_estimation.exe>`
- `--threshold 0.96`
- `--knn-neighbors 10`
- `--device cpu|cuda`

### 2) Labeling-Interface starten (auch direkt über IDE)

#### Direkt aus IDE (Run / Debug)
`validation_interface.py` kann jetzt **ohne Argumente** gestartet werden.  
Dann wird automatisch die **neueste** Predictions-Datei aus `artefact_02_validation/results` verwendet (falls vorhanden).  
Falls keine Datei gefunden wird, startet die UI trotzdem und oben kann eine Predictions-Datei per Upload geladen werden.

```bash
python artefact_02_validation/validation_interface.py
```

Falls Port `7860` bereits belegt ist, versucht das Interface automatisch einen freien Port.
Alternativ kann ein fester Port gesetzt werden:
```bash
python artefact_02_validation/validation_interface.py --port 7861
```

Im Interface gibt es oben den Bereich **„Prediction-Datei hochladen (.xlsx/.json)”**.  
Nach dem Laden erkennt das Tool Projekt/Scan/Zeilen automatisch aus der Datei.

#### Optional explizite Datei angeben
```bash
python artefact_02_validation/validation_interface.py --prediction-file artefact_02_validation/results/Indoor_Production_QC/Indoor_Production_QC_all_projects_predictions.xlsx
```

Ground-Truth Labels sind binär: **0 oder 1**.

### 3) Metriken aus gelabelten Vorhersagen berechnen
```bash
python artefact_02_validation/eval_metrics.py --input artefact_02_validation/results/Indoor_Production_QC/Indoor_Production_QC_all_projects_predictions.xlsx
```

Outputs (im jeweiligen `metrics`-Unterordner):
- Confusion-Matrix PNGs
- `metrics_summary.xlsx` und `metrics_summary.json`
- Metadaten-Exports pro Scope

### 4) Zusätzliche Registrierungsausgabe: PPF-Ergebnis vor ICP als `.ply`
Wenn du bereits eine PPF-Transformation hast, kannst du die Punktwolke **nach PPF und vor ICP** als `.ply` speichern:

```bash
python artefact_02_validation/register_after_ppf.py \
  --source <source_scan.ply> \
  --target <target_model.ply> \
  --ppf-transform <ppf_transform.json> \
  --out-dir artefact_02_validation/registration_outputs
```

Das Skript:
- prüft zusätzlich Symmetrie-Kandidaten (`+180°` um Y und Z) als Startvarianten,
- prüft standardmäßig auch die inverse PPF-Transformation (hilfreich bei Richtung/Koordinatensystem-Problemen),
- wählt den besten Start auf Basis eines Distanz-Scores,
- speichert `source_after_ppf_before_icp.ply`,
- führt danach ICP aus und speichert `source_after_icp.ply`.
- schätzt fehlende Normalen automatisch, damit Point-to-Plane ICP nicht mit `requires target pointcloud to have normals` fehlschlägt.
- schreibt zusätzlich Transformationsdateien:
  - `T_est_world.txt` (Transformation im Weltkoordinatensystem)
  - `T_est_centered.txt` (Transformation im zentrierten Frame, oft stabiler für Fehlervergleich mit künstlicher Initial-Transformation)

Debug-Tipps:
- Mit `--save-all-candidates` werden alle getesteten Startlagen als `.ply` gespeichert.
- Wenn globale Fehlerwerte unrealistisch hoch sind (z. B. sehr große `Translation Error` trotz guter Überlagerung), ist häufig die Transformationsrichtung oder Einheit (mm/m) inkonsistent.
