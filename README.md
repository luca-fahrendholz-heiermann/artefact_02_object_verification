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
Dann wird automatisch die **neueste** Predictions-Datei aus `artefact_02_validation/results` verwendet.

```bash
python artefact_02_validation/validation_interface.py
```

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
