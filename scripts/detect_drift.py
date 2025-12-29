#!/usr/bin/env python3
"""
Script de détection de drift avec Evidently
"""

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd
import sys

def detect_drift(reference_path, current_path):
    """Détecte le drift entre deux datasets"""
    
    # Chargement des données
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    
    # Création du rapport
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Sauvegarde
    report.save_html('reports/drift_report.html')
    
    # Vérification du drift
    drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        print("⚠️ Data drift detected!")
        return 1
    else:
        print("✅ No significant drift detected")
        return 0

if __name__ == "__main__":
    reference_path = sys.argv[1] if len(sys.argv) > 1 else "data/reference.csv"
    current_path = sys.argv[2] if len(sys.argv) > 2 else "data/current.csv"
    
    sys.exit(detect_drift(reference_path, current_path))