"""
Anwendungsbeispiel: Spline-Optimierung mit eigenen Daten
"""

import pandas as pd
import numpy as np
from spline_optimization import SplineOptimizer

# ============================================================================
# SCHRITT 1: Lade deine Daten
# ============================================================================
# Beispiel: df = pd.read_csv('meine_daten.csv')
# Dein DataFrame sollte haben:
# - Eine Spalte mit IDs (z.B. 'id' oder 'object_id')
# - Eine Spalte mit Node-Arrays: [(x1,y1), (x2,y2), ..., (xn,yn)]

# Falls deine Nodes als String gespeichert sind, konvertiere sie:
# import ast
# df['nodes'] = df['nodes'].apply(ast.literal_eval)

# Für dieses Beispiel verwenden wir Testdaten
from spline_optimization import create_example_data
df = create_example_data()

print("Daten geladen:")
print(f"  - {len(df)} Messungen")
print(f"  - {df['id'].nunique()} verschiedene Objekte")
print(f"  - Durchschnittlich {len(df) / df['id'].nunique():.1f} Messungen pro Objekt\n")

# ============================================================================
# SCHRITT 2: Initialisiere den Optimizer
# ============================================================================
optimizer = SplineOptimizer(
    df, 
    id_column='id',        # Name deiner ID-Spalte
    nodes_column='nodes'   # Name deiner Nodes-Spalte
)

# ============================================================================
# SCHRITT 3: Optimiere die Spline-Parameter
# ============================================================================
print("Starte Optimierung...")
print("(Dies kann einige Minuten dauern, abhängig von der Datenmenge)\n")

results = optimizer.optimize_all(verbose=False)

# ============================================================================
# SCHRITT 4: Analysiere die Ergebnisse
# ============================================================================

# Bericht ausgeben
print(optimizer.generate_report())

# ============================================================================
# SCHRITT 5: Finde die besten Parameter
# ============================================================================

print("\n" + "="*70)
print("EMPFOHLENE PARAMETER FÜR DEINE ZUKÜNFTIGE SPLINE-ERSTELLUNG:")
print("="*70 + "\n")

# Nach verschiedenen Kriterien
metrics = {
    'rmse': 'Root Mean Square Error (ausgeglichene Metrik)',
    'mean_deviation': 'Mittlere Abweichung (durchschnittlicher Fehler)',
    'max_deviation': 'Maximale Abweichung (worst-case)',
    'mean_hausdorff': 'Hausdorff-Distanz (Kurvenähnlichkeit)'
}

for metric, description in metrics.items():
    best = optimizer.get_best_parameters(metric)
    print(f"Optimiert nach {description}:")
    print(f"  Methode: {best['method']}")
    print(f"  Smoothing (s): {best['smoothing_s']}")
    print(f"  Grad (k): {best['degree_k']}")
    print(f"  Wert: {best[f'{metric}_mean']:.6f}\n")

# ============================================================================
# SCHRITT 6: Visualisiere Ergebnisse
# ============================================================================

print("\nErstelle Visualisierungen...")

# Visualisiere für jedes Objekt die Top-5-Konfigurationen
for obj_id in df['id'].unique():
    optimizer.plot_comparison(obj_id, top_n=5, metric='rmse')

print(f"\nVisualisierungen wurden erstellt!")

# ============================================================================
# SCHRITT 7: Speichere detaillierte Ergebnisse
# ============================================================================

# Speichere alle getesteten Konfigurationen
results.to_csv('/mnt/user-data/outputs/alle_konfigurationen.csv', index=False)

# Erstelle Summary mit den besten Konfigurationen
summary = results.groupby(['method', 'smoothing_s', 'degree_k']).agg({
    'rmse': ['mean', 'std', 'min', 'max'],
    'mean_deviation': 'mean',
    'max_deviation': 'mean',
    'id': 'count'
}).round(6)

summary.to_csv('/mnt/user-data/outputs/konfigurationen_summary.csv')

print("\nErgebnisse gespeichert:")
print("  - alle_konfigurationen.csv (alle Tests)")
print("  - konfigurationen_summary.csv (Zusammenfassung)")
print("  - spline_comparison_id_*.png (Visualisierungen)")

# ============================================================================
# SCHRITT 8: Anwendung der optimalen Parameter
# ============================================================================

print("\n" + "="*70)
print("SO VERWENDEST DU DIE OPTIMALEN PARAMETER IN ZUKUNFT:")
print("="*70 + "\n")

best = optimizer.get_best_parameters('rmse')

code_example = f"""
from scipy.interpolate import UnivariateSpline, splprep, splev
import numpy as np

# Deine Messdaten (mehrere Messungen des gleichen Objekts)
measurements = [...]  # Liste von Arrays [(x1,y1), (x2,y2), ...]

# Kombiniere alle Messungen
all_nodes = np.vstack(measurements)
sorted_idx = np.argsort(all_nodes[:, 0])
x = all_nodes[sorted_idx, 0]
y = all_nodes[sorted_idx, 1]

# Entferne Duplikate
unique_x, inverse = np.unique(x, return_inverse=True)
unique_y = np.array([y[inverse == i].mean() for i in range(len(unique_x))])

# Erstelle Spline mit optimalen Parametern
"""

if best['method'] == 'univariate':
    code_example += f"""
spline = UnivariateSpline(
    unique_x, 
    unique_y, 
    s={best['smoothing_s']},  # Smoothing-Parameter
    k={best['degree_k']}      # Spline-Grad
)

# Verwende den Spline
x_new = np.linspace(unique_x.min(), unique_x.max(), 100)
y_new = spline(x_new)
"""
else:
    code_example += f"""
tck, u = splprep(
    [unique_x, unique_y], 
    s={best['smoothing_s']},  # Smoothing-Parameter
    k={best['degree_k']}      # Spline-Grad
)

# Verwende den Spline
u_new = np.linspace(0, 1, 100)
x_new, y_new = splev(u_new, tck)
"""

print(code_example)

# ============================================================================
# ZUSÄTZLICHE ANALYSE
# ============================================================================

print("\n" + "="*70)
print("ZUSÄTZLICHE ERKENNTNISSE:")
print("="*70 + "\n")

# Vergleich der Methoden
method_comparison = results.groupby('method')['rmse'].agg(['mean', 'std', 'count'])
print("Vergleich der Methoden (RMSE):")
print(method_comparison)
print()

# Einfluss des Smoothing-Parameters
smoothing_comparison = results.groupby('smoothing_s')['rmse'].agg(['mean', 'std'])
print("Einfluss des Smoothing-Parameters (RMSE):")
print(smoothing_comparison.sort_values('mean'))
print()

# Einfluss des Spline-Grads
degree_comparison = results.groupby('degree_k')['rmse'].agg(['mean', 'std'])
print("Einfluss des Spline-Grads (RMSE):")
print(degree_comparison.sort_values('mean'))
