"""
Spline-Optimierung für mehrere Messungen pro Objekt
Findet die optimalen Parameter für Spline-Interpolation/Approximation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, splprep, splev, make_interp_spline
from scipy.spatial.distance import directed_hausdorff
from itertools import product
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SplineOptimizer:
    """Optimiert Spline-Parameter für mehrere Messungen desselben Objekts"""
    
    def __init__(self, df: pd.DataFrame, id_column: str = 'id', nodes_column: str = 'nodes', sort_axis=0):
        """
        Args:
            df: DataFrame mit Messungen
            id_column: Name der ID-Spalte
            nodes_column: Name der Spalte mit Node-Arrays
        """
        self.df = df
        self.id_column = id_column
        self.nodes_column = nodes_column
        self.results = []
        if sort_axis>0:
            # swap x/y Achse in den Nodes, falls nötig
            self.df[self.nodes_column] = self.df[self.nodes_column].apply(lambda nodes: [(y, x) for x, y in nodes])
        
    def calculate_deviation_metrics(self, reference_spline_func, measurement_nodes: List[np.ndarray]) -> Dict:
        """Berechnet verschiedene Abweichungsmetriken"""
        
        deviations = []
        hausdorff_distances = []term
        
        for nodes in measurement_nodes:
            # Punkte auf der Referenz-Spline für jeden x-Wert der Messung
            x_vals = nodes[:, 0]
            
            # Evaluiere Spline an diesen Punkten
            try:
                y_spline = reference_spline_func(x_vals)
                y_measured = nodes[:, 1]
                
                # Punkt-zu-Punkt Abweichung
                point_deviations = np.abs(y_spline - y_measured)
                deviations.extend(point_deviations)
                
                # Hausdorff-Distanz (max. Abweichung)
                spline_points = np.column_stack([x_vals, y_spline])
                hausdorff_dist = max(
                    directed_hausdorff(nodes, spline_points)[0],
                    directed_hausdorff(spline_points, nodes)[0]
                )
                hausdorff_distances.append(hausdorff_dist)
                
            except Exception as e:
                continue
        
        if not deviations:
            return None
            
        return {
            'mean_deviation': np.mean(deviations),
            'median_deviation': np.median(deviations),
            'max_deviation': np.max(deviations),
            'std_deviation': np.std(deviations),
            'rmse': np.sqrt(np.mean(np.array(deviations)**2)),
            'mean_hausdorff': np.mean(hausdorff_distances),
            'max_hausdorff': np.max(hausdorff_distances)
        }
    
    def fit_univariate_spline(self, nodes_list: List[np.ndarray], s: float, k: int) -> Tuple:
        """Fitted einen UnivariateSpline auf gemittelte Daten"""
        
        # Kombiniere alle Messungen
        all_nodes = np.vstack(nodes_list)
        
        # Sortiere nach x
        sorted_idx = np.argsort(all_nodes[:, 0])
        x = all_nodes[sorted_idx, 0]
        y = all_nodes[sorted_idx, 1]
        
        # Entferne Duplikate in x (mittle y-Werte)
        unique_x, inverse = np.unique(x, return_inverse=True)
        unique_y = np.array([y[inverse == i].mean() for i in range(len(unique_x))])
        
        try:
            spline = UnivariateSpline(unique_x, unique_y, s=s, k=k)
            return spline, unique_x, unique_y
        except Exception as e:
            return None, None, None
    
    def fit_parametric_spline(self, nodes_list: List[np.ndarray], s: float, k: int) -> Tuple:
        """Fitted einen parametrischen Spline (splprep/splev) auf gemittelte Daten"""
        
        # Kombiniere alle Messungen
        all_nodes = np.vstack(nodes_list)
        
        # Sortiere nach x
        sorted_idx = np.argsort(all_nodes[:, 0])
        points = all_nodes[sorted_idx]
        
        try:
            # Parametrischer Spline
            tck, u = splprep([points[:, 0], points[:, 1]], s=s, k=k)
            
            # Erstelle Evaluierungsfunktion
            def eval_func(x_vals):
                # Finde Parameter u für gegebene x-Werte
                spline_x = splev(u, tck)[0]
                spline_y = splev(u, tck)[1]
                # Interpoliere y für gegebene x
                return np.interp(x_vals, spline_x, spline_y)
            
            return eval_func, tck, u
        except Exception as e:
            return None, None, None
    
    def optimize_for_id(self, object_id, verbose: bool = False) -> pd.DataFrame:
        """Optimiert Spline-Parameter für eine bestimmte ID"""
        
        # Filtere Daten für diese ID
        id_data = self.df[self.df[self.id_column] == object_id]
        nodes_list = [np.array(nodes) for nodes in id_data[self.nodes_column].values]
        
        if len(nodes_list) < 2:
            print(f"ID {object_id}: Nur {len(nodes_list)} Messung(en), überspringe...")
            return pd.DataFrame()
        
        # Parameter-Grid
        smoothing_values = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        k_values = [1, 2, 3]  # Spline-Grad
        methods = ['univariate', 'parametric']
        
        results = []
        
        for method, s, k in product(methods, smoothing_values, k_values):
            try:
                if method == 'univariate':
                    spline_func, x, y = self.fit_univariate_spline(nodes_list, s, k)
                else:
                    spline_func, tck, u = self.fit_parametric_spline(nodes_list, s, k)
                
                if spline_func is None:
                    continue
                
                # Berechne Abweichungen
                metrics = self.calculate_deviation_metrics(spline_func, nodes_list)
                
                if metrics is None:
                    continue
                
                results.append({
                    'id': object_id,
                    'method': method,
                    'smoothing_s': s,
                    'degree_k': k,
                    **metrics
                })
                
            except Exception as e:
                if verbose:
                    print(f"Fehler bei {method}, s={s}, k={k}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def optimize_all(self, verbose: bool = False) -> pd.DataFrame:
        """Optimiert für alle IDs im DataFrame"""
        
        all_results = []
        unique_ids = self.df[self.id_column].unique()
        
        print(f"Optimiere Splines für {len(unique_ids)} verschiedene Objekte...")
        
        for i, object_id in enumerate(unique_ids, 1):
            if verbose or i % 10 == 0:
                print(f"Verarbeite ID {object_id} ({i}/{len(unique_ids)})...")
            
            id_results = self.optimize_for_id(object_id, verbose=verbose)
            if not id_results.empty:
                all_results.append(id_results)
        
        if not all_results:
            return pd.DataFrame()
        
        self.results = pd.concat(all_results, ignore_index=True)
        return self.results
    
    def get_best_parameters(self, metric: str = 'rmse') -> Dict:
        """Findet die besten Parameter basierend auf einer Metrik"""
        
        if self.results is None or len(self.results) == 0:
            return None
        
        # Gruppiere nach Parameterkombination und mittele über alle IDs
        grouped = self.results.groupby(['method', 'smoothing_s', 'degree_k'])[metric].agg(['mean', 'std', 'count'])
        grouped = grouped.reset_index()
        
        # Finde Minimum
        best_idx = grouped['mean'].idxmin()
        best = grouped.loc[best_idx]
        
        return {
            'method': best['method'],
            'smoothing_s': best['smoothing_s'],
            'degree_k': best['degree_k'],
            f'{metric}_mean': best['mean'],
            f'{metric}_std': best['std'],
            'n_objects': int(best['count'])
        }
    
    def plot_comparison(self, object_id, top_n: int = 5, metric: str = 'rmse'):
        """Visualisiert die besten Spline-Fits für ein Objekt"""
        
        # Hole Daten für diese ID
        id_data = self.df[self.df[self.id_column] == object_id]
        nodes_list = [np.array(nodes) for nodes in id_data[self.nodes_column].values]
        
        # Hole beste Konfigurationen für diese ID
        id_results = self.results[self.results['id'] == object_id].sort_values(metric).head(top_n)
        
        if len(id_results) == 0:
            print(f"Keine Ergebnisse für ID {object_id}")
            return
        
        fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 4))
        if top_n == 1:
            axes = [axes]
        
        # Gemeinsamer x-Bereich
        all_x = np.concatenate([nodes[:, 0] for nodes in nodes_list])
        x_range = np.linspace(all_x.min(), all_x.max(), 200)
        
        for idx, (ax, (_, row)) in enumerate(zip(axes, id_results.iterrows())):
            # Plot Original-Messungen
            for i, nodes in enumerate(nodes_list):
                ax.plot(nodes[:, 0], nodes[:, 1], 'o', alpha=0.3, markersize=3, 
                       label=f'Messung {i+1}' if idx == 0 else None)
            
            # Fit und plot Spline
            method = row['method']
            s = row['smoothing_s']
            k = int(row['degree_k'])
            
            if method == 'univariate':
                spline_func, _, _ = self.fit_univariate_spline(nodes_list, s, k)
            else:
                spline_func, _, _ = self.fit_parametric_spline(nodes_list, s, k)
            
            if spline_func:
                y_spline = spline_func(x_range)
                ax.plot(x_range, y_spline, 'r-', linewidth=2, label='Spline')
            
            ax.set_title(f"{method}\ns={s}, k={k}\n{metric}={row[metric]:.4f}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'output/spline_comparison_id_{object_id}.png', dpi=150, bbox_inches='tight')
        print(f"Vergleichsplot gespeichert für ID {object_id}")
        
        return fig
    
    def generate_report(self) -> str:
        """Erstellt einen detaillierten Bericht über die Optimierung"""
        
        report = []
        report.append("=" * 70)
        report.append("SPLINE-OPTIMIERUNGS-BERICHT")
        report.append("=" * 70)
        report.append("")
        
        # Allgemeine Statistiken
        n_objects = self.results['id'].nunique()
        n_configs = len(self.results.groupby(['method', 'smoothing_s', 'degree_k']))
        report.append(f"Anzahl getesteter Objekte: {n_objects}")
        report.append(f"Anzahl getesteter Konfigurationen: {n_configs}")
        report.append("")
        
        # Beste Parameter für verschiedene Metriken
        metrics = ['rmse', 'mean_deviation', 'max_deviation', 'mean_hausdorff']
        
        for metric in metrics:
            report.append(f"\n--- Beste Parameter nach {metric.upper()} ---")
            best = self.get_best_parameters(metric)
            if best:
                report.append(f"Methode: {best['method']}")
                report.append(f"Smoothing (s): {best['smoothing_s']}")
                report.append(f"Grad (k): {best['degree_k']}")
                report.append(f"Mittlerer {metric}: {best[f'{metric}_mean']:.6f} (±{best[f'{metric}_std']:.6f})")
                report.append(f"Basierend auf {best['n_objects']} Objekten")
        
        # Top 5 Konfigurationen nach RMSE
        report.append("\n" + "=" * 70)
        report.append("TOP 5 KONFIGURATIONEN (nach RMSE)")
        report.append("=" * 70)
        
        grouped = self.results.groupby(['method', 'smoothing_s', 'degree_k']).agg({
            'rmse': ['mean', 'std'],
            'mean_deviation': 'mean',
            'max_deviation': 'mean',
            'id': 'count'
        }).reset_index()
        
        grouped.columns = ['method', 'smoothing_s', 'degree_k', 'rmse_mean', 'rmse_std', 
                          'mean_dev', 'max_dev', 'n_objects']
        grouped = grouped.sort_values('rmse_mean').head(5)
        
        for i, row in grouped.iterrows():
            report.append(f"\n{i+1}. {row['method'].upper()} | s={row['smoothing_s']} | k={int(row['degree_k'])}")
            report.append(f"   RMSE: {row['rmse_mean']:.6f} (±{row['rmse_std']:.6f})")
            report.append(f"   Mean Dev: {row['mean_dev']:.6f} | Max Dev: {row['max_dev']:.6f}")
            report.append(f"   Objekte: {int(row['n_objects'])}")
        
        return "\n".join(report)


def create_example_data() -> pd.DataFrame:
    """Erstellt Beispieldaten zum Testen"""
    
    np.random.seed(42)
    data = []
    
    # Erstelle 3 verschiedene Objekte mit je 5-10 Messungen
    for obj_id in range(3):
        n_measurements = np.random.randint(5, 11)
        
        # Wahre Kurve (unterschiedlich für jedes Objekt)
        if obj_id == 0:
            true_func = lambda x: np.sin(x)
        elif obj_id == 1:
            true_func = lambda x: x**2 / 10
        else:
            true_func = lambda x: np.exp(-x/5) * np.cos(x)
        
        for meas in range(n_measurements):
            # Zufällige Anzahl von Punkten
            n_points = np.random.randint(20, 50)
            x = np.sort(np.random.uniform(0, 10, n_points))
            
            # Füge Rauschen hinzu
            noise_level = 0.1
            y = true_func(x) + np.random.normal(0, noise_level, n_points)
            
            nodes = list(zip(x, y))
            data.append({'id': f'object_{obj_id}', 'nodes': nodes})
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Erstelle Beispieldaten
    print("Erstelle Beispieldaten...")
    df = create_example_data()
    print(f"Datensatz: {len(df)} Messungen für {df['id'].nunique()} Objekte\n")
    
    # Initialisiere Optimizer
    optimizer = SplineOptimizer(df, id_column='id', nodes_column='nodes')
    
    # Optimiere
    results = optimizer.optimize_all(verbose=False)
    
    # Erstelle Bericht
    report = optimizer.generate_report()
    print(report)
    
    # Speichere Ergebnisse
    results.to_csv('output/spline_optimization_results.csv', index=False)
    print("\n\nDetailierte Ergebnisse gespeichert: spline_optimization_results.csv")
    
    # Erstelle Visualisierung für erstes Objekt
    print("\nErstelle Visualisierung...")
    optimizer.plot_comparison('object_0', top_n=5, metric='rmse')
    
    # Beste Parameter ausgeben
    print("\n" + "=" * 70)
    print("EMPFOHLENE PARAMETER FÜR ZUKÜNFTIGE ANWENDUNGEN:")
    print("=" * 70)
    best = optimizer.get_best_parameters('rmse')
    print(f"\nMethode: {best['method']}")
    print(f"Smoothing-Parameter (s): {best['smoothing_s']}")
    print(f"Spline-Grad (k): {best['degree_k']}")
    print(f"\nErwarteter RMSE: {best['rmse_mean']:.6f} (±{best['rmse_std']:.6f})")
