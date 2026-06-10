"""
Interaktives Visualisierungstool für Wirbelsäulenmessungen
Ermöglicht die Anzeige und Filterung von Cobb-Winkelmessungen nach ID und Messtyp
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

# Importiere die benötigten Funktionen
from bspline_python import (
    interpolate_bspline, 
    calculate_tangents, 
    get_perpendicular_lines_at_indices,
    map_nodes_to_spline_and_keys,
    spine_labels,
    NodeItem
)
from dotmap import DotMap


class SpineMeasurementViewer:
    """Interaktiver Viewer für Wirbelsäulenmessungen"""
    
    def __init__(self, data_path):
        """
        Initialisiert den Viewer
        
        Args:
            data_path: Pfad zur JSON-Datei mit den Messdaten oder DataFrame
        """
        # Lade Daten
        if type(data_path) == eval("str"):
            self.df = pd.read_json(data_path)
        else:
            self.df = data_path

        print("Datensatz geladen mit {} Einträgen".format(len(self.df)))
        # Extrahiere verfügbare IDs und Messtypen
        self.available_ids = sorted(self.df['id'].unique())
        self.available_cobb_types = sorted(self.df['angle_data'].apply(
            lambda x: x.get('cobbType', 'Unknown') if isinstance(x, dict) else 'Unknown'
        ).unique())
        # Marker-Styles für bessere Unterscheidbarkeit innerhalb eines Typs
        self.marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        self.line_styles = ['-', '--', '-.', ':']

        # Setup UI
        self.setup_widgets()
        self.setup_plot()
        
    def setup_widgets(self):
        """Erstellt die UI-Widgets"""
        # ID-Auswahl (Dropdown für bessere Übersicht)
        self.id_selector = widgets.Dropdown(
            options=self.available_ids,
            value=self.available_ids[0],
            description='Patient ID:',
            style={'description_width': 'initial'}
        )
        
        # Messtyp-Auswahl (Mehrfachauswahl)
        self.type_selector = widgets.SelectMultiple(
            options=self.available_cobb_types,
            value=self.available_cobb_types,
            description='Messtypen:',
            style={'description_width': 'initial'},
            rows=len(self.available_cobb_types)
        )

        # Measurement-Auswahl (Mehrfachauswahl)
        self.measurement_selector = widgets.SelectMultiple(
            options=[],
            value=[],
            description='Messungen:',
            style={'description_width': 'initial'},
            rows=10
        )
        
        # Navigation Buttons
        self.prev_button = widgets.Button(
            description='◀ Vorherige ID',
            button_style='info',
            icon='arrow-left'
        )
        self.next_button = widgets.Button(
            description='Nächste ID ▶',
            button_style='info',
            icon='arrow-right'
        )
        
        # Info-Label
        self.info_label = widgets.HTML(value='')
        
        # Event Handler
        self.id_selector.observe(self.on_id_change, names='value')
        self.type_selector.observe(self.on_type_change, names='value')
        self.prev_button.on_click(self.on_prev_click)
        self.next_button.on_click(self.on_next_click)

    def setup_plot(self):
        """Erstellt die Plot-Figur"""
        self.output = widgets.Output()
        
    def on_id_change(self, change):
        """Callback wenn ID geändert wird"""
        self.update_plot()
        
    def on_type_change(self, change):
        """Callback wenn Messtypen geändert werden"""
        self.update_plot()
        
    def on_prev_click(self, b):
        """Navigiert zur vorherigen ID"""
        current_idx = self.available_ids.index(self.id_selector.value)
        if current_idx > 0:
            self.id_selector.value = self.available_ids[current_idx - 1]
            
    def on_next_click(self, b):
        """Navigiert zur nächsten ID"""
        current_idx = self.available_ids.index(self.id_selector.value)
        if current_idx < len(self.available_ids) - 1:
            self.id_selector.value = self.available_ids[current_idx + 1]
    
    def get_filtered_data(self):
        """Filtert Daten nach aktueller Auswahl"""
        patient_id = self.id_selector.value
        selected_types = list(self.type_selector.value)
        
        # Filtere nach ID
        filtered = self.df[self.df['id'] == patient_id].copy()
        
        # Filtere nach Messtyp
        filtered = filtered[filtered['angle_data'].apply(
            lambda x: x.get('cobbType', 'Unknown') in selected_types if isinstance(x, dict) else False
        )]
        
        return filtered
    
    def extract_nodes_from_measurement(self, angle_data):
        """
        Extrahiert Knoten aus angle_data
        Versucht verschiedene Formate
        """
        # Wenn Linien vorhanden sind, extrahiere Punkte
        if 'upperLine' in angle_data and 'lowerLine' in angle_data:
            upper = angle_data['upperLine']
            lower = angle_data['lowerLine']
            
            # Erstelle vereinfachte Punkte-Liste
            nodes = [
                (upper.get('x1', 0), upper.get('y1', 0)),
                (upper.get('x2', 0), upper.get('y2', 0)),
                (lower.get('x1', 0), lower.get('y1', 0)),
                (lower.get('x2', 0), lower.get('y2', 0))
            ]
            return nodes
        
        return None

    def plot_measurement(self, ax, angle_data, color, label, measurement_index=0):
        """Plottet eine einzelne Messung"""
        cobb_angle_data = angle_data.get('cobbAngle', {})
        angle = cobb_angle_data.get('angle', 0)
        vertebra_upper = cobb_angle_data.get('vertebra_upper', '?')
        vertebra_lower = cobb_angle_data.get('vertebra_lower', '?')

        # Extrahiere Linien
        upper_line = angle_data.get('upperLine')
        lower_line = angle_data.get('lowerLine')
        
        if upper_line and lower_line:
            # Wähle Marker und Linestyle basierend auf Index
            marker = self.marker_styles[measurement_index % len(self.marker_styles)]
            linestyle = self.line_styles[measurement_index % len(self.line_styles)]

            # Plotte obere Linie
            ax.plot(
                [upper_line.get('x1', 0), upper_line.get('x2', 0)],
                [upper_line.get('y1', 0), upper_line.get('y2', 0)],
                color=color, linewidth=2, alpha=0.7, label=label,
                solid_capstyle='round', linestyle=linestyle,
                marker=marker, markersize=8, markeredgewidth=2, markeredgecolor='white'
            )
            
            # Plotte untere Linie
            ax.plot(
                [lower_line.get('x1', 0), lower_line.get('x2', 0)],
                [lower_line.get('y1', 0), lower_line.get('y2', 0)],
                color=color, linewidth=2, alpha=0.7,
                solid_capstyle='round', linestyle=linestyle,
                marker=marker, markersize=8, markeredgewidth=2, markeredgecolor='white'
            )
            
            # Berechne Mittelpunkte für Text-Annotation
            mid_x = (upper_line.get('x1', 0) + upper_line.get('x2', 0) + 
                    lower_line.get('x1', 0) + lower_line.get('x2', 0)) / 4
            mid_y = ((upper_line.get('y1', 0) + upper_line.get('y2', 0) +
                    lower_line.get('y1', 0) + lower_line.get('y2', 0)) / 4) + (measurement_index *2)
            
            # Annotation mit Winkel
            ax.annotate(
                f'{marker} {angle}° ({vertebra_upper}-{vertebra_lower})',
                xy=(mid_x, mid_y),
                fontsize=9,
                color=color,
                weight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, alpha=0.8)
            )
    
    def update_plot(self):
        """Aktualisiert den Plot mit gefilterten Daten"""
        with self.output:
            clear_output(wait=True)
            
            filtered_data = self.get_filtered_data()
            
            if len(filtered_data) == 0:
                print("Keine Messungen für diese Auswahl gefunden.")
                return
            
            # Erstelle Figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Farben für verschiedene Messtypen
            colors = plt.cm.tab10.colors
            type_color_map = {t: colors[i % len(colors)] 
                            for i, t in enumerate(self.available_cobb_types)}

            # Zähle Messungen pro Typ für Index
            type_counters = {t: 0 for t in self.available_cobb_types}
            # Plotte alle Messungen
            plotted_labels = set()
            for idx, row in filtered_data.iterrows():
                angle_data = row['angle_data']
                cobb_type = angle_data.get('cobbType', 'Unknown')
                
                # Label nur einmal pro Typ anzeigen
                label = cobb_type if cobb_type not in plotted_labels else None
                if label:
                    plotted_labels.add(cobb_type)
                
                color = type_color_map.get(cobb_type, 'gray')
                measurement_index = type_counters[cobb_type]
                type_counters[cobb_type] += 1
                self.plot_measurement(ax, angle_data, color, label, measurement_index)
            
            # Formatierung
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_title(
                f'Wirbelsäulenmessungen - Patient ID: {self.id_selector.value}',
                fontsize=14, fontweight='bold'
            )
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # Y-Achse invertieren für medizinische Darstellung
            ax.axis('equal')
            
            plt.tight_layout()
            plt.show()
            
            # Update Info
            self.info_label.value = f"""
            <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                <b>Anzahl Messungen:</b> {len(filtered_data)}<br>
                <b>Bildtyp:</b> {filtered_data['img_type'].iloc[0] if len(filtered_data) > 0 else 'N/A'}<br>
                <b>Aufnahmedatum:</b> {filtered_data['acquisition_date'].iloc[0] if len(filtered_data) > 0 else 'N/A'}
            </div>
            """
    
    def display(self):
        """Zeigt das komplette UI an"""
        # Layout
        navigation = widgets.HBox([self.prev_button, self.next_button])
        controls = widgets.VBox([
            self.id_selector,
            navigation,
            self.type_selector,
            self.info_label
        ])
        
        ui = widgets.HBox([controls, self.output])
        
        display(ui)
        
        # Initial plot
        self.update_plot()


# Verwendung
if __name__ == '__main__':
    plt.ion()
    # Erstelle Viewer
    viewer = SpineMeasurementViewer('data/df5c_with_reconstructed_angles.json')
    
    # Zeige UI an
    viewer.display()