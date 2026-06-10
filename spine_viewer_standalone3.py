#!/usr/bin/env python3
"""
Standalone Wirbelsäulenmessungs-Viewer
Kann direkt als Python-Skript ausgeführt werden: python spine_viewer.py

Benötigt: pandas, matplotlib, numpy
Optional für bessere UI: PyQt5 oder tkinter
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox, RadioButtons
from matplotlib.lines import Line2D
import json
import sys
import argparse
from pathlib import Path
import numpy as np

spine_labels = ['C0','C1','C2','C3','C4','C5','C6','C7','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5','S1']
spine_regions = {'CT':[0,9],'PT': [10,12],'MT':[13,18],'TL':[19,20],'LU':[21,25]}
total_regions = list(spine_regions.keys())+spine_labels


# Source - https://stackoverflow.com/a
# Posted by WayToDoor
# Retrieved 2026-01-18, License - CC BY-SA 4.0

def link(uri, label=None):
    if label is None:
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)


class SpineViewer:
    """Interaktiver Viewer für Wirbelsäulenmessungen"""
    
    def __init__(self, data_path, key_path):
        """
        Initialisiert den Viewer
        
        Args:
            data_path: Pfad zur JSON-Datei mit den Messdaten
        """
        print(f"Lade Daten aus: {data_path}")
        
        # Lade Daten
        try:
            self.df = pd.read_json(data_path)
            print(f"✓ {len(self.df)} Messungen geladen")
        except Exception as e:
            print(f"✗ Fehler beim Laden der Daten: {e}")
            sys.exit(1)

        print(f"Lade Keys aus: {key_path}")
        try:
            self.df_keys = pd.read_json(key_path).set_index('id')
        except Exception as e:
            print(f"✗ Fehler beim Laden der Daten: {e}")
            sys.exit(1)
        
        # Extrahiere verfügbare IDs und Messtypen
        self.available_ids = sorted(self.df['id'].unique())
        print(f"✓ {len(self.available_ids)} unterschiedliche Image-IDs gefunden")
        
        self.available_cobb_types = sorted(self.df['angle_data'].apply(
            lambda x: x.get('cobbType', 'Unknown') if isinstance(x, dict) else 'Unknown'
        ).unique())
        print(f"✓ Messtypen: {', '.join(self.available_cobb_types)}")
        
        # Aktueller Status
        self.current_id_index = 0
        self.current_id = self.available_ids[self.current_id_index]
        self.active_types = {t: True for t in self.available_cobb_types}
        self.selected_dataset_indices = 'all'  # 'all' oder Liste von measure_dataset_index
        
        # Farben für Messtypen
        colors = plt.cm.tab10.colors
        self.type_color_map = {
            t: colors[i % len(colors)] 
            for i, t in enumerate(self.available_cobb_types)
        }
        
        # Marker-Styles für bessere Unterscheidbarkeit innerhalb eines Typs
        self.marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
        self.line_styles = ['-', '--', '-.', ':']
        
        # Marker-zu-Unicode Mapping für Annotationen
        self.marker_to_unicode = {
            'o': '●',  # Kreis
            's': '■',  # Quadrat
            '^': '▲',  # Dreieck oben
            'D': '◆',  # Diamant
            'v': '▼',  # Dreieck unten
            '<': '◀',  # Dreieck links
            '>': '▶',  # Dreieck rechts
            'p': '⬟',  # Pentagon
            '*': '★',  # Stern
            'h': '⬢',  # Hexagon
            'X': '✕',  # X
            'P': '⊕',  # Plus im Kreis
        }
        
        # Figure und Widgets
        self.fig = None
        self.ax_main = None
        self.ax_nodes = None
        self.ax_stats =None
        self.widgets = {}
        
    def setup_ui(self):
        """Erstellt die Benutzeroberfläche"""
        # Erstelle Figure
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.canvas.manager.set_window_title('Wirbelsäulenmessungs-Viewer')
        
        # Layout: Hauptplot (70%) + Controls (30%)
        # Hauptplot
        self.ax_main = plt.subplot2grid((14, 10), (0, 0), colspan=4, rowspan=14)
        self.ax_nodes = plt.subplot2grid((14, 10), (0, 3), colspan=4, rowspan=14)
        
        # Control-Bereich rechts
        # Navigation
        ax_prev = plt.subplot2grid((14, 10), (0, 7), colspan=1, rowspan=2)
        ax_next = plt.subplot2grid((14, 10), (0, 9), colspan=1, rowspan=2)
        
        self.widgets['btn_prev'] = Button(ax_prev, '◀ Vorherige ID', 
                                          color='lightblue', hovercolor='skyblue')
        self.widgets['btn_prev'].on_clicked(self.on_prev)
        
        self.widgets['btn_next'] = Button(ax_next, 'Nächste ID ▶', 
                                          color='lightblue', hovercolor='skyblue')
        self.widgets['btn_next'].on_clicked(self.on_next)
        
        # ID-Eingabe
        ax_text = plt.subplot2grid((14, 10), (0, 8), colspan=1, rowspan=2)
        self.widgets['text_box'] = TextBox(ax_text, 'ID:', 
                                           initial=str(self.current_id))
        self.widgets['text_box'].on_submit(self.on_id_input)
        
        # Messauswahl (Radio Buttons)
        ax_radio = plt.subplot2grid((14, 10), (2, 7), colspan=3, rowspan=3)
        ax_radio.set_title('Messungen (Dataset) auswählen', fontsize=10, fontweight='bold', pad=10)
        
        # Erstelle Optionen basierend auf max(measure_dataset_index)
        max_dataset_index = self._get_max_dataset_index()
        radio_labels = ['Alle'] + [f'Dataset {i}' for i in range(max_dataset_index + 1)]
        
        self.widgets['radio'] = RadioButtons(ax_radio, radio_labels, active=0)
        self.widgets['radio'].on_clicked(self.on_dataset_select)
        
        # Messtyp-Checkboxen
        ax_check = plt.subplot2grid((14, 10), (5, 7), colspan=3, rowspan=4)
        ax_check.set_title('Messtypen', fontsize=10, fontweight='bold', pad=10)
        
        self.widgets['check'] = CheckButtons(
            ax_check, 
            self.available_cobb_types,
            [self.active_types[t] for t in self.available_cobb_types]
        )
        self.widgets['check'].on_clicked(self.on_type_toggle)
        # Erstelle Statistik-Subplot
        self.ax_stats = plt.subplot2grid((14, 10), (9, 7), colspan=3, rowspan=5)
        self.ax_stats.set_title('Statistik nach Wirbelsäulenregion', fontsize=10, fontweight='bold', pad=10)
        # Info-Bereich
        # self.ax_info = plt.subplot2grid((14, 10), (9, 7), colspan=3, rowspan=3)
        # self.ax_info.axis('off')
        
        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout(pad=2.0)
        
    def _get_max_dataset_index(self):
        """Ermittelt den maximalen measure_dataset_index über alle IDs"""
        if 'measure_dataset_index' in self.df.columns:
            return int(self.df['measure_dataset_index'].max())
        return 0
    
    def _get_max_dataset_index_for_id(self, image_id):
        """Ermittelt den maximalen measure_dataset_index für eine bestimmte ID"""
        if 'measure_dataset_index' in self.df.columns:
            filtered = self.df[self.df['id'] == image_id]
            if len(filtered) > 0:
                return int(filtered['measure_dataset_index'].max())
        return 0
    
    def on_dataset_select(self, label):
        """Callback für Dataset-Auswahl"""
        if label == 'Alle':
            self.selected_dataset_indices = 'all'
            print(f"Dataset-Auswahl: Alle")
        else:
            # Extrahiere Index aus Label (z.B. "Dataset 0" -> 0)
            index = int(label.split()[1])
            self.selected_dataset_indices = [index]
            print(f"Dataset-Auswahl: {index}")
        
        self.update_plot()
        
    def on_key_press(self, event):
        """Keyboard-Shortcuts"""
        if event.key == 'left' or event.key == 'pageup':
            self.on_prev(None)
        elif event.key == 'right' or event.key == 'pagedown':
            self.on_next(None)
        elif event.key == 'home':
            self.current_id_index = 0
            self.current_id = self.available_ids[self.current_id_index]
            self.widgets['text_box'].set_val(str(self.current_id))
            self.update_plot()
        elif event.key == 'end':
            self.current_id_index = len(self.available_ids) - 1
            self.current_id = self.available_ids[self.current_id_index]
            self.widgets['text_box'].set_val(str(self.current_id))
            self.update_plot()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)
        elif event.key.isdigit():
            # Schnellauswahl: 0-9 für Dataset-Index
            index = int(event.key)
            max_idx = self._get_max_dataset_index_for_id(self.current_id)
            
            if index <= max_idx:
                self.selected_dataset_indices = [index]
                print(f"Schnellauswahl: Dataset {index}")
                self.update_plot()
            else:
                print(f"⚠ Dataset {index} nicht verfügbar (max: {max_idx})")
        elif event.key == 'a':
            # 'a' für "Alle"
            self.selected_dataset_indices = 'all'
            print("Schnellauswahl: Alle Datasets")
            self.update_plot()
    
    def on_prev(self, event):
        """Vorherige ID"""
        if self.current_id_index > 0:
            self.current_id_index -= 1
            self.current_id = self.available_ids[self.current_id_index]
            self.widgets['text_box'].set_val(str(self.current_id))
            self.update_plot()
    
    def on_next(self, event):
        """Nächste ID"""
        if self.current_id_index < len(self.available_ids) - 1:
            self.current_id_index += 1
            self.current_id = self.available_ids[self.current_id_index]
            self.widgets['text_box'].set_val(str(self.current_id))
            self.update_plot()
    
    def on_id_input(self, text):
        """ID-Eingabe"""
        try:
            new_id = int(text)
            if new_id in self.available_ids:
                self.current_id = new_id
                self.current_id_index = self.available_ids.index(new_id)
                self.update_plot()
            else:
                print(f"⚠ ID {new_id} nicht gefunden!")
                print(f"  Verfügbare IDs: {self.available_ids}")
                self.widgets['text_box'].set_val(str(self.current_id))
        except ValueError:
            print(f"⚠ Ungültige ID: {text}")
            self.widgets['text_box'].set_val(str(self.current_id))
    
    def on_type_toggle(self, label):
        """Messtyp an-/ausschalten"""
        self.active_types[label] = not self.active_types[label]
        print(f"{'✓' if self.active_types[label] else '✗'} {label}")
        self.update_plot()
    
    def get_filtered_data(self):
        """Filtert Daten nach aktueller Auswahl"""
        # Filtere nach ID
        filtered = self.df[self.df['id'] == self.current_id].copy()
        
        # Filtere nach aktiven Messtypen
        active_type_list = [t for t, active in self.active_types.items() if active]
        filtered = filtered[filtered['angle_data'].apply(
            lambda x: x.get('cobbType', 'Unknown') in active_type_list 
            if isinstance(x, dict) else False
        )]
        
        # Filtere nach ausgewählten Dataset-Indizes
        if self.selected_dataset_indices != 'all':
            filtered = filtered[filtered['measure_dataset_index'].isin(self.selected_dataset_indices)]
        
        return filtered
    
    def get_dataset_counts_per_type(self):
        """
        Ermittelt die Anzahl der verfügbaren Datasets pro Messtyp für die aktuelle ID
        
        Returns:
            Dictionary mit {cobbType: {dataset_index: count}}
        """
        image_data = self.df[self.df['id'] == self.current_id]
        active_type_list = [t for t, active in self.active_types.items() if active]
        
        counts = {}
        for cobb_type in active_type_list:
            type_data = image_data[image_data['angle_data'].apply(
                lambda x: x.get('cobbType', 'Unknown') == cobb_type 
                if isinstance(x, dict) else False
            )]
            
            # Zähle pro dataset_index
            dataset_counts = {}
            for idx in type_data['measure_dataset_index'].unique():
                count = len(type_data[type_data['measure_dataset_index'] == idx])
                dataset_counts[int(idx)] = count
            
            counts[cobb_type] = dataset_counts
        
        return counts
    
    def plot_measurement(self, angle_data, color, label, dataset_index=0, nodes_data=None):
        """Plottet eine einzelne Messung"""
        cobb_angle_data = angle_data.get('cobbAngle', {})
        angle = cobb_angle_data.get('angle', 0)
        vertebra_upper = cobb_angle_data.get('vertebra_upper', '?')
        vertebra_lower = cobb_angle_data.get('vertebra_lower', '?')
        
        upper_line = angle_data.get('upperLine')
        lower_line = angle_data.get('lowerLine')
        
        if not (upper_line and lower_line):
            return None
        
        # Wähle Marker und Linestyle basierend auf dataset_index
        marker = self.marker_styles[dataset_index % len(self.marker_styles)]
        linestyle = self.line_styles[dataset_index % len(self.line_styles)]
        marker_unicode = self.marker_to_unicode.get(marker, '●')
        
        # Plotte obere Linie mit Marker
        self.ax_main.plot(
            [upper_line.get('x1', 0), upper_line.get('x2', 0)],
            [upper_line.get('y1', 0), upper_line.get('y2', 0)],
            color=color, linewidth=2.5, alpha=0.8, label=label,
            solid_capstyle='round', linestyle=linestyle,
            marker=marker, markersize=10, markeredgewidth=2, markeredgecolor='white'
        )
        
        # Plotte untere Linie mit Marker
        self.ax_main.plot(
            [lower_line.get('x1', 0), lower_line.get('x2', 0)],
            [lower_line.get('y1', 0), lower_line.get('y2', 0)],
            color=color, linewidth=2.5, alpha=0.8,
            solid_capstyle='round', linestyle=linestyle,
            marker=marker, markersize=10, markeredgewidth=2, markeredgecolor='white'
        )
        
        # Mittelpunkt für Annotation
        mid_x = (upper_line.get('x1', 0) + upper_line.get('x2', 0) + 
                lower_line.get('x1', 0) + lower_line.get('x2', 0)) / 4
        mid_y = (upper_line.get('y1', 0) + upper_line.get('y2', 0) + 
                lower_line.get('y1', 0) + lower_line.get('y2', 0)) / 4

        # Nodes Daten
        if nodes_data:
            if len(nodes_data) >= 2:
                x_nodes = [node[0] for node in nodes_data]
                y_nodes = [node[1] for node in nodes_data]

                self.ax_nodes.plot(
                    x_nodes, y_nodes,
                    color=color, linewidth=2.0, alpha=0.7,
                    marker=marker, markersize=6, markeredgewidth=1.5, markeredgecolor='white',
                    label=f'DS{dataset_index}'
                )
        # Gebe Position zurück für intelligente Annotation-Platzierung
        return {
            'x': mid_x,
            'y': mid_y,
            'angle': angle,
            'upper': vertebra_upper,
            'lower': vertebra_lower,
            'color': color,
            'marker': marker,
            'marker_unicode': marker_unicode,
            'dataset_index': dataset_index
        }
    
    def _place_annotations_smart(self, annotations_data, min_distance=100):
        """
        Platziert Annotationen intelligent um Überlappungen zu vermeiden
        
        Args:
            annotations_data: Liste von Dictionaries mit Positions- und Text-Daten
            min_distance: Minimaler Abstand zwischen Annotationen in Pixeln
        """
        if not annotations_data:
            return
        
        # Konvertiere zu numpy array für einfachere Berechnungen
        positions = np.array([[ann['x'], ann['y']] for ann in annotations_data])
        adjusted_positions = positions.copy()
        
        # Iterative Verschiebung um Überlappungen zu vermeiden
        for iteration in range(10):  # Maximal 10 Iterationen
            forces = np.zeros_like(positions)
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    diff = adjusted_positions[i] - adjusted_positions[j]
                    dist = np.linalg.norm(diff)
                    
                    if dist < min_distance and dist > 0.1:
                        # Berechne Abstoßungskraft
                        force_magnitude = (min_distance - dist) / min_distance
                        force_direction = diff / dist
                        force = force_direction * force_magnitude * 5
                        
                        forces[i] += force
                        forces[j] -= force
            
            # Aktualisiere Positionen
            adjusted_positions += forces
            
            # Prüfe ob Konvergenz erreicht
            if np.max(np.abs(forces)) < 0.5:
                break
        
        # Platziere Annotationen
        for i, ann in enumerate(annotations_data):
            original_pos = positions[i]
            final_pos = adjusted_positions[i]
            
            # Prüfe ob verschoben
            displacement = np.linalg.norm(final_pos - original_pos)
            use_arrow = displacement > 5
            
            text = f'{ann["marker_unicode"]} DS{ann["dataset_index"]} \n{ann["upper"]}-{ann["lower"]}: {ann["angle"]}°'
            
            if use_arrow:
                self.ax_main.annotate(
                    text,
                    xy=(original_pos[0], original_pos[1]),
                    xytext=(final_pos[0], final_pos[1]),
                    fontsize=9,
                    color=ann['color'],
                    weight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=ann['color'], alpha=0.95, linewidth=2),
                    arrowprops=dict(arrowstyle='->', color=ann['color'], 
                                   lw=1.5, alpha=0.7, connectionstyle='arc3,rad=0.2')
                )
            else:
                self.ax_main.annotate(
                    text,
                    xy=(original_pos[0], original_pos[1]),
                    fontsize=9,
                    color=ann['color'],
                    weight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=ann['color'], alpha=0.95, linewidth=2)
                )

    def _create_statistics_plot(self, stats_df):
        possible_cobbTypes = [i[0] for i in self.active_types.items() if i[1]]
        df_current = stats_df.loc[(stats_df['id'] == self.current_id) & (stats_df['cobbType'].isin(possible_cobbTypes))]
        current_img_type = df_current['img_type'].values[0]
        _possible_spine_regions = list(df_current['spine_region'].unique())
        possible_spine_regions = [r for r in total_regions if r in _possible_spine_regions]
        x = np.arange(len(possible_spine_regions))  # the label locations
        width = 1 / (len(possible_cobbTypes) + 1)  # the width of the bars
        multiplier = 0

        ax = self.ax_stats
        cobbStds = {}
        cobbErrors = {}
        activeVals = {}
        for cobbType in possible_cobbTypes:
            stds = []
            errs = []
            vals = []
            for region in possible_spine_regions:
                s = stats_df.loc[(stats_df['img_type'] == current_img_type) & (stats_df['cobbType'] == cobbType) & (
                            stats_df['spine_region'] == region), 'angle_std'].mean()
                stds.append(s)
                e = stats_df.loc[(stats_df['img_type'] == current_img_type) & (stats_df['cobbType'] == cobbType) & (
                            stats_df['spine_region'] == region), 'angle_std'].std()
                errs.append(e)
                t = df_current.loc[
                    (df_current['cobbType'] == cobbType) & (df_current['spine_region'] == region), 'angle_std'].mean()
                vals.append(t)
            activeVals[cobbType] = vals
            cobbStds[cobbType] = stds
            cobbErrors[cobbType] = errs

        for (attribute, measurement), (_, error), (_, current_measurement) in zip(cobbStds.items(), cobbErrors.items(),
                                                                                  activeVals.items()):
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, color=self.type_color_map.get(attribute, 'gray'), yerr=error, label=attribute)
            ax.errorbar(x + offset, current_measurement, fmt='o', color='black', capsize=5)
            multiplier += 1

        ax.set_xticks(x + width, possible_spine_regions)
        ax.legend(loc='upper left', ncols=3)


    def update_plot(self):
        """Aktualisiert den Hauptplot"""
        self.ax_main.clear()
        self.ax_nodes.clear()
        self.ax_stats.clear()
        # self.ax_info.clear()
        # self.ax_info.axis('off')

        # ERSETZE die bisherige Scatterplot-Erstellung durch:
        try:
            # Lade Statistik-Daten (falls vorhanden)
            stats_path = Path('data/df6g_angle_stats_by_spine_region.json')
            if stats_path.exists():
                stats_df = pd.read_json(stats_path)
                self._create_statistics_plot(stats_df)
            else:
                print(f"⚠ Statistik-Datei nicht gefunden: {stats_path}")
        except Exception as e:
            print(f"⚠ Statistik-Plot konnte nicht erstellt werden: {e}")
            import traceback
            traceback.print_exc()

        filtered_data = self.get_filtered_data()
        
        # Ermittle Dataset-Counts pro Typ
        dataset_counts_per_type = self.get_dataset_counts_per_type()
        
        # Ermittle verfügbare Dataset-Indizes für diese ID
        max_dataset_idx = self._get_max_dataset_index_for_id(self.current_id)
        total_datasets = max_dataset_idx + 1
        
        print(f"\n{'='*60}")
        print(f"Image ID: {self.current_id} ({self.current_id_index + 1}/{len(self.available_ids)})")
        print(f"Verfügbare Datasets: 0-{max_dataset_idx} (insgesamt {total_datasets})")
        print(f"Angezeigte Messungen: {len(filtered_data)}")
        
        # Detaillierte Ausgabe pro Typ und Dataset
        for cobb_type, dataset_dict in dataset_counts_per_type.items():
            print(f"\n  {cobb_type}:")
            for ds_idx in sorted(dataset_dict.keys()):
                count = dataset_dict[ds_idx]
                print(f"    Dataset {ds_idx}: {count} Messungen")
        
        if len(filtered_data) == 0:
            self.ax_main.text(0.5, 0.5, 'Keine Messungen für diese Auswahl',
                            ha='center', va='center', fontsize=16,
                            transform=self.ax_main.transAxes)
            self.fig.canvas.draw()
            return
        
        # Sammle Annotations für intelligente Platzierung
        annotations_data = []
        plotted_labels = set()
        
        for idx, row in filtered_data.iterrows():
            angle_data = row['angle_data']
            nodes_data = row.get('nodes', [])
            cobb_type = angle_data.get('cobbType', 'Unknown')
            dataset_idx = int(row['measure_dataset_index'])
            
            # Label nur einmal pro Typ
            label = cobb_type if cobb_type not in plotted_labels else None
            if label:
                plotted_labels.add(cobb_type)
            
            color = self.type_color_map.get(cobb_type, 'gray')
            
            # Plotte und sammle Annotation-Daten
            ann_data = self.plot_measurement(angle_data, color, label, dataset_idx,nodes_data)
            if ann_data:
                annotations_data.append(ann_data)

        # Platziere Annotationen intelligent
        self._place_annotations_smart(annotations_data)
        
        # Formatierung
        self.ax_main.set_xlabel('X [px]', fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel('Y [px]', fontsize=12, fontweight='bold')

        title_suffix = ""
        if self.selected_dataset_indices != 'all':
            title_suffix = f" - Dataset {self.selected_dataset_indices[0]}"
        
        self.ax_main.set_title(
            f'Wirbelsäulenmessungen - Image ID: {self.current_id}{title_suffix}, ({self.df_keys.loc[self.current_id,"img_type"]})',
            fontsize=14, fontweight='bold', pad=15
        )
        
        # Erweiterte Legende mit Markern
        if plotted_labels:
            legend_elements = []
            for label in plotted_labels:
                color = self.type_color_map.get(label, 'gray')
                legend_elements.append(Line2D([0], [0], color=color, lw=3, label=label))
            
            legend = self.ax_main.legend(handles=legend_elements, loc='lower left',bbox_to_anchor=(1.05, 0.2),
                                        fontsize=10, framealpha=0.9)
            legend.get_frame().set_linewidth(2)

        self.ax_main.sharey(self.ax_nodes)

        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.invert_yaxis()
        self.ax_main.set_aspect('equal', adjustable='box', share=True)

        self.ax_nodes.grid(True, alpha=0.3, linestyle='--')

        # Info-Text mit Dataset-Counts
        info_lines = [
            f"Angezeigte Datasets: {len(filtered_data)} / {total_datasets}; Messungen (0-{max_dataset_idx})",
            ""
        ]
        
        for cobb_type, dataset_dict in dataset_counts_per_type.items():
            if self.active_types[cobb_type]:
                total_for_type = sum(dataset_dict.values())
                info_lines.append(f"{cobb_type}: {total_for_type}")
                for ds_idx in sorted(dataset_dict.keys()):
                    info_lines.append(f"  DS{ds_idx}: {dataset_dict[ds_idx]}")
        
        info_lines.extend([
            "",
            f"Bildtyp: {filtered_data['img_type'].iloc[0]}",
            "",
            "Shortcuts:",
            "  ← → : Navigation",
            "  0-9 : Dataset wählen",
            "  A : Alle Datasets",
            "  Q : Beenden"
        ])
        #
        # self.ax_info.text(0.05, 0.95, '\n'.join(info_lines),
        #                  fontsize=8, verticalalignment='top',
        #                  family='monospace',
        #                  bbox=dict(boxstyle='round,pad=0.5',
        #                           facecolor='lightyellow', alpha=0.8))
        
        print('Image UUID: %s'%(self.df_keys.loc[self.current_id,'uuid']))
        print(link(self.df_keys.loc[self.current_id,'img']))
        print(f'https://xray-annotator.renecol.org/?relayto={self.df_keys.loc[self.current_id,"uuid"]}')
        
        self.fig.canvas.draw()
    
    def run(self):
        """Startet den Viewer"""
        print("\n" + "="*60)
        print("Wirbelsäulenmessungs-Viewer gestartet")
        print("="*60)
        print("\nTastatur-Shortcuts:")
        print("  ← → oder PageUp/PageDown : Navigation zwischen IDs")
        print("  Home / End                : Erste / Letzte ID")
        print("  A                         : Alle Datasets anzeigen")
        print("  0-9                       : Einzelnes Dataset auswählen")
        print("  Q oder Escape             : Beenden")
        print("="*60 + "\n")
        
        self.setup_ui()
        self.update_plot()
        plt.show()


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(
        description='Interaktiver Viewer für Wirbelsäulenmessungen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python spine_viewer.py
  python spine_viewer.py --data /pfad/zu/daten.json --keys /pfad/zu/keys.json
  python spine_viewer.py --id 480
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        default='data/df5c_with_reconstructed_angles_all.json',
        help='Pfad zur JSON-Datei mit Messdaten'
    )

    parser.add_argument(
        '--keys',
        type=str,
        default='data/df_keys.json',
        help='Pfad zur JSON-Datei mit Keys'
    )
    
    parser.add_argument(
        '--id',
        type=int,
        help='Start-ID (optional)'
    )
    
    args = parser.parse_args()
    
    # Prüfe ob Datei existiert
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ Fehler: Datei nicht gefunden: {data_path}")
        print(f"  Aktuelles Verzeichnis: {Path.cwd()}")
        sys.exit(1)

    # Prüfe ob Datei existiert
    keys_path = Path(args.keys)
    if not keys_path.exists():
        print(f"✗ Fehler: Datei nicht gefunden: {keys_path}")
        print(f"  Aktuelles Verzeichnis: {Path.cwd()}")
        sys.exit(1)
    # Erstelle und starte Viewer
    viewer = SpineViewer(str(data_path),str(keys_path))
    
    # Setze Start-ID falls angegeben
    if args.id is not None:
        if args.id in viewer.available_ids:
            viewer.current_id = args.id
            viewer.current_id_index = viewer.available_ids.index(args.id)
            print(f"✓ Starte mit ID: {args.id}")
        else:
            print(f"⚠ ID {args.id} nicht gefunden, verwende erste ID")
    
    viewer.run()


if __name__ == '__main__':
    main()