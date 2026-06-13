"""
B-Spline interpolation and spine analysis utilities.
Converted from TypeScript to Python.
"""

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from dotmap import DotMap

Point = Tuple[float, float]

spine_labels = ['C0','C1','C2','C3','C4','C5','C6','C7','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5','S1']
spine_regions = {'CT':[0,9],'PT': [10,12],'MT':[13,18],'TL':[19,20],'LU':[21,25]}


def smooth_nodes(nodes: List[Point], sigma: float) -> List[Point]:
    """Gaussian smoothing of control-point (x,y) positions before B-spline fitting.

    mode='nearest' clamps the signal at boundaries so endpoint nodes are pulled
    toward their immediate neighbours rather than reflected, which is the desired
    behaviour for stabilising endpoint tangents.
    """
    arr = np.array(nodes, dtype=float)
    smoothed_x = gaussian_filter1d(arr[:, 0], sigma=sigma, mode='nearest')
    smoothed_y = gaussian_filter1d(arr[:, 1], sigma=sigma, mode='nearest')
    return list(zip(smoothed_x.tolist(), smoothed_y.tolist()))

def interpolate_bspline(points: List[Point], num_steps: int = 10) -> List[Point]:
    """
    Interpolates an array of points using B-Spline interpolation.
    
    Args:
        points: Array of points [(x1, y1), (x2, y2), ..., (xn, yn)]
        num_steps: Number of interpolated steps between each point
        
    Returns:
        Interpolated points

    """
    if len(points) < 3:
        return []
    
    # Separate x and y coordinates
    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    
    # Create parameter t for parametric curve
    t = np.linspace(0, 1, len(points))

    # estimate endpoint derivatives (dx/dt) with forward/backward differences
    if len(x_coords) >= 2:
        dxdt0 = (x_coords[1] - x_coords[0]) / (t[1] - t[0])
        dxdtN = (x_coords[-1] - x_coords[-2]) / (t[-1] - t[-2])
    else:
        dxdt0 = dxdtN = 0.0

    # Create B-spline for x and y separately (degree 3 = cubic spline)
    spl_x = make_interp_spline(t, x_coords, k=3,  bc_type=([(1, dxdt0)], [(1, dxdtN)]))
    spl_y = PchipInterpolator(t, y_coords)
    
    # Generate interpolation points
    num_total_steps = num_steps * len(points)
    t_new = np.linspace(0, 1, num_total_steps)
    
    # Evaluate spline at new points
    x_new = spl_x(t_new)
    y_new = spl_y(t_new)
    
    # Combine into list of tuples
    result = [(float(x_new[i]), float(y_new[i])) for i in range(len(x_new))]
    
    return result


@dataclass
class TangentRecord:
    splineindex: int
    line: Dict[str, float]  # {x1, y1, x2, y2}
    angle: float


def calculate_tangents(pts: List[Point], at_indices: List[int], length: float) -> List[TangentRecord]:
    """
    Calculate tangent lines at each point along the curve.
    
    Args:
        pts: List of points
        at_indices: Indices where to calculate tangents, use all if empty
        length: Length of tangent lines
        
    Returns:
        List of tangent records
    """
    tangents = []
    # Rotate points: swap and negate y
    points = [(-y, x) for x, y in pts]
    if len(at_indices)==0:
        at_indices = list(range(len(points)))
    for i in at_indices:
        if i == 0:
            # Forward difference for the first point
            direction = [
                points[i + 1][0] - points[i][0],
                points[i + 1][1] - points[i][1]
            ]
        elif i == len(points) - 1:
            # Backward difference for the last point
            direction = [
                points[i][0] - points[i - 1][0],
                points[i][1] - points[i - 1][1]
            ]
        else:
            # Central difference for the inner points
            direction = [
                points[i + 1][0] - points[i - 1][0],
                points[i + 1][1] - points[i - 1][1]
            ]


        # Normalize the direction vector
        magnitude = np.sqrt(direction[0]**2 + direction[1]**2)
        direction = [direction[0] / magnitude, direction[1] / magnitude]
        
        # Calculate start and end points of the tangent line
        start = [
            pts[i][0] - (length / 2) * direction[0],
            pts[i][1] - (length / 2) * direction[1]
        ]
        end = [
            pts[i][0] + (length / 2) * direction[0],
            pts[i][1] + (length / 2) * direction[1]
        ]
        
        # Calculate angle between tangent and Y axis
        angle = np.arctan2(direction[0], abs(direction[1])) * (180 / np.pi)
        
        tangents.append(TangentRecord(
            splineindex=i,
            line={'x1': start[0], 'y1': start[1], 'x2': end[0], 'y2': end[1]},
            angle=angle
        ))
    
    return tangents


def get_perpendicular_lines_at_indices(
    spline: List[Tuple[float, float]], 
    length: float, 
    indices: List[int]
) -> List[Dict[str, float]]:
    """
    Get perpendicular lines at specified indices of a spline.
    
    Args:
        spline: List of spline points
        length: Length of perpendicular lines
        indices: Indices where to create perpendicular lines
        
    Returns:
        List of line dictionaries with x1, y1, x2, y2
    """
    lines = []
    
    if spline and len(spline) > 2:
        for i in indices:
            if i + 1 > len(spline):
                continue

            if i==0:
                # Forward difference for the first point
                x1, y1 = spline[i]
                x2, y2 = spline[i + 1]
            elif i == len(spline) - 1:
                # Backward difference for the last point
                x1, y1 = spline[i - 1]
                x2, y2 = spline[i]
            else:
                # Central difference for inner points
                x1, y1 = spline[i - 1]
                x2, y2 = spline[i+1]
            #x1, y1 = spline[i]
            #x2, y2 = spline[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            magnitude = np.sqrt(dx * dx + dy * dy)
            
            if magnitude == 0:
                continue
                
            perp_dx = -dy / magnitude
            perp_dy = dx / magnitude
            
            lines.append({
                'x1': spline[i][0] - perp_dx * (length / 2),
                'y1': spline[i][1] - perp_dy * (length / 2),
                'x2': spline[i][0] + perp_dx * (length / 2),
                'y2': spline[i][1] + perp_dy * (length / 2)
            })
    
    return lines


@dataclass
class Line:
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    mode: Optional[str] = None


def make_perpendicular_line_at_line_center(line: Line, length: float) -> Line:
    """
    Create a perpendicular line at the center of a given line.
    
    Args:
        line: Input line
        length: Length of perpendicular line
        
    Returns:
        Perpendicular line
    """
    if not all([line.x1, line.y1, line.x2, line.y2]):
        return line
    
    mid_x = (line.x1 + line.x2) / 2
    mid_y = (line.y1 + line.y2) / 2
    
    dx = line.x2 - line.x1
    dy = line.y2 - line.y1
    magnitude = np.sqrt(dx * dx + dy * dy)
    
    if magnitude == 0:
        return line
    
    perp_dx = -dy / magnitude
    perp_dy = dx / magnitude
    
    return Line(
        x1=mid_x - perp_dx * length / 2,
        y1=mid_y - perp_dy * length / 2,
        x2=mid_x + perp_dx * length / 2,
        y2=mid_y + perp_dy * length / 2
    )


def is_point_above_line(point: Dict[str, float], line: Line) -> bool:
    """
    Check if a point is above an infinite line extended from the given line segment.
    
    Args:
        point: Point dictionary with 'x' and 'y'
        line: Line object
        
    Returns:
        True if point is above the line
    """
    if not all([line.x1, line.y1, line.x2, line.y2]):
        return False
    
    x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
    
    if x2 == x1:
        # Vertical line case
        return point['x'] < x1
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    line_y_at_point_x = slope * point['x'] + intercept
    
    return point['y'] < line_y_at_point_x


@dataclass
class NodeItem:
    x: float
    y: float
    type: Optional[str] = None
    radius: Optional[float] = None
    fill: Optional[str] = None


def calculate_angle(line1: Line, line2: Line) -> int:
    """
    Calculate the angle between two lines.
    
    Args:
        line1: First line
        line2: Second line
        
    Returns:
        Angle in degrees (0-90)
    """
    angle1 = np.arctan2(
        (line1.y2 or 0) - (line1.y1 or 0),
        (line1.x2 or 0) - (line1.x1 or 0)
    )
    angle2 = np.arctan2(
        (line2.y2 or 0) - (line2.y1 or 0),
        (line2.x2 or 0) - (line2.x1 or 0)
    )
    
    angle = (angle2 - angle1) * (180 / np.pi)
    
    # Normalize the angle to be between 0 and 180 degrees
    angle = (angle + 360) % 180
    if angle > 90:
        angle = 180 - angle
    
    return int(np.floor(angle))


def find_tangent_extrema_indices(tangent_angles: List[float], smoothing_window: int = 15) -> List[int]:
    """Find both local maxima and minima of tangent angles (Richtungswechsel).

    Applies a running-average smoothing before peak detection to suppress
    numerical noise from the dense spline discretisation.

    Args:
        tangent_angles: Tangent angles at every spline point.
        smoothing_window: Width of the running-average kernel (set to 1 to disable).

    Returns:
        Sorted list of indices where local maxima or minima occur.
    """
    angles = np.array(tangent_angles, dtype=float)
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        angles = np.convolve(angles, kernel, mode='same')
    result = []
    for i in range(1, len(angles) - 1):
        dt1 = angles[i] - angles[i - 1]
        dt2 = angles[i + 1] - angles[i]
        if (dt1 > 0 and dt2 < 0) or (dt1 < 0 and dt2 > 0):
            result.append(i)
    return result


def get_local_max_tan_indices_of_splines(tangent_angles: List[float]) -> List[int]:
    """
    Find local maxima indices in tangent angles.
    
    Args:
        tangent_angles: List of tangent angles
        
    Returns:
        List of indices where local maxima occur
    """
    max_indices = []
    
    for i in range(1, len(tangent_angles) - 1):
        prev = tangent_angles[i - 1]
        curr = tangent_angles[i]
        next_val = tangent_angles[i + 1]
        
        dt1 = curr - prev
        dt2 = next_val - curr
        
        if dt1 > 0 and dt2 < 0:  # exclude small fluctuations
            max_indices.append(i)

    return max_indices


def get_local_extrema_indices_of_spline(points: List[Tuple[float, float]]) -> List[int]:
    """
    Find local extrema (minima and maxima) indices in a spline.
    
    Args:
        points: List of spline points
        
    Returns:
        List of indices where extrema occur
    """
    # Rotate points: swap and negate y
    rotated_points = [(-y, x) for x, y in points]
    extrema_indices = [0]
    
    for i in range(1, len(rotated_points) - 1):
        x_prev, y_prev = rotated_points[i - 1]
        x_curr, y_curr = rotated_points[i]
        x_next, y_next = rotated_points[i + 1]
        
        dy1 = y_curr - y_prev
        dy2 = y_next - y_curr
        
        if dy1 > 0 and dy2 < 0:
            extrema_indices.append(i)
        
        if dy1 < 0 and dy2 > 0:
            extrema_indices.append(i)
    
    extrema_indices.append(len(rotated_points) - 1)
    
    return extrema_indices


@dataclass
class MappedNodesRecord:
    x: float
    y: float
    spline_idx: int
    label: str


@dataclass
class CobbAngleType:
    vertebra_upper: str
    vertebra_lower: str
    angle: float


@dataclass
class CobbMeasurement:
    upperLine: Line
    lowerLine: Line
    lineColor: str
    cobbAngle: CobbAngleType
    cobbType: str


@dataclass
class GlobalSpineParameters:
    hipcenter_s1_distance_x_raw: Optional[float] = None
    hipcenter_s1_distance_x_relative: Optional[float] = None
    hipcenter_c7_distance_x_raw: Optional[float] = None
    hipcenter_c7_distance_x_relative: Optional[float] = None
    hipline_angle_to_ground: Optional[float] = None
    pelvic_incidence: Optional[float] = None


def map_nodes_to_spline_and_keys(
    nodes: List[NodeItem], 
    keys: List[str], 
    spline: List[Point]
) -> List[MappedNodesRecord]:
    """
    Map nodes to their closest points on a spline with labels.
    
    Args:
        nodes: List of node items
        keys: List of label strings
        spline: List of spline points
        
    Returns:
        List of mapped node records
    """
    int_spline_y = [int(np.floor(y)) for x, y in spline]
    result = []
    
    for i, n in enumerate(nodes):
        # Get the closest match of node in the spline
        goal = n.y
        output = min(int_spline_y, key=lambda curr: abs(curr - goal))
        idx = int_spline_y.index(output)
        
        label = keys[i] if i < len(keys) else '?'
        result.append(MappedNodesRecord(
            x=n.x,
            y=n.y,
            spline_idx=idx,
            label=label
        ))
    
    return result

def convertMeasurementBorderToSpineRegion(upper_vertebra, lower_vertebra):
    # returns the spine region whose given range is most contained (compared to the other regions) in the range given by upper and lower vertebra
    # convert upper and lower vertebra labels to indices
    upper_vertebra_idx = spine_labels.index(upper_vertebra)
    try:
        lower_vertebra_idx = spine_labels.index(lower_vertebra)
    except ValueError:
        lower_vertebra_idx = len(spine_labels) - 1
    max_overlap = 0
    selected_region = None
    for region, (start_idx, end_idx) in spine_regions.items():
        overlap_start = max(upper_vertebra_idx, start_idx)
        overlap_end = min(lower_vertebra_idx, end_idx)
        overlap = max(0, overlap_end - overlap_start + 1)
        if overlap > max_overlap:
            max_overlap = overlap
            selected_region = region
    return selected_region

def get_closest_node_idx_from_y(nodes: List[NodeItem], y: Optional[float]) -> int:
    """
    Get the index of the node closest to a given y-coordinate.
    
    Args:
        nodes: List of nodes
        y: Target y-coordinate
        
    Returns:
        Index of closest node, or -1 if y is None
    """
    if y is None:
        return -1
    
    node_ys = [n.y for n in nodes]
    output = min(node_ys, key=lambda curr: abs(curr - y))
    idx = node_ys.index(output)
    
    return idx


def get_closest_before_and_after(
    nodes: List[NodeItem], 
    goal_upper: Optional[float], 
    goal_lower: Optional[float]
) -> Dict[str, int]:
    """
    Find indices of nodes closest to upper and lower goals.
    
    Args:
        nodes: List of nodes
        goal_upper: Upper boundary y-coordinate
        goal_lower: Lower boundary y-coordinate
        
    Returns:
        Dictionary with 'before' and 'after' indices
    """
    before = None
    after = None
    
    if not goal_upper or not goal_lower:
        return {'before': 0, 'after': len(nodes) - 1}
    
    for node in nodes:
        if node.y <= goal_upper:
            if not before or goal_upper - node.y < goal_upper - before.y:
                before = node
        if node.y >= goal_lower:
            if not after or node.y - goal_lower < after.y - goal_lower:
                after = node
    
    if not before:
        i = get_closest_node_idx_from_y(nodes, goal_upper)
        before = nodes[i] if i > -1 else nodes[0]
    
    if not after:
        i = get_closest_node_idx_from_y(nodes, goal_lower)
        after = nodes[i] if i > -1 else nodes[len(nodes) - 1]
    
    return {
        'before': nodes.index(before),
        'after': nodes.index(after)
    }


def generate_COBB_angles_from_tangents(tangents, mapped, perp_lines, mode='regional_L5', angle_threshold=10):
    """
    Generate COBB angles from tangent lines at specified indices.
    :param tangents: tangent angles calculated from spline
    :param mapped: mapped to vertebra labels
    :param perp_lines: perpendicular lines at tangent indices
    :param mode: terminating at either L5 or S1 as regional COBB angles or segmental
    :param angle_threshold: reporting threshold for regional COBB angles, threshol for segmental is 0
    :return: record with from_index, to_index as the positions in the spline; from_label, to_label, cobb_angle
    """
    cobb_angles = []
    max_tan_indices = get_local_max_tan_indices_of_splines([t.angle for t in tangents])
    max_tan_spline_indices = []
    if mode=='regional_L5':
        # Filter to only include maxima around L5
        l5_index = next((i for i, m in enumerate(mapped) if m.label == 'L5'), None)
        if l5_index is not None:
            max_tan_spline_indices = [0, *max_tan_indices, l5_index]
        else:
            max_tan_spline_indices = [0, *max_tan_indices, len(mapped)-1]
    if mode=='regional_S1':
        # Filter to only include maxima around S1
        s1_index = next((i for i, m in enumerate(mapped) if m.label == 'S1'), None)
        if s1_index is not None:
            max_tan_spline_indices = [0, *max_tan_indices, s1_index]
        else:
            max_tan_spline_indices = [0, *max_tan_indices, len(mapped)-1]
    if mode=='segmental':
        max_tan_spline_indices = list(range(len(mapped)))

    if len(max_tan_spline_indices) < 2:
        print("Nicht genügend Extrempunkte für Cobb-Winkelberechnung gefunden.")
        return cobb_angles

    max_tan_labels = [mapped[i].label for i in max_tan_spline_indices]

    if mode=='segmental':
        angle_threshold=0

    for i in range(len(max_tan_spline_indices) - 1):
        idx_start = max_tan_spline_indices[i]
        idx_end = max_tan_spline_indices[i + 1]

        pl1 = DotMap(perp_lines[idx_start])
        pl2 = DotMap(perp_lines[idx_end])

        p_diff = calculate_angle(pl1, pl2)
        if p_diff>=angle_threshold:  # Nur Winkel über 10 Grad berücksichtigen
            cobb_angles.append({
                'from_index': idx_start,
                'to_index': idx_end,
                'from_label': max_tan_labels[i],
                'to_label': max_tan_labels[i + 1],
                'upperLine': pl1.toDict(),
                'lowerLine': pl2.toDict(),
                'lineColor': 'red',
                'cobb_angle': p_diff
            })
    return cobb_angles

def plot_spline_and_lines(
    original_points: Optional[List[Point]] = None,
    interpolated_points: Optional[List[Point]] = None,
    lines: Optional[List[Dict[str, float]]] = None,
    tangents: Optional[List[TangentRecord]] = None,
    nodes: Optional[List[NodeItem]] = None,
    mapped_nodes: Optional[List[MappedNodesRecord]] = None,
    show_angles: bool = True,
    invert_yaxis: bool = False,
    title: str = "B-Spline Visualization",
    figsize: Tuple[int, int] = (12, 8),
    show_grid: bool = True,
        fig = None, ax = None
):
    """
    Plot interpolated spline points and various lines with different colors.
    
    Args:
        original_points: Original control points (plotted as red dots)
        interpolated_points: Interpolated spline points (plotted as blue line)
        lines: List of line dictionaries with x1, y1, x2, y2 (plotted as green lines)
        tangents: List of TangentRecord objects (plotted as orange lines)
        nodes: List of NodeItem objects (plotted as purple circles)
        mapped_nodes: List of MappedNodesRecord with labels for annotation
        show_angles: Whether to show angles between consecutive perpendicular lines
        invert_yaxis: Whether to flip the y-axis (useful for images/medical scans)
        title: Plot title
        figsize: Figure size (width, height)
        show_grid: Whether to show grid
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot interpolated spline
    if interpolated_points:
        x_interp = [p[0] for p in interpolated_points]
        y_interp = [p[1] for p in interpolated_points]
        ax.plot(x_interp, y_interp, 'b-', linewidth=2, label='Interpolated Spline', alpha=0.7)
    
    # Plot original control points
    if original_points:
        x_orig = [p[0] for p in original_points]
        y_orig = [p[1] for p in original_points]
        ax.plot(x_orig, y_orig, 'ro', markersize=8, label='Original Points', zorder=5)
    
    # Plot perpendicular lines
    if lines:
        for line in lines:
            ax.plot(
                [line['x1'], line['x2']], 
                [line['y1'], line['y2']], 
                'g-', 
                linewidth=1.5, 
                alpha=0.6
            )
        # Add single legend entry for all lines
        ax.plot([], [], 'g-', linewidth=1.5, alpha=0.6, label='Perpendicular Lines')
        
        # Calculate and display angles between consecutive lines
        if show_angles and len(lines) > 1:
            for i in range(len(lines) - 1):
                line1 = Line(
                    x1=lines[i].get('x1'),
                    y1=lines[i].get('y1'),
                    x2=lines[i].get('x2'),
                    y2=lines[i].get('y2')
                )
                line2 = Line(
                    x1=lines[i + 1].get('x1'),
                    y1=lines[i + 1].get('y1'),
                    x2=lines[i + 1].get('x2'),
                    y2=lines[i + 1].get('y2')
                )
                angle = calculate_angle(line1, line2)
                
                # Place angle text between the two lines
                mid_x = (lines[i]['x2'] + lines[i + 1]['x2']) / 2
                mid_y = (lines[i]['y2'] + lines[i + 1]['y2']) / 2
                
                ax.annotate(
                    f'{angle}°',
                    xy=(mid_x, mid_y),
                    fontsize=9,
                    color='darkgreen',
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7)
                )
    
    # Plot tangent lines
    if tangents:
        for tangent in tangents:
            line = tangent.line
            ax.plot(
                [line['x1'], line['x2']], 
                [line['y1'], line['y2']], 
                'orange', 
                linewidth=1.5, 
                alpha=0.6
            )
        # Add single legend entry for all tangents
        ax.plot([], [], 'orange', linewidth=1.5, alpha=0.6, label='Tangent Lines')
    
    # Plot nodes
    if nodes:
        for node in nodes:
            circle = plt.Circle(
                (node.x, node.y), 
                node.radius if node.radius else 5, 
                color=node.fill if node.fill else 'purple', 
                alpha=0.5,
                zorder=4
            )
            ax.add_patch(circle)
        # Add legend entry
        ax.plot([], [], 'o', color='purple', markersize=10, alpha=0.5, label='Nodes')
    
    # Add labels from mapped_nodes
    if mapped_nodes:
        for mapped_node in mapped_nodes:
            ax.annotate(
                mapped_node.label,
                xy=(mapped_node.x, mapped_node.y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                color='darkblue',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='darkblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkblue', lw=1.5)
            )
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.axis('equal')
    
    if invert_yaxis:
        ax.invert_yaxis()
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_spline_and_lines_multiple(
    datasets: list,
    show_angles: bool = True,
    invert_yaxis: bool = False,
    title: str = "B-Spline Visualization",
    figsize: tuple = (12, 8),
    show_grid: bool = True,
    fig=None,
    ax=None
):
    """
    Plot multiple datasets into a single plot. Each dataset is a dict with optional keys:
      - original_points: List of (x,y)
      - interpolated_points: List of (x,y)
      - lines: List[Dict] with keys x1,y1,x2,y2
      - tangents: List[TangentRecord] (expects .line dict with x1,y1,x2,y2)
      - nodes: List[NodeItem] (expects .x, .y, optional .radius, .fill)
      - mapped_nodes: List[MappedNodesRecord] (expects .x, .y, .label)
      - label: str (dataset label for legend)
      - color: matplotlib color (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import matplotlib.cm as cm
    except Exception as e:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Generate a color cycle if not provided
    cmap = cm.get_cmap('tab10')
    n = max(1, len(datasets))
    color_cycle = [cmap(i % cmap.N) for i in range(n)]

    # Track whether we've added global legend entries for special types
    added_perp_legend = False
    added_tangent_legend = False
    added_nodes_legend = False

    for idx, ds in enumerate(datasets):
        color = ds.get('color', color_cycle[idx])
        label = ds.get('label', f"dataset_{idx+1}")

        # interpolated spline
        interp = ds.get('interpolated_points')
        if interp:
            x_interp = [p[0] for p in interp]
            y_interp = [p[1] for p in interp]
            ax.plot(x_interp, y_interp, '-', color=color, linewidth=2, label=label, alpha=0.9)

        # original control points (smaller, same color edge)
        orig = ds.get('original_points')
        if orig:
            x_orig = [p[0] for p in orig]
            y_orig = [p[1] for p in orig]
            ax.plot(x_orig, y_orig, 'o', markersize=6, markeredgecolor=color, markerfacecolor='white', zorder=5, alpha=0.9)

        # perpendicular lines
        lines = ds.get('lines')
        if lines:
            for line in lines:
                ax.plot([line['x1'], line['x2']], [line['y1'], line['y2']], '-', color=color, linewidth=1.2, alpha=0.6)
            if not added_perp_legend:
                # proxy for legend
                ax.plot([], [], '-', color=color, linewidth=1.2, alpha=0.6, label='Perpendicular Lines')
                added_perp_legend = True

            # show angles between consecutive lines (per dataset)
            if show_angles and len(lines) > 1:
                for i in range(len(lines) - 1):
                    # expects Line and calculate_angle available in scope
                    line1 = Line(x1=lines[i].get('x1'), y1=lines[i].get('y1'), x2=lines[i].get('x2'), y2=lines[i].get('y2'))
                    line2 = Line(x1=lines[i+1].get('x1'), y1=lines[i+1].get('y1'), x2=lines[i+1].get('x2'), y2=lines[i+1].get('y2'))
                    angle = calculate_angle(line1, line2)
                    mid_x = (lines[i]['x2'] + lines[i+1]['x2']) / 2
                    mid_y = (lines[i]['y2'] + lines[i+1]['y2']) / 2
                    ax.annotate(f'{angle}°', xy=(mid_x, mid_y), fontsize=8, color=color, ha='center',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.7))

        # tangents
        tangents = ds.get('tangents')
        if tangents:
            for tangent in tangents:
                line = tangent.line
                ax.plot([line['x1'], line['x2']], [line['y1'], line['y2']], '--', color=color, linewidth=1.2, alpha=0.7)
            if not added_tangent_legend:
                ax.plot([], [], '--', color='gray', linewidth=1.2, alpha=0.7, label='Tangent Lines')
                added_tangent_legend = True

        # nodes as circles
        nodes = ds.get('nodes')
        if nodes:
            for node in nodes:
                circ = plt.Circle((node.x, node.y), node.radius if getattr(node, 'radius', None) else 4,
                                  color=(node.fill if getattr(node, 'fill', None) else color), alpha=0.4, zorder=4)
                ax.add_patch(circ)
            if not added_nodes_legend:
                ax.plot([], [], 'o', color=color, markersize=8, alpha=0.5, label='Nodes')
                added_nodes_legend = True

        # mapped node labels
        mapped_nodes = ds.get('mapped_nodes')
        if mapped_nodes:
            for mn in mapped_nodes:
                ax.annotate(mn.label, xy=(mn.x, mn.y), xytext=(6, 6), textcoords='offset points',
                            fontsize=9, color=color, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color, lw=1))

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')

    ax.invert_yaxis()
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return fig, ax
def save_plot_to_file(
    filename: str,
    original_points: Optional[List[Point]] = None,
    interpolated_points: Optional[List[Point]] = None,
    lines: Optional[List[Dict[str, float]]] = None,
    tangents: Optional[List[TangentRecord]] = None,
    nodes: Optional[List[NodeItem]] = None,
    mapped_nodes: Optional[List[MappedNodesRecord]] = None,
    show_angles: bool = True,
    invert_yaxis: bool = False,
    title: str = "B-Spline Visualization",
    figsize: Tuple[int, int] = (12, 8),
    show_grid: bool = True,
    dpi: int = 300
):
    """
    Save spline and lines plot to a file.
    
    Args:
        filename: Output filename (e.g., 'plot.png', 'plot.pdf', 'plot.svg')
        original_points: Original control points
        interpolated_points: Interpolated spline points
        lines: List of line dictionaries
        tangents: List of TangentRecord objects
        nodes: List of NodeItem objects
        mapped_nodes: List of MappedNodesRecord with labels for annotation
        show_angles: Whether to show angles between consecutive perpendicular lines
        title: Plot title
        figsize: Figure size (width, height)
        show_grid: Whether to show grid
        dpi: Resolution for raster formats (PNG, JPG)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if interpolated_points:
        x_interp = [p[0] for p in interpolated_points]
        y_interp = [p[1] for p in interpolated_points]
        ax.plot(x_interp, y_interp, 'b-', linewidth=2, label='Interpolated Spline', alpha=0.7)
    
    if original_points:
        x_orig = [p[0] for p in original_points]
        y_orig = [p[1] for p in original_points]
        ax.plot(x_orig, y_orig, 'ro', markersize=8, label='Original Points', zorder=5)
    
    if lines:
        for line in lines:
            ax.plot(
                [line['x1'], line['x2']], 
                [line['y1'], line['y2']], 
                'g-', 
                linewidth=1.5, 
                alpha=0.6
            )
        ax.plot([], [], 'g-', linewidth=1.5, alpha=0.6, label='Perpendicular Lines')
        
        # Calculate and display angles between consecutive lines
        if show_angles and len(lines) > 1:
            for i in range(len(lines) - 1):
                line1 = Line(
                    x1=lines[i].get('x1'),
                    y1=lines[i].get('y1'),
                    x2=lines[i].get('x2'),
                    y2=lines[i].get('y2')
                )
                line2 = Line(
                    x1=lines[i + 1].get('x1'),
                    y1=lines[i + 1].get('y1'),
                    x2=lines[i + 1].get('x2'),
                    y2=lines[i + 1].get('y2')
                )
                angle = calculate_angle(line1, line2)
                
                # Place angle text between the two lines
                mid_x = (lines[i]['x2'] + lines[i + 1]['x2']) / 2
                mid_y = (lines[i]['y2'] + lines[i + 1]['y2']) / 2
                
                ax.annotate(
                    f'{angle}°',
                    xy=(mid_x, mid_y),
                    fontsize=9,
                    color='darkgreen',
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7)
                )
    
    if tangents:
        for tangent in tangents:
            line = tangent.line
            ax.plot(
                [line['x1'], line['x2']], 
                [line['y1'], line['y2']], 
                'orange', 
                linewidth=1.5, 
                alpha=0.6
            )
        ax.plot([], [], 'orange', linewidth=1.5, alpha=0.6, label='Tangent Lines')
    
    if nodes:
        for node in nodes:
            circle = plt.Circle(
                (node.x, node.y), 
                node.radius if node.radius else 5, 
                color=node.fill if node.fill else 'purple', 
                alpha=0.5,
                zorder=4
            )
            ax.add_patch(circle)
        ax.plot([], [], 'o', color='purple', markersize=10, alpha=0.5, label='Nodes')
    
    # Add labels from mapped_nodes
    if mapped_nodes:
        for mapped_node in mapped_nodes:
            ax.annotate(
                mapped_node.label,
                xy=(mapped_node.x, mapped_node.y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                color='darkblue',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='darkblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkblue', lw=1.5)
            )
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.axis('equal')
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filename}")
