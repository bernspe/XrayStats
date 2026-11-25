import requests
import pandas as pd

import bspline_python
from bspline_python import *
from dotmap import DotMap


def load_data():
    api_url='https://xray-api.renecol.org'
    token='VyE8fcuCJTcV7qBSH70YfhcRlj4Ijq'
    image_suffix='/api/imagedata/'
    params={'showall':True,'detailed':True}
    response = requests.get(headers={'Authorization': 'Bearer ' + token},url=api_url+image_suffix, params=params).json()
    df = pd.DataFrame(response)
    df2 = df[['id','patient_id','measure_dataset','referring_patient','img_type','acquisition_date']]
    # filter those rows where measure_dataset is not an empty list
    df3 = df2[df2['measure_dataset'].map(lambda d: len(d)) > 0]

    # unpack measure_dataset column and combine it with the rest of df3, renaming the columns appropriately
    df4 = pd.concat([df3.drop(columns=['measure_dataset']), df3['measure_dataset'].apply(pd.Series)], axis=1)
    # transfer to long format
    df4b=df4.melt(id_vars=['id','patient_id','referring_patient','img_type','acquisition_date'], value_name='measure_dataset', var_name='measure_dataset_index').dropna(subset=['measure_dataset'])
    df4c = pd.concat([df4b.drop(columns=['measure_dataset']), df4b['measure_dataset'].apply(pd.Series)['data']], axis=1)
    df4d = pd.concat([df4c.drop(columns=['data']), df4c['data'].apply(pd.Series)], axis=1)
    df4d.to_pickle('data/df4d_spine_measurements.pkl')
    return df4d



def load_from_pickle():
    df4d = pd.read_pickle('data/df4d_spine_measurements.pkl')
    return df4d


def COBB_angles_spline_reconstruction(m):
    """ take the nodes from a measurement, reconstruct a B-spline, map the nodes to the spline, calculate tangents, and reconstruct regional and segmental COBB angles"""
    nodes = [(i['x'], i['y']) for i in m['stateObject']['nodes']]
    selected_vertebra = spine_labels[m['stateObject']['vertebra_range'][0]:m['stateObject']['vertebra_range'][1] + 1]

    spline = interpolate_bspline(nodes, num_steps=10)
    nodesmap = [DotMap(i) for i in m['stateObject']['nodes']]
    mapped = map_nodes_to_spline_and_keys(nodesmap, selected_vertebra, spline)
    node_idcs = [i.spline_idx for i in mapped]
    tangents = calculate_tangents(spline, node_idcs, length=30)

    perp_lines = get_perpendicular_lines_at_indices(spline, length=20, indices=node_idcs)

    # Optimization. If there is more than one occurence of the same region in result dict, repeat generate_COBB_angles_from_tangents with a higher angle_threshold to reduce number of angles
    result = {}
    start = True
    threshold = 10
    while start | ((any(len(v) > 1 for v in result.values())) & (threshold<=30)):
        start = False
        result = {}
        cobb_angles = generate_COBB_angles_from_tangents(tangents, mapped, perp_lines, mode='regional_S1',angle_threshold=threshold)

        for angle_info in cobb_angles:
            region = convertMeasurementBorderToSpineRegion(angle_info['from_label'], angle_info['to_label'], m['stateObject'])
            if region not in result:
                result[region] = [angle_info]
            else:
                result[region].append(angle_info)
        threshold+=3
    return result, threshold



def main():
    #df4d=load_data()
    df4d=load_from_pickle()
    for i in range(10):
        m = df4d.iloc[i, 6]
        result, threshold = COBB_angles_spline_reconstruction(m)
        print(f"Measurement {i} COBB angles (used {threshold} degree as cutoff):")
        for region, angles in result.items():
            for angle_info in angles:
                print(f"  Region: {region}, Angle: {angle_info['cobb_angle']:.2f}°, From: {angle_info['from_label']}, To: {angle_info['to_label']}")


"""    
    # Plot anzeigen
    plot_spline_and_lines(
        original_points=nodes,
        interpolated_points=spline,
        lines=perp_lines,
        tangents=tangents,
        mapped_nodes=mapped,
        show_angles=True,
        invert_yaxis=True,
        title="Meine Spline-Analyse"
    )
"""
if __name__ == "__main__":
    main()
