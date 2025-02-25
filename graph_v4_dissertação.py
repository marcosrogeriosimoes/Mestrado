#! /usr/bin/env python3

# Import modules
try:
    import os
    import subprocess
    import csv
    import sys
    import argparse
    from Bio.PDB.PDBParser import  *
    import warnings
    from Bio import BiopythonWarning
    warnings.simplefilter('ignore', BiopythonWarning)
    from Bio.PDB.vectors import calc_dihedral
    import toml
    import numpy as np
    from itertools import *
    from igraph import *
    from scipy.spatial import distance
    from math import ceil
    from collections import Counter
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

except ImportError as e:
    print("### WARNING! Missing module ###")
    print("### Module missing: {}".format(e))


def plot_cavities (cavities_dir: str,
                   dt: float,
                   dt_unit: str,
                   smooth: bool,
                   window: int,
                   verbose: bool = False):

    """
    Plot number of cavities and volume for Dynamics simulation
    """

    cavities_dataframe = pd.DataFrame(columns=['frame', f'time ({dt_unit})', 'cavity_id', 'number of cavities'])
    volume_dataframe = pd.DataFrame(columns=['frame', f'time ({dt_unit})', 'cavity volume'])

    kv_results_files = sorted([f.replace('.pdb', '') for f in os.listdir(cavities_dir) if f.endswith('.pdb')])

    # Plot number of cavities for all frames
    for kv_file in kv_results_files:
        kv_finder_result = toml.load(f"{cavities_dir}/KV_Files/{kv_file}/{kv_file}.KVFinder.results.toml")
        kv_info = dict()
        kv_file_cavities = pd.DataFrame({'frame':kv_file, f'time ({dt_unit})':int(kv_file)*dt, 'number of cavities':len(kv_finder_result['RESULTS']['RESIDUES'])}, index=[0])
        cavities_dataframe = pd.concat([cavities_dataframe, kv_file_cavities])
    print(cavities_dataframe)

    # Plot number of cavities
    fig = plt.figure(dpi=300)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax1 = plt.subplots()
    st = fig.suptitle('Número de cavidades detectadas ao longo da dinâmica molecular')
    ax1.plot(f'time ({dt_unit})', 'number of cavities', data=cavities_dataframe, color='black', label='Número de cavidades detectadas')
    ax1.set_ylabel('Número de cavidades detectadas')
    ax1.set_xlabel(f'time ({dt_unit})')
    ax1.tick_params(axis='x', labelbottom=True)
    plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200], rotation=90)
    ax1.grid(True)
    ax1.legend(loc=(1.03, 0.85), prop=fontP)
    fig.tight_layout(rect=(0, 0, 1.0, 1.0))
    fig.subplots_adjust(top=0.91, hspace=0.1)
    st.set_y(0.97)
    fig.savefig(f"{cavities_dir}/KV_Files/number_of_cavities.png", dpi=300)

    # Plot cavities volume for all frames
    for kv_file in kv_results_files:
        kv_finder_result = toml.load(f"{cavities_dir}/KV_Files/{kv_file}/{kv_file}.KVFinder.results.toml")
        kv_info = dict()
        kv_file_cavities = pd.DataFrame({'frame':kv_file, f'time ({dt_unit})':int(kv_file)*dt, 'cavity volume':float(kv_finder_result['RESULTS']['VOLUME']['KAA'])}, index=[0])
        volume_dataframe = pd.concat([volume_dataframe, kv_file_cavities])
    print(volume_dataframe)

    # Plot cavities volume
    fig2 = plt.figure(dpi=300)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig2, ax2 = plt.subplots()
    # st = fig2.suptitle('Volume das cavidades detectadas ao longo da dinâmica molecular')
    if smooth:
        ax2.plot(f'time ({dt_unit})', 'cavity volume', data=volume_dataframe, color='gray', label='Volume das cavidades detectadas')
    else:
        ax2.plot(f'time ({dt_unit})', 'cavity volume', data=volume_dataframe, color='black', label='Volume das cavidades detectadas')
    if smooth:
        volume_dataframe['moving average volume'] = volume_dataframe.iloc[:, volume_dataframe.columns.get_loc('cavity volume')].rolling(window=int(window)).mean()
        ax2.plot(f'time ({dt_unit})', 'moving average volume', data=volume_dataframe, color='black', label='Média móvel do volume')
    ax2.set_ylabel('Volume das cavidades detectadas (Å3)', fontsize=15)
    ax2.set_xlabel("Tempo (ns)", fontsize=15)
    ax2.set_xticks(np.arange(0, 201 + 1, 10).astype(int))
    ax2.set_xticklabels(np.arange(0, 201 + 1, 10).astype(int), rotation=90)
    # ax2.tick_params(axis='x', labelbottom=True)
    # plt.xticks([1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201], rotation=90)
    ax2.axis([0, 201, 0, 2500])
    ax2.grid(True)
    ax2.legend(loc=(1.03, 0.85), prop=fontP)
    fig2.tight_layout(rect=(0, 0, 1.0, 1.0))
    fig2.subplots_adjust(top=0.91, hspace=0.1)
    st.set_y(0.97)
    fig2.savefig(f"{cavities_dir}/KV_Files/cavities_volume_10_scale.png", dpi=300)

    return True

def αC_extract_coords (cavities_dir: str,
                    verbose: bool = False) -> dict:
    """
    :Extract coordinates from cavity residues
    :param cavities_dir: Directory containing trajectory frames and KVFinderMD output
    :param verbose: provides additional details as to what the function is doing
    :type cavities_dir: str
    :type verbose: bool
    :return: dictionary containing coordinates from alpha carbons of cavity residues for all frames
    """

    # Get all frames (files that endswith .pdb)
    frames = sorted([f for f in os.listdir(cavities_dir) if f.endswith('.pdb')])

    # Get all KVFinderMD Results
    kv_results_files = sorted([f.replace('.pdb', '') for f in os.listdir(cavities_dir) if f.endswith('.pdb')])

    # Create empty dictionary for cavity info of all frames
    frames_coords = dict()

    # Extract C-alpha coordinates from .pdb files using BioPython
    for frame, kv_file in zip(frames, kv_results_files):
        dict_coords = dict()
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(f"{frame}", f"{cavities_dir}/{frame}")
        for model in structure.get_list():
            for chain in model.get_list():
                for residue in chain.get_list():
                    if residue.has_id("CA"):
                        dict_coords["{}_{}".format(residue.get_id()[1], residue.get_full_id()[2])] = residue["CA"].get_coord()

    # Create dictionary for each cavity containing residue number, chain and C-alpha coordinates
        kv_finder_result = toml.load(f"{cavities_dir}/KV_Files/{kv_file}/{kv_file}.KVFinder.results.toml")
        kv_info = dict()
        for kv_tag in kv_finder_result['RESULTS']['RESIDUES'].keys():
            kv_info[kv_tag] = dict()
            for residue_info in kv_finder_result['RESULTS']['RESIDUES'][kv_tag]:
                residue_id = residue_info[0]+"_"+residue_info[1]
                kv_info[kv_tag][residue_id] = dict_coords[residue_id]
        frames_coords[kv_file] = kv_info

    if verbose:
        print(f"> {cavities_dir} files were parsed and a dictionary containing alpha carbon coordinates was created")

    return frames_coords

def αC_calc_distance (frames_coords: dict,
                   verbose: bool = False) -> dict:
    """
    :Calculate Euclidian distance between C-alpha from cavity residues
    :param frames_coords: dictionary containing C-alpha coordinates from cavity residues for all frames
    :param verbose:provides additional details as to what the function is doing
    :type frames_coords: dict
    :type verbose: bool
    :return: dictionary containing Euclidian distance between C-alpha from cavity residues
    """

    # Create xyz coordinates array for residues
    for kv_file in frames_coords.keys():
        for kv_tag in frames_coords[kv_file].keys():
            coordinates_array = np.empty((0,3))
            for coordinates in frames_coords[kv_file][kv_tag].values():
                coordinates_array = np.append(coordinates_array, np.array([coordinates]), axis=0)

            # Create combinations using coordinates and residues
            coordinates_combination = combinations(coordinates_array, 2)
            residues_combination = combinations(frames_coords[kv_file][kv_tag].keys(), 2)

            # Calculate Euclidian distances between residues for each cavity
            residues_distance = dict()
            for coordinates_pair, residue_pair in zip(coordinates_combination, residues_combination):
                distance = np.linalg.norm(coordinates_pair[0]-coordinates_pair[1])
                residues_distance.update({residue_pair:distance})
            frames_coords[kv_file][kv_tag]= residues_distance

    if verbose:
        print (f"> Distances between alpha carbons were successfully calculated")

    return frames_coords

def create_graphs (frames_coords: dict,
                   cavities_dir: str,
                   αC_cutoff: float,
                   output_dir: str,
                   base_name: str,
                   verbose: bool = False) -> dict:
    """
    :Create graphs for representation of cavity residues
    :param frames_coords: dictionary containing Euclidian distance between C-alpha from cavity residues
    :param cavities_dir: Directory containing trajectory frames and KVFinderMD output
    :param output_dir: Directory used for saving output files
    :param αC_cutoff: Maximum distance for edge formation between αC atoms
    :param base_name: Base output file name
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type cavities_dir: str
    :type output_dir: str
    :type αC_cutoff: float
    :type base_name: str
    :type verbose: bool
    :return graphs saved in output_dir
    """

    # Create directory for graph saving
    try:
        os.mkdir(f"{output_dir}/graph_properties")
    except FileExistsError:
        pass
    for kv_file in frames_coords.keys():
        try:
            os.mkdir(f"{output_dir}/graph_properties/{kv_file}")
        except FileExistsError:
            pass

    # Create graph for each cavity containing all residues as vertices
    for kv_file in frames_coords.keys():
        for kv_tag in frames_coords[kv_file].keys():
            αC_graph = Graph()
            αC_graph_attribute = Graph()

            # Name graph according to the kv_tag
            αC_graph["name"] = kv_tag
            αC_graph_attribute["name"] = kv_tag

            # Add name vertex according to the residues
            kv_finder_result = toml.load(f"{cavities_dir}/KV_Files/{kv_file}/{kv_file}.KVFinder.results.toml")
            for residue_info in kv_finder_result['RESULTS']['RESIDUES'][kv_tag]:
                residue_id = residue_info[0]+"_"+residue_info[1]
                αC_graph.add_vertex(name=residue_id)
                αC_graph_attribute.add_vertex(name=residue_id)

            # Add edges and attributes according to distance between alpha Carbons
            for residue_pair, distance in frames_coords[kv_file][kv_tag].items():
                if distance < αC_cutoff:
                    αC_graph.add_edge(residue_pair[0], residue_pair[1], color="black", interaction="αC distance")
                    αC_graph_attribute.add_edge(residue_pair[0], residue_pair[1], αC_distance=distance, color="black", interaction="αC distance")

            # Save graph
            αC_graph.write_graphml(f=f"{output_dir}/graph_properties/{kv_file}/{base_name}_{kv_file}_{kv_tag}_αC_graph.graphml")
            αC_graph_attribute.write_graphml(f=f"{output_dir}/graph_properties/{kv_file}/{base_name}_{kv_file}_{kv_tag}_αC_graph_attribute.graphml")

    if verbose:
        print(f"> graphs were created for all frames")

    return True

def calc_diameter (frames_coords: dict,
                   output_dir: str,
                   base_name: str,
                   dt: float,
                   dt_unit: str,
                   verbose: bool = False) -> dict:
    """
    :Graph Diameter calculation
    :param frames_coords: dictionary containing Euclidian distance between C-alpha from cavity residues
    :param output_dir: Directory used for saving output files
    :param base_name: Base output file name
    :param dt: timestep of dynamics
    :param dt_unit: timestep unit
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type output_dir: str
    :type base_name: str
    :type dt: float
    :type dt_unit: str
    :type verbose: bool
    :return: dataframe containing graph diameter for all frames
    """

    # Linhas comentadas para calculo do diametro utilizando os atributos das arestas

    # Create empty dataframe for diameter values
    diameter_dataframe = pd.DataFrame(columns=['frame', 'cavity_id', 'diameter for alpha-carbon interaction'])

    # Read graphs saved in output_dir
    for kv_file in frames_coords.keys():
        for kv_tag in frames_coords[kv_file].keys():
            # αC_graph = Graph.Read_GraphML(f"{output_dir}/graph_properties/{kv_file}/{base_name}_{kv_file}_{kv_tag}_αC_graph.graphml")
            αC_graph = Graph.Read_GraphML(f"{output_dir}/graph_properties/{kv_file}/{base_name}_{kv_file}_{kv_tag}_αC_graph_attribute.graphml")

            # Calculate diameter for each edge attribute
            Calpha_path_diameter_list = list()
            if αC_graph.ecount() > 0:
                # Calpha_diameter = αC_graph.diameter(directed='False', unconn='True')
                Calpha_diameter = αC_graph.diameter(directed='False', unconn='True', weights='αC_distance')
            else:
                Calpha_diameter = 0

            if dt is not False:
                kv_file_diameter = pd.DataFrame({'frame':kv_file, f'time ({dt_unit})':int(kv_file)*dt, 'cavity_id':kv_tag, 'diameter for alpha-carbon interaction':float(Calpha_diameter)}, index=[0])
            else:
                kv_file_diameter = pd.DataFrame({'frame':kv_file, 'cavity_id':kv_tag, 'diameter for alpha-carbon interaction':float(Calpha_diameter)}, index=[0])

            diameter_dataframe = pd.concat([diameter_dataframe, kv_file_diameter])

    print(diameter_dataframe)

    if verbose:
        print(f"> diameter was successfully calculated for all frames in {output_dir}")

    return diameter_dataframe

def plot_diameter (diameter_dataframe:pd.core.frame.DataFrame,
                   output_dir: str,
                   base_name: str,
                   dt: str,
                   dt_unit: str,
                   y1_max: int,
                   window: 10,
                   verbose: bool = False,
                   smooth: bool = False) -> bool:
    """
    :Plot graph diameter using lineplot
    :param diameter_dataframe: Dataframe containing diameter values for all frames
    :param output_dir: Directory used for saving output files
    :param base_name: Base output file name
    :param dt: timestep of dynamics
    :param dt_unit: timestep unit
    :param y1_max: set the y-limit of diameter lineplot based on Alpha-carbon distance
    :param y2_max: set the y-limit of diameter lineplot based on other interactions
    :param y3_max: set the y-limit of diameter lineplot based on all interactions
    :param window:  Size of the moving window
    :param verbose: provides additional details as to what the function is doing
    :param smooth: plot a smooth line
    :type diameter_dataframe: pd.core.frame.Dataframe
    :type output_dir: str
    :type base_name: str
    :type dt: float
    :type dt_unit: str
    :type y1_max: int
    :type y2_max: int
    :type y3_max: int
    :type window: int
    :type verbose: bool
    :type smooth: bool
    :return: diameter lineplot saved in output_dir
    """

    if dt is not False:
        x = f'time ({dt_unit})'
    else:
        x = 'frame'

    # Plot diameter over time considering each alpha-carbon interaction
    fig = plt.figure(dpi=300)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax1 = plt.subplots()
    st = fig.suptitle('Diâmetro do grafo')
    if smooth:
        ax1.plot(x, 'diameter for alpha-carbon interaction', data=diameter_dataframe, color='gray', label='Diâmetro')
    else:
        ax1.plot(x, 'diameter for alpha-carbon interaction', data=diameter_dataframe, color='black', label='Diâmetro')
    ax1.set_ylabel('Diâmetro', fontsize=15)
    # ax1.set_xlabel(x)
    ax1.set_xlabel("Tempo (ns)", fontsize=15)
    xmin, ymin = (0, 0)
    if y1_max is False:
        xmax1, ymax1 = (max(diameter_dataframe[x]), 1.25 * (ceil(max(diameter_dataframe['diameter for alpha-carbon interaction']))))
    else:
        xmax1, ymax1 = (max(diameter_dataframe[x]), y1_max)
    ax1.axis([xmin, xmax1, ymin, ymax1])
    ax1.tick_params(axis='x', labelbottom=True)
    # plt.xticks([1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201], rotation=90)
    ax1.set_xticks(np.arange(0, xmax1 + 1, 10).astype(int))
    ax1.set_xticklabels(np.arange(0, xmax1 + 1, 10).astype(int), rotation=90)
    ax1.grid(True)

    # Plot smooth line
    if smooth:
        diameter_dataframe['moving average diameter for alpha-carbon interaction'] = diameter_dataframe.iloc[:, diameter_dataframe.columns.get_loc('diameter for alpha-carbon interaction')].rolling(window=int(window)).mean()
        ax1.plot(x, 'moving average diameter for alpha-carbon interaction', data=diameter_dataframe, color='black', label='Média móvel do diâmetro')
    ax1.legend(loc=(1.03, 0.85), prop=fontP)
    fig.tight_layout(rect=(0, 0, 1.0, 1.0))
    fig.subplots_adjust(top=0.91, hspace=0.1)
    st.set_y(0.97)
    fig.savefig(f"{output_dir}/graph_properties/{base_name}_diameter_lineplot_10_scale.png", dpi=300)

    if verbose:
        print(f"> diameter plots were successfully saved in {output_dir}/graph_properties")

def calc_centrality (frames_coords:dict,
                   output_dir: str,
                   base_name: str,
                   dt: str,
                   dt_unit: str,
                   verbose: bool = False):
    """
    :Calculate closeness centrality for all vertices
    :param frames_coords: Dictionary containing Euclidian distance between C-alpha from cavity residues
    :param output_dir: Directory used for saving output files
    :param base_name: Base output file name
    :param dt: timestep of dynamics
    :param dt_unit: timestep unit
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type output_dir: str
    :type base_name: str
    :type dt: float
    :type dt_unit: str
    :type verbose: bool
    :return: dataframe containing degree and closeness centrality
    """

    # Create empty dataframe for degree and closeness centrality values
    centrality_dataframe = pd.DataFrame(columns=['frame', 'cavity_id', 'residue', 'closeness alpha-carbon centrality with edge attribute', 'closeness alpha-carbon centrality without edge attribute'])
    for kv_file in frames_coords.keys():
        for kv_tag in frames_coords[kv_file].keys():
            αC_graph = Graph.Read_GraphML(f"{output_dir}/graph_properties/{kv_file}/{base_name}_{kv_file}_{kv_tag}_αC_graph.graphml")
            αC_graph_attribute = Graph.Read_GraphML(f"{output_dir}/graph_properties/{kv_file}/{base_name}_{kv_file}_{kv_tag}_αC_graph_attribute.graphml")

            # Calculate degree centrality based on Alpha-carbon interactions or all interactions together
            for residue_id in αC_graph.vs["name"]:
                # αC_graph_degree_centrality = αC_graph.degree(residue_id)
                # αC_graph_attribute_degree_centrality = αC_graph_attribute.degree(residue_id)

                # Calculate closeness centrality
                closeness_αC_graph_centrality = αC_graph.closeness(vertices=residue_id, mode='all')
                closeness_αC_graph_attribute_centrality = αC_graph_attribute.closeness(vertices=residue_id, mode='all', weights='αC_distance')
                if dt is not False:
                    kv_file_centrality = pd.DataFrame({'frame':int(kv_file), f'time ({dt_unit})':int(kv_file)*dt, 'cavity_id':kv_tag, 'residue':residue_id, 'closeness alpha-carbon centrality without edge attribute':float(closeness_αC_graph_centrality), 'closeness alpha-carbon centrality with edge attribute':float(closeness_αC_graph_attribute_centrality)}, index=[0])
                else:
                    kv_file_centrality = pd.DataFrame({'frame':int(kv_file), 'cavity_id':kv_tag, 'residue':residue_id, 'closeness alpha-carbon centrality without edge attribute':float(closeness_αC_graph_centrality), 'closeness alpha-carbon centrality with edge attribute':float(closeness_αC_graph_attribute_centrality)}, index=[0])

                centrality_dataframe = pd.concat([centrality_dataframe, kv_file_centrality])

    if verbose:
        print(f"> centrality was successfully calculated for all frames in {output_dir}")

    centrality_dataframe.to_csv(f"{output_dir}/graph_properties/{base_name}_centrality_dataframe.csv")

    return centrality_dataframe

def plot_degree_centrality (frames_coords: dict,
                     centrality_dataframe: pd.core.frame.DataFrame,
                     output_dir: str,
                     base_name: str,
                     dt: str,
                     dt_unit: str,
                     y4_max: float,
                     y5_max: float,
                     y6_max: float,
                     y7_max: float,
                     res_list: list,
                     occupancy_rate: float,
                     window: 10,
                     smooth: bool = False,
                     verbose: bool = False):

    """Plot heatmap for degree centrality
    :param frames_coords: Dictionary containing Euclidian distance between C-alpha from cavity residues
    :param centrality_dataframe: Dataframe containing degree and closeness centrality values
    :param output_dir: Directory used for saving output files
    :param base_name: Base output file name
    :param dt: timestep of dynamics
    :param dt_unit: timestep unit
    :param y4_max: set the anchor value of degree heatmap from Alpha-carbon distance for all cavity residues
    :param y5_max: set the anchor value of degree heatmap from all interactions for all cavity residues
    :param y6_max: set the anchor value of degree heatmap from Alpha-carbon interactions for res_list residues
    :param y7_max: set the anchor value of degree heatmap fro all interactions for res_list residues
    :param res_list: list of residues to be plotted in heatmap
    :param occupancy_rate: residues occupancy from dynamics
    :param window: Size of the moving window
    :param smooth: Plot a smooth line
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type centrality_dataframe: pd.core.frame.Dataframe
    :type output_dir: str
    :type base_name: str
    :type dt: float
    :type dt_unit: str
    :type y4_max: float
    :type y5_max: float
    :type y6_max: float
    :type y7_max: float
    :type res_list: list
    :type occupancy_rate: float
    :type window: int
    :type smooth: bool
    :type verbose: bool
    :return: degree centrality heatmap saved in output_dir, dataframe containing residues occupancy in dynamics
    """

    if dt is not False:
        x = f'time ({dt_unit})'
    else:
        x = 'frame'

    # Calculate residues occupancy in dynamics and exclude residues below occupancy_rate
    residue_frequency = centrality_dataframe['residue'].value_counts()
    # residue_frequency.to_csv(f"{output_dir}/graph_properties/residue_frequency.csv", index=True, header=['occupancy'])
    for residue, frequency in residue_frequency.items():
        if frequency/len(frames_coords.keys()) <= occupancy_rate:
            centrality_dataframe.drop(centrality_dataframe[centrality_dataframe['residue'] == residue].index, inplace=True)

    # Plot degree centralities based on Alpha-carbon distance
    alphacarbon_dataframe = centrality_dataframe[['frame', 'residue', 'alpha-carbon centrality']]
    alphacarbon_result = alphacarbon_dataframe.pivot(index='residue', columns='frame', values='alpha-carbon centrality')
    alphacarbon_result = alphacarbon_result.fillna(0)
    ax1 = plt.figure(dpi=300)
    if y4_max is False:
        sns_plot = sns.heatmap(alphacarbon_result, yticklabels=True)
    else:
        sns_plot = sns.heatmap(alphacarbon_result, yticklabels=True, vmax=y4_max)
    sns_plot.axes.set_title("Heatmap of vertices centralities based on alpha-carbon distance", fontsize=11)
    sns_plot.set_xlabel(x, fontsize=10)
    sns_plot.set_ylabel("residue", fontsize=10)
    sns_plot.tick_params(axis="x", labelsize=8)
    sns_plot.tick_params(axis="y", labelsize=6)
    fig = sns_plot.get_figure()
    fig.tight_layout(rect=(0, 0, 1.0, 1.0))
    fig.savefig(f"{output_dir}/graph_properties/{base_name}_alpha-carbon_centrality_heatmap.png", dpi=300)

    # Plot degree centralities based on all interactions
    allinteractions_dataframe = centrality_dataframe[['frame', 'residue', 'all-interactions centrality']]
    allinteractions_result = allinteractions_dataframe.pivot(index='residue', columns='frame', values='all-interactions centrality')
    allinteractions_result = allinteractions_result.fillna(0)
    ax2 = plt.figure(dpi=300)
    if y5_max is False:
        sns2_plot = sns.heatmap(allinteractions_result, yticklabels=True)
    else:
        sns2_plot = sns.heatmap(allinteractions_result, yticklabels=True, vmax=y5_max)
    sns2_plot.axes.set_title("Heatmap of vertices centralities based on all interactions", fontsize=11)
    sns2_plot.set_xlabel(x, fontsize=10)
    sns2_plot.set_ylabel("residue", fontsize=10)
    sns2_plot.tick_params(axis="x", labelsize=8)
    sns2_plot.tick_params(axis="y", labelsize=6)
    fig2 = sns2_plot.get_figure()
    fig2.savefig(f"{output_dir}/graph_properties/{base_name}_all-interactions_centrality_heatmap.png", dpi=300)

    # Select rows in centrality_dataframe based on res_list
    if res_list is False:
        return residue_frequency
    else:
        res_dataframe = centrality_dataframe[centrality_dataframe["residue"].isin(res_list)]

        # Plot degree centrality of selected residues based on Alpha-carbon distance
        res_alphacarbon_dataframe = res_dataframe[['frame', 'residue', 'alpha-carbon centrality']]
        res_alphacarbon_result = res_alphacarbon_dataframe.pivot(index='residue', columns='frame', values='alpha-carbon centrality')
        res_alphacarbon_result = res_alphacarbon_result.fillna(0)
        ax3 = plt.figure(dpi=300)
        if y6_max is False:
            sns3_plot = sns.heatmap(res_alphacarbon_result, yticklabels=True)
        else:
            sns3_plot = sns.heatmap(res_alphacarbon_result, yticklabels=True, vmax=y6_max)
        sns3_plot.axes.set_title("Heatmap of vertices centralities based on alpha-carbon distance", fontsize=11)
        sns3_plot.set_xlabel(x, fontsize=10)
        sns3_plot.set_ylabel("residue", fontsize=10)
        sns3_plot.tick_params(axis="x", labelsize=8)
        sns3_plot.tick_params(axis="y", labelsize=6)
        fig3 = sns3_plot.get_figure()
        fig3.savefig(f"{output_dir}/graph_properties/{base_name}_residues_alpha-carbon_centrality_heatmap.png", dpi=300)

        # Plot degree centrality of selected residues based on all interactions
        res_allinteractions_dataframe = res_dataframe[['frame', 'residue', 'all-interactions centrality']]
        res_allinteractions_result = res_allinteractions_dataframe.pivot(index='residue', columns='frame', values='all-interactions centrality')
        res_allinteractions_result = res_allinteractions_result.fillna(0)
        ax4 = plt.figure(dpi=300)
        if y7_max is False:
            sns4_plot = sns.heatmap(res_allinteractions_result, yticklabels=True)
        else:
            sns4_plot = sns.heatmap(res_allinteractions_result, yticklabels=True, vmax=y7_max)
        sns4_plot.axes.set_title("Heatmap of vertices centralities based on all interactions", fontsize=11)
        sns4_plot.set_xlabel(x, fontsize=10)
        sns4_plot.set_ylabel("residue", fontsize=10)
        sns4_plot.tick_params(axis="x", labelsize=8)
        sns4_plot.tick_params(axis="y", labelsize=6)
        fig4 = sns4_plot.get_figure()
        fig4.savefig(f"{output_dir}/graph_properties/{base_name}_residues_all-interactions_centrality_heatmap.png", dpi=300)

        if verbose:
            print(f"> degree centrality plots were successfully saved in {output_dir}/graph_properties")

        return residue_frequency

def plot_closeness_centrality (frames_coords: dict,
                     centrality_dataframe: pd.core.frame.DataFrame,
                     output_dir: str,
                     dt: str,
                     dt_unit: str,
                     y8_max: float,
                     y9_max: float,
                     y10_max: float,
                     y11_max: float,
                     res_list: list,
                     occupancy_rate: float,
                     window: 10,
                     smooth: bool = False,
                     verbose: bool = False):

    """Plot heatmap for closeness centrality
    :param frames_coords: Dictionary containing Euclidian distance between C-alpha from cavity residues
    :param centrality_dataframe: Dataframe containing degree and closeness centrality values
    :param output_dir: Directory used for saving output files
    :param base_name: Base output file name
    :param dt: timestep of dynamics
    :param dt_unit: timestep unit
    :param y9_max: set the anchor value of closeness heatmap from Alpha-carbon distance for all cavity residues
    :param y10_max: set the anchor value of closeness heatmap from all interactions for all cavity residues
    :param y11_max: set the anchor value of closeness heatmap from Alpha-carbon interactions for res_list residues
    :param y12_max: set the anchor value of closeness heatmap fro all interactions for res_list residues
    :param res_list: list of residues to be plotted in heatmap
    :param occupancy_rate: residues occupancy from dynamics
    :param window: Size of the moving window
    :param smooth: Plot a smooth line
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type centrality_dataframe: pd.core.frame.Dataframe
    :type output_dir: str
    :type base_name: str
    :type dt: float
    :type dt_unit: str
    :type y8_max: float
    :type y9_max: float
    :type y10_max: float
    :type y11_max: float
    :type res_list: list
    :type occupancy_rate: float
    :type window: int
    :type smooth: bool
    :type verbose: bool
    :return: function success
    """

    if dt is not False:
        x = f'time ({dt_unit})'
    else:
        x = 'frame'

    # Reset index dataframe
    centrality_dataframe.reset_index(drop=True, inplace=True)

    print(centrality_dataframe)

    # Calculate residues occupancy in dynamics and exclude residues below occupancy_rate
    residue_frequency = centrality_dataframe['residue'].value_counts()
    for residue, frequency in residue_frequency.items():
        if frequency/len(frames_coords.keys()) <= occupancy_rate:
            centrality_dataframe.drop(centrality_dataframe[centrality_dataframe['residue'] == residue].index, inplace=True)

    # Extrair apenas o número do resíduo (presumindo que o resíduo seja uma string como '8_A', '80_A', etc.)
    alphacarbon_dataframe = centrality_dataframe[[x, 'residue', 'closeness alpha-carbon centrality with edge attribute']]
    alphacarbon_dataframe[x] = alphacarbon_dataframe[x].astype(int)
    alphacarbon_result = alphacarbon_dataframe.pivot(index='residue', columns=x, values='closeness alpha-carbon centrality with edge attribute')
    # Passo 1: Selecionar as duas últimas linhas
    ultimas_duas = alphacarbon_result.iloc[-2:]
    # Passo 3: Selecionar o meio do DataFrame (excluindo as primeiras e últimas duas linhas)
    meio = alphacarbon_result.iloc[:-2]
    # Passo 4: Concatenar as partes na nova ordem
    alphacarbon_result = pd.concat([ultimas_duas, meio], axis=0)
    alphacarbon_result = alphacarbon_result.fillna(0)
    print(alphacarbon_result)

    # Plot closeness centralities with edge attribute
    ax1 = plt.figure(dpi=300)
    if y8_max is False:
        sns_plot = sns.heatmap(alphacarbon_result, vmin=0, yticklabels=True)
    else:
        sns_plot = sns.heatmap(alphacarbon_result, yticklabels=True, vmin=0, vmax=y8_max)
        sns_plot.collections[0].set_clim(vmin=0, vmax=y8_max)
        sns_plot.collections[0].colorbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        sns_plot.collections[0].colorbar.set_ticklabels([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    sns_plot.set_xlabel("Tempo (ns)", fontsize=14)
    sns_plot.set_xticks(np.arange(0, 201 + 1, 10).astype(int))
    sns_plot.set_xticklabels(np.arange(0, 201 + 1, 10).astype(int), rotation=90)
    sns_plot.set_ylabel("Resíduo", fontsize=14)
    sns_plot.tick_params(axis="x", labelsize=8)
    sns_plot.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig(f"{output_dir}/graph_properties/withattribute_closeness_centrality_heatmap.png", dpi=300)


    # Extrair apenas o número do resíduo (presumindo que o resíduo seja uma string como '8_A', '80_A', etc.)
    alphacarbon_dataframe_noattribute = centrality_dataframe[[x, 'residue', 'closeness alpha-carbon centrality without edge attribute']]
    alphacarbon_dataframe_noattribute[x] = alphacarbon_dataframe_noattribute[x].astype(int)
    alphacarbon_result_noattribute = alphacarbon_dataframe_noattribute.pivot(index='residue', columns=x, values='closeness alpha-carbon centrality without edge attribute')
    # Passo 1: Selecionar as duas últimas linhas
    ultimas_duas = alphacarbon_result_noattribute.iloc[-2:]
    # Passo 3: Selecionar o meio do DataFrame (excluindo as primeiras e últimas duas linhas)
    meio = alphacarbon_result_noattribute.iloc[:-2]
    # Passo 4: Concatenar as partes na nova ordem
    alphacarbon_result_noattribute = pd.concat([ultimas_duas, meio], axis=0)
    alphacarbon_result_noattribute = alphacarbon_result_noattribute.fillna(0)
    print(alphacarbon_result_noattribute)

    # Plot closeness centralities without edge attribute for frames 100 e 177
    ax2 = plt.figure(dpi=300)
    if y9_max is False:
        sns2_plot = sns.heatmap(alphacarbon_result_noattribute, vmin=0, yticklabels=True)
    else:
        sns2_plot = sns.heatmap(alphacarbon_result_noattribute, yticklabels=True, vmin=0, vmax=y9_max)
        sns2_plot.collections[0].set_clim(vmin=0, vmax=y9_max)
        sns2_plot.collections[0].colorbar.set_ticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
        sns2_plot.collections[0].colorbar.set_ticklabels([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    sns2_plot.set_xlabel("Tempo (ns)", fontsize=14)
    sns2_plot.set_xticks(np.arange(0, 201 + 1, 10).astype(int))
    sns2_plot.set_xticklabels(np.arange(0, 201 + 1, 10).astype(int), rotation=90)
    sns2_plot.set_ylabel("Resíduo", fontsize=14)
    sns2_plot.tick_params(axis="x", labelsize=8)
    sns2_plot.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig2 = sns2_plot.get_figure()
    fig2.savefig(f"{output_dir}/graph_properties/noattribute_closeness_centrality_heatmap.png", dpi=300)

    return True

if "__main__"== __name__:
    cavities_directory = "input/HIV1_DM_analysis/fixed_hiv-open_dt1000ps"
    output_directory = "input/HIV1_DM_analysis/fixed_hiv-open_dt1000ps"
    # plot = plot_cavities(cavities_dir=cavities_directory, verbose="True", dt=1, dt_unit="ns", smooth=True, window=10)
    αC_coords = αC_extract_coords(cavities_dir=cavities_directory, verbose="True")
    # αC_distance = αC_calc_distance(frames_coords=αC_coords, verbose="True")
    # graph = create_graphs(cavities_dir=cavities_directory, frames_coords=αC_distance, αC_cutoff=10, output_dir=output_directory, base_name="121023", verbose="True")
    # diameter = calc_diameter(output_dir=output_directory, frames_coords=αC_distance, base_name="121023", dt = 1, dt_unit = "ns", verbose="True")
    # diameter_plot = plot_diameter(output_dir=output_directory, diameter_dataframe=diameter, smooth="True", base_name="comatributo", dt = 1, dt_unit = "ns", y1_max=80, window=10, verbose="True")
    centrality = calc_centrality(frames_coords=αC_coords, output_dir=output_directory, base_name="121023", dt = 1, dt_unit = "ns", verbose="True")
    # degree_centrality_plot = plot_degree_centrality (frames_coords=αC_coords, centrality_dataframe=centrality, output_dir=output_directory, base_name="teste", dt = False, dt_unit = False, y4_max=False, y5_max=False, y6_max=False, y7_max=False, res_list=["48_A", "48_B", "49_A", "49_B", "50_A", "50_B", "51_A", "51_B", "52_A", "52_B", "25_A", "25_B", "26_A", "26_B", "27_A", "27_B", "47_A", "47_B", "71_A", "71_B"], occupancy_rate = 0.75, window=10, smooth="True", verbose="True")
    closeness_centrality_plot = plot_closeness_centrality (frames_coords=αC_coords, centrality_dataframe=centrality, output_dir=output_directory, dt = 1, dt_unit = "ns", y8_max=0.09, y9_max=0.65, y10_max=False, y11_max=False, res_list=False, occupancy_rate = 0.95, window=10, smooth="True", verbose="True")
