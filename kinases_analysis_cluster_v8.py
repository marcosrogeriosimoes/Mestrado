#! /usr/bin/env python3

try:
    import os
    import subprocess
    import csv
    import sys
    import argparse
    import traceback
    from Bio.PDB.PDBParser import  *
    from Bio.PDB.Structure import Structure
    from Bio.PDB.PDBIO import PDBIO, Select
    from Bio.PDB.Chain import Chain
    import warnings
    from Bio import BiopythonWarning
    warnings.simplefilter('ignore', BiopythonWarning)
    from Bio.PDB.vectors import calc_dihedral
    from Bio.Align.Applications import ClustalOmegaCommandline
    from Bio.Align.Applications import MuscleCommandline
    from Bio import AlignIO
    from Bio import SeqIO
    import toml
    import numpy as np
    from itertools import *
    from igraph import *
    from scipy.spatial import distance
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
    from math import ceil
    from collections import Counter
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from pymol import cmd
    from Bio import Phylo
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    import requests
    from matplotlib.patches import Rectangle
    from sklearn.cluster import AgglomerativeClustering
    from matplotlib.pyplot import gcf
    import matplotlib.patches as mpatches
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

except ImportError as e:
    print("### WARNING! Missing module ###")
    print("### Module missing: {}".format(e))

def pdb_download (input_directory: str):
    """
    Download of all human kinases sequences available in KinCore database from PDB
    """

    # Open tsv file containing all human kinases structures
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')
    # Exclude last character of PDB code related to chain
    kincore_dataframe['PDB'] = kincore_dataframe['PDB'].str[:-1]
    # Exclude duplicates in tsv file
    kincore_dataframe = kincore_dataframe.drop_duplicates(subset='PDB', keep='first')
    # Download PDB structures
    for index, row in kincore_dataframe.iterrows():
        pdb_id = row['PDB']

        # Monta a URL do arquivo PDB
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        # Define o caminho de destino para salvar o arquivo PDB
        pdb_path = f"{input_directory}/pdb_structures/{pdb_id}.pdb"

        # Faz o download do arquivo PDB usando a biblioteca requests
        response = requests.get(pdb_url)

        # Verifica se o download foi bem-sucedido
        if response.status_code == 200:
            # Salva o arquivo PDB no diretório de destino
            with open(pdb_path, 'wb') as file:
                file.write(response.content)
                print(f"Arquivo PDB {pdb_id} baixado com sucesso")
        else:
            print(f"Erro ao baixar o arquivo PDB {pdb_id}.")

def fix_pdb(input_directory: str):
    """
    Use PDBFixer to build missing residues
    """

    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    # Create directory for structures containing only chain A
    if not os.path.exists(f"{input_directory}/pdb_fixed"):
        os.mkdir(f"{input_directory}/pdb_fixed")

    kinases_files = sorted([f for f in os.listdir(f"{input_directory}/pdb_structures") if f.endswith('.pdb')])

    for kinase in kinases_files:
        if os.path.isfile(f"{input_directory}/pdb_fixed/{kinase}_fixed.pdb"):
            pass
        else:
            try:
                fixer = PDBFixer(filename=f"{input_directory}/pdb_structures/{kinase}")
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{input_directory}/pdb_fixed/{kinase}_fixed.pdb", 'w'),  keepIds=True)
            except:
                print(f"{kinase} cannot be fixed by PDBFixer")

    return True

def split_chains (input_directory: str):
    """
    Select chains from pdb that are presented in kincore database and exclude ligands
    """

    # Create directory for structures containing only chain A
    if not os.path.exists(f"{input_directory}/chain_fixed"):
        os.mkdir(f"{input_directory}/chain_fixed")

    # Open tsv file containing all human kinases structures
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')

    # Select chain of pdb files from data in kincore_dataframe
    kincore_dataframe['PDB_file'] = kincore_dataframe['PDB'].str[:-1]
    kincore_dataframe['chain'] = kincore_dataframe['PDB'].str[-1]

    # Select chain from structure based on dataframe information and exclude heteroatoms
    class ChainSelector(Select):
        def accept_chain(self, chain):
            if chain.id == row['chain']:
                return True
            else:
                return False
        def accept_residue(self, residue):
            if residue.id[0] == " ":
                return True
            else:
                return False

    parser = PDBParser()

    # Iterate over kincore_dataframe to save chains from pdb files and also exclude ligands from file
    for index, row in kincore_dataframe.iterrows():
        io = PDBIO()
        try:
            structure = parser.get_structure(row['PDB_file'], f"{input_directory}/pdb_fixed/{row['PDB_file']}.pdb_fixed.pdb")
            print(index)
            io.set_structure(structure)
            io.save(f"{input_directory}/chain_fixed/{row['PDB_file']}_{row['chain']}.pdb_fixed.pdb", ChainSelector())
        except:
            pass

    return True

def structure_alignment (input_directory: str,
                         reference_structure: str):
    """
    Align all kinase structures using a structure with ATP as reference
    """

    if not os.path.exists(f"{input_directory}/aligned_PDB_fixed"):
        os.mkdir(f"{input_directory}/aligned_PDB_fixed")

    kinases_files = sorted([f for f in os.listdir(f"{input_directory}/chain_fixed") if f.endswith('.pdb')])

    # Download reference structure
    reference = cmd.fetch(reference_structure)
    cmd.remove("solvent")
    cmd.remove("chain B")
    # cmd.remove("chain C")
    # cmd.remove("chain D")

    # Align all kinases structures separately using 2hyy as reference
    for kinase in kinases_files:
        cmd.load(f"{input_directory}/chain_fixed/{kinase}")
        cmd.remove("solvent")
        object_names = cmd.get_object_list()
        if len(object_names) == 2:
            reference_name = object_names[0]
            kinase_name = object_names[1]
            cealign = cmd.cealign(reference_name, kinase_name)
            cmd.save(f"{input_directory}/aligned_PDB_fixed/{kinase}", kinase_name)
            cmd.delete(kinase_name)
        else:
            print(f"{kinase} has no pdb structure")
            pass

    return True

def pykvfinder_test (input_directory: str):

    """
    Test cavities detection using ligands
    """

    import pyKVFinder

    # files = ["6S9W_A", "3CS9_A", "8GDS_A"]
    files = ["4L6Q_A", "7P6N_H", "4TWP_A"]

    for file in files:
        print(file)
        if not os.path.exists(f"{input_directory}/fixed_KV_Files_300_D_ligand"):
            os.mkdir(f"{input_directory}/fixed_KV_Files_300_D_ligand")
        if not os.path.exists(f"{input_directory}/fixed_KV_Files_300_D_ligand/{file}"):
            os.mkdir(f"{input_directory}/fixed_KV_Files_300_D_ligand/{file}")

        # Read ligand file
        latomic = pyKVFinder.read_pdb(f"{input_directory}/ligante_4rix.pdb")
        print(latomic)
        # Read pdb file
        atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{file}.pdb_fixed.pdb")
        # Detect cavities
        vertices_load = toml.load(f"{input_directory}/kinases_vertices.toml")
        vertices = vertices_load["box"]
        ligand_cutoff_list = [5, 6, 7, 8, 9, 10]
        for ligand_cutoff in ligand_cutoff_list:
            ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, latomic=latomic, volume_cutoff=300, ligand_cutoff=ligand_cutoff)
            residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
            surface, volume, area = pyKVFinder.spatial(cavities)
            print(ncav)
            print(cavities)
            # Export results
            pyKVFinder.export(f"{input_directory}/fixed_KV_Files_300_D_ligand/{file}/{file}_{ligand_cutoff}_output.pdb", cavities, None, vertices)
            pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_300_D_ligand/{file}/{file}_{ligand_cutoff}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{file}", ligand=f"{input_directory}/ligante_2hyy.pdb", output=f"{input_directory}/fixed_KV_Files_300_D_ligand/{file}/{file}_{ligand_cutoff}_output.pdb", residues=residues, volume=volume)

    return True

def pykvfinder_ligand (input_directory: str,
                       volume_cutoff: int):
    """
    Run pyKVFinder for all kinases based on ligand position
    ATP_ligand directories: cavities detection for all proteins used ATP as ligand
    imatinib_ligand directories: cavities detection for all proteins used imatinib as ligand
    files out of folders: DFG_in used ATP as ligand, DFG_Out used imatinib
    """

    print("START")

    # Open tsv file containing all human kinases structures
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')

    dfgin_count = kincore_dataframe[kincore_dataframe['SpatialLabel'] == 'DFGin'].shape[0]
    dfgout_count = kincore_dataframe[kincore_dataframe['SpatialLabel'] == 'DFGout'].shape[0]

    kinases_files = sorted([f for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])
    kinases_names = sorted([f.replace('.pdb', '') for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])

    # Set ligand cutoff values in a list
    # ligand_cutoff_list = [5, 6, 7, 8, 9, 10]
    ligand_cutoff_list = [5]

    # Import pyKVFinder and create results directories
    import pyKVFinder

    # Create directory for pyKVFinder files
    if not os.path.exists(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand"):
        os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand")

    # Create empty array for ocurrence calculation for each ligand cutoff
    for ligand_cutoff in ligand_cutoff_list:
        var_name_in = f"in_occurrence_{ligand_cutoff}"
        globals()[var_name_in] = None
        var_name_out = f"out_occurrence_{ligand_cutoff}"
        globals()[var_name_out] = None

    # Set vertices for detection
    vertices_load = toml.load(f"{input_directory}/kinases_vertices.toml")
    vertices = vertices_load["box"]

    if os.path.exists(f"{input_directory}/kinases_vertices.toml"):
        for kinase, file in zip(kinases_files, kinases_names):
            file = file.replace("_fixed", "")
            file = file.replace("_", "")
            kinase_conformation = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'SpatialLabel'].values[0]
            if kinase_conformation == "DFGin" or kinase_conformation == "DFGout":
                # Create directory if SpatialLabel is DFGIn or DFGOut
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file} already exists")
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/ATP_ligand")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/ATP_ligand already exists")
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/imatinib_ligand")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/imatinib_ligand already exists")

                # Detect cavities for all ligand cutoffs values in ligand_cutoff_list
                for ligand_cutoff in ligand_cutoff_list:
                    var_name_in = f"in_occurrence_{ligand_cutoff}"
                    var_name_out = f"out_occurrence_{ligand_cutoff}"
                    try:
                        # Read pdb ligand file
                        # If all structures use DFGin ligand:
                        # latomic = pyKVFinder.read_pdb(f"{input_directory}/ligante_4rix.pdb")
                        # If all structures use DFGout ligand:
                        latomic = pyKVFinder.read_pdb(f"{input_directory}/ligante_2hyy.pdb")
                        # if kinase_conformation == "DFGin":
                        #     latomic = pyKVFinder.read_pdb(f"{input_directory}/ligante_4rix.pdb")
                        # if kinase_conformation == "DFGout":
                        #     latomic = pyKVFinder.read_pdb(f"{input_directory}/ligante_2hyy.pdb")
                        # Read pdb protein file
                        atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                        ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, latomic=latomic, volume_cutoff=volume_cutoff, ligand_cutoff=ligand_cutoff)
                        # Identification of interface residues and cavity volume
                        residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
                        surface, volume, area = pyKVFinder.spatial(cavities)
                        # Export results
                        pyKVFinder.export(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/imatinib_ligand/{file}_{ligand_cutoff}_output.pdb", cavities, None, vertices)
                        # if kinase_conformation == "DFGin":
                        pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/imatinib_ligand/{file}_{ligand_cutoff}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{kinase}", ligand=f"{input_directory}/ligante_2hyy.pdb", output=f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/imatinib_ligand/{file}_{ligand_cutoff}_output.pdb", residues=residues, volume=volume)
                        # if kinase_conformation == "DFGout":
                        #     pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/{file}_{ligand_cutoff}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{kinase}", ligand=f"{input_directory}/ligante_2hyy.pdb", output=f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{file}/{file}_{ligand_cutoff}_output.pdb", residues=residues, volume=volume)

                        # Calculate occurrence if number of cavities = 1
                        if kinase_conformation == "DFGin":
                            if globals()[var_name_in] is None and ncav == 1:
                                globals()[var_name_in] = (cavities > 1).astype(int)
                            elif globals()[var_name_in] is not None and ncav == 1:
                                globals()[var_name_in] += (cavities > 1).astype(int)
                            else:
                                print(kinase, ncav)

                        if kinase_conformation == "DFGout":
                            if globals()[var_name_out] is None and ncav == 1:
                                globals()[var_name_out] = (cavities > 1).astype(int)
                            elif globals()[var_name_out] is not None and ncav == 1:
                                globals()[var_name_out] += (cavities > 1).astype(int)
                            else:
                                print(kinase, ncav)
                    except:
                        print(f"{file} could not be used to detect cavities")


    for ligand_cutoff in ligand_cutoff_list:
        var_name_in = f"in_occurrence_{ligand_cutoff}"
        var_name_out = f"out_occurrence_{ligand_cutoff}"
        percentage_in = (globals()[var_name_in]/dfgin_count) * 100
        percentage_out = (globals()[var_name_out]/dfgout_count) * 100
        cavities_in = ((globals()[var_name_in] > 0).astype('int32'))*2
        cavities_out = ((globals()[var_name_out] > 0).astype('int32'))*2
        pyKVFinder.export(fn=f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/imatinib_ligand/grid_occurrence_{ligand_cutoff}_in.pdb", cavities=cavities_in, surface=None, vertices=vertices, B=percentage_in)
        pyKVFinder.export(fn=f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/imatinib_ligand/grid_occurrence_{ligand_cutoff}_out.pdb", cavities=cavities_out, surface=None, vertices=vertices, B=percentage_out)

    print("END: Detect cavities and Calculate ocurrence for all ligand cutoffs")

    return True

def pyKVFinder (input_directory: str,
                reference_structure: str,
                volume_cutoff: int,
                detection_residues: list):
    """
    Run pyKVFinder for all kinases
    """

    print("START")

    # Open tsv file containing all human kinases structures
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')

    # Select D residue from DFG in kincore_dataframe
    kincore_dataframe['DFG_D'] = kincore_dataframe['DFG_Phe'] - 1

    kinases_files = sorted([f for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])
    kinases_names = sorted([f.replace('.pdb', '') for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])

    # Import pyKVFinder and create results directories
    import pyKVFinder

    if "D" in detection_residues:
        if not os.path.exists(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D"):
            os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D")
    if "F" in detection_residues:
        if not os.path.exists(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F"):
            os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F")

    # Detect cavities considering D residue from DFG
    if "D" in detection_residues:
        # Create empty array for ocurrence calculation
        occurrence = None
        if os.path.exists(f"{input_directory}/kinases_vertices.toml"):
            for kinase, file in zip(kinases_files, kinases_names):
                file = file.replace("_fixed", "")
                file = file.replace("_", "")
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file}")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file} already exists")
                D_res = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'DFG_D'].values[0]
                try:
                    # Read pdb file
                    atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                    # Detect cavities
                    vertices_load = toml.load(f"{input_directory}/kinases_vertices.toml")
                    vertices = vertices_load["box"]
                    ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, volume_cutoff=volume_cutoff)
                    # Identification of interface residues and cavity volume
                    residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
                    surface, volume, area = pyKVFinder.spatial(cavities)
                    # Select cavities if D residue belongs to them
                    cavity_selection = list()
                    for key, value in residues.items():
                        for residue in value:
                            if int(residue[0]) == D_res:
                                cavity_selection.append(key)
                    # Detect cavity interface residues
                    residues_constitutional = {key:value for key, value in residues.items() if key in cavity_selection}
                    cavity_volume = {key:value for key, value in volume.items() if key in cavity_selection}
                    # Export results
                    pyKVFinder.export(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file}/{file}_output.pdb", cavities, None, vertices, selection=cavity_selection)
                    pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file}/{file}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{kinase}", ligand=None, output=f"{input_directory}/fixed_KV_Files_{volume_cutoff}/{file}/{file}_output.pdb", residues=residues_constitutional, volume=cavity_volume)

                    # Calculate occurrence if number of cavities = 1
                    if occurrence is None and len(cavity_selection) == 1:
                        occurrence = (cavities > 1).astype(int)
                    elif occurrence is not None and len(cavity_selection) == 1:
                        occurrence += (cavities > 1).astype(int)
                    else:
                        print(kinase, len(cavity_selection))
                except:
                    print(f"{file} could not be used to detect cavities")
        else:
            # Load coordinates data from structure
            atomic = np.empty((0, 8))
            for kinase in kinases_files:
                print(kinase)
                new = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                atomic = np.concatenate((atomic, new), axis=0)

            # Dimension 3D grid based on atomic coordinates
            vertices = pyKVFinder.get_vertices(atomic, probe_out=6.0)

            #Convert vertices coordinates to toml
            vertices_list = vertices.tolist()
            vertices_toml = toml.dumps({"box": vertices_list})

            with open(f"{input_directory}/kinases_vertices.toml", "w") as toml_file:
                toml_file.write(vertices_toml)

        # Detect cavities for all kinases and replace B-factor by occurrence of grids
            for kinase, file in zip(kinases_files, kinases_names):
                file = file.replace("_fixed", "")
                file = file.replace("_", "")
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file}")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file} already exists")
                D_res = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'DFG_D'].values[0]
                # Read pdb file
                atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                # Detect cavities
                ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, volume_cutoff=volume_cutoff)
                # Identification of interface residues
                residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
                surface, volume, area = pyKVFinder.spatial(cavities)
                # Select cavities if D residue belongs to them
                cavity_selection = list()
                for key, value in residues.items():
                    for residue in value:
                        if int(residue[0]) == D_res:
                            cavity_selection.append(key)
                # Detect cavity interface residues
                residues_constitutional = {key:value for key, value in residues.items() if key in cavity_selection}
                # Export results
                pyKVFinder.export(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file}/{file}_output.pdb", cavities, None, vertices, selection=cavity_selection)
                pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/{file}/{file}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{kinase}", ligand=None, output=f"{input_directory}/fixed_KV_Files_{volume_cutoff}/{file}/{file}_output.pdb", residues=residues_constitutional, volume=cavity_volume)

                # Calculate occurrence if number of cavities = 1
                if occurrence is None and len(cavity_selection) == 1:
                    occurrence = (cavities > 1).astype(int)
                elif occurrence is not None and len(cavity_selection) == 1:
                    occurrence += (cavities > 1).astype(int)
                else:
                    print(kinase, len(cavity_selection))

    percentage = (occurrence/len(kinases_files)) * 100
    cavities = ((occurrence > 0).astype('int32'))*2
    pyKVFinder.export(fn=f"{input_directory}/fixed_KV_Files_{volume_cutoff}_D/grid_occurrence.pdb", cavities=cavities, surface=None, vertices=vertices, B=percentage)

    print("END: Detect cavities and Calculate ocurrence for D residue")

    print("START: Detect cavities and Calculate ocurrence for F residue")

    # Detect cavities considering F residue from DFG
    if "F" in detection_residues:
        # Create empty array for ocurrence calculation
        occurrence = None
        if os.path.exists(f"{input_directory}/kinases_vertices.toml"):
            for kinase, file in zip(kinases_files, kinases_names):
                file = file.replace("_fixed", "")
                file = file.replace("_", "")
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volum_cutoff}_F/{file}")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file} already exists")
                F_res = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'DFG_Phe'].values[0]
                try:
                    # Read pdb file
                    atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                    # Detect cavities
                    vertices_load = toml.load(f"{input_directory}/kinases_vertices.toml")
                    vertices = vertices_load["box"]
                    ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, volume_cutoff=volume_cutoff)
                    # Identification of interface residues
                    residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
                    surface, volume, area = pyKVFinder.spatial(cavities)
                    # Select cavities if F residue belongs to them
                    cavity_selection = list()
                    for key, value in residues.items():
                        for residue in value:
                            if int(residue[0]) == F_res:
                                cavity_selection.append(key)
                    # Detect cavity interface residues
                    residues_constitutional = {key:value for key, value in residues.items() if key in cavity_selection}
                    # Export results
                    pyKVFinder.export(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file}/{file}_output.pdb", cavities, None, vertices, selection=cavity_selection)
                    pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file}/{file}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{kinase}", ligand=None, output=f"{input_directory}/fixed_KV_Files_{volume_cutoff}/{file}/{file}_output.pdb", residues=residues_constitutional, volume=cavity_volume)

                    # Calculate occurrence if number of cavities = 1
                    if occurrence is None and len(cavity_selection) == 1:
                        occurrence = (cavities > 1).astype(int)
                    elif occurrence is not None and len(cavity_selection) == 1:
                        occurrence += (cavities > 1).astype(int)
                    else:
                        print(kinase, len(cavity_selection))
                except:
                    print(f"{file} could not be used to detect cavities")
        else:
            # Load coordinates data from structure
            atomic = np.empty((0, 8))
            for kinase in kinases_files:
                new = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                atomic = np.concatenate((atomic, new), axis=0)

            # Dimension 3D grid based on atomic coordinates
            vertices = pyKVFinder.get_vertices(atomic, probe_out=6.0)

            #Convert vertices coordinates to toml
            vertices_list = vertices.tolist()
            vertices_toml = toml.dumps({"box": vertices_list})

            with open(f"{input_directory}/kinases_vertices.toml", "w") as toml_file:
                toml_file.write(vertices_toml)

            # Detect cavities for all kinases and replace B-factor by occurrence of grids
            for kinase, file in zip(kinases_files, kinases_names):
                file = file.replace("_fixed", "")
                file = file.replace("_", "")
                try:
                    os.mkdir(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file}")
                except:
                    print(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file} already exists")
                F_res = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'DFG_Phe'].values[0]
                # Read pdb file
                atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{kinase}")
                # Detect cavities
                ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, volume_cutoff=volume_cutoff)
                # Identification of interface residues
                residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
                surface, volume, area = pyKVFinder.spatial(cavities)
                # Select cavities if F residue belongs to them
                cavity_selection = list()
                for key, value in residues.items():
                    for residue in value:
                        if int(residue[0]) == F_res:
                            cavity_selection.append(key)
                # Detect cavity interface residues
                residues_constitutional = {key:value for key, value in residues.items() if key in cavity_selection}
                # Export results
                pyKVFinder.export(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file}/{file}_output.pdb", cavities, None, vertices, selection=cavity_selection)
                pyKVFinder.write_results(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/{file}/{file}_output.toml", input=f"{input_directory}/aligned_PDB_fixed/{kinase}", ligand=None, output=f"{input_directory}/fixed_KV_Files_{volume_cutoff}/{file}/{file}_output.pdb", residues=residues_constitutional, volume=cavity_volume)

                # Calculate occurrence if number of cavities = 1
                if occurrence is None and len(cavity_selection) == 1:
                    occurrence = (cavities > 1).astype(int)
                elif occurrence is not None and len(cavity_selection) == 1:
                    occurrence += (cavities > 1).astype(int)
                else:
                    print(kinase, len(cavity_selection))

    percentage = (occurrence/len(kinases_files)) * 100
    cavities = ((occurrence > 0).astype('int32'))*2
    pyKVFinder.export(fn=f"{input_directory}/fixed_KV_Files_{volume_cutoff}_F/grid_occurrence.pdb", cavities=cavities, surface=None, vertices=vertices, B=percentage)

    print("END: Detect cavities and Calculate ocurrence for F residue")

    return True


# def occurrence_protein (input_directory: str,
#                         reference_structure: str,
#                         volume_cutoff: int):
#     """
#     Run pyKVFinder for selected kinases in order to calculate ocurrence for different groups
#     """
#
#     print("START")
#
#     # Open tsv file containing all human kinases structures
#     kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')
#
#     # Select D residue from DFG in kincore_dataframe
#     kincore_dataframe['DFG_D'] = kincore_dataframe['DFG_Phe'] - 1
#
#     kinases_files = sorted([f for f in os.listdir(f"{input_directory}/aligned_PDB") if f.endswith('.pdb')])
#     kinases_names = sorted([f.replace('.pdb', '') for f in os.listdir(f"{input_directory}/aligned_PDB") if f.endswith('.pdb')])
#
#     # Import pyKVFinder and create results directories
#     import pyKVFinder
#
#     if not os.path.exists(f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao"):
#         os.mkdir(f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao")
#
#     # Ocurrence sets
#     abl1_in_left = ["4TWP_A", "2HZ4_C", "2HZ4_B"]
#     abl1_out_left = ["3QRK_A", "3QRI_A", "3QRJ_B", "5HU9_A"]
#     abl1_in_right = ["2V7A_A", "2F4J_A"]
#     abl1_out_right = ["6NPV_A", "3PYY_A", "3CS9_A", "6NPE_A"]
#     epha2_out = ["8BOH_A", "6HEU_A", "6HEY_A", "8BOM_A", "8BOD_A"]file
#     epha2_in = ["7KJC_A", "1MQB_A", "4TRL_A", "6FNH_A", "5IA1_A", "6FNH_C"]
#     akt_in = ["3CQU_A", "3MVH_A", "4EKL_A", "3MV5_A"]
#     akt_out = ["3QKM_A", "4EJN_A", "6S9W_A", "6HHJ_A"]
#
#     # Merge occurrence sets
#     joined_sets = [abl1_in_left, abl1_out_left, abl1_in_right, abl1_out_right, epha2_out, epha2_in, akt_in, akt_out]
#
#     for set in joined_sets:
#         print(set, len(set))
#         # Create empty array for ocurrence calculation
#         occurrence = None
#         # Detect cavities for all kinases and replace B-factor by occurrence of grids
#         for kinase in set:
#             print(kinase)
#             file = kinase.replace("_", "")
#             try:
#                 os.mkdir(f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao/{file}")
#             except:
#                 print(f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao/{file} already exists")
#             D_res = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'DFG_D'].values[0]
#             # Read pdb file
#             atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB/{kinase}.pdb")
#             # Detect cavities
#             vertices_load = toml.load(f"{input_directory}/kinases_vertices.toml")
#             vertices = vertices_load["box"]
#             ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, volume_cutoff=volume_cutoff)
#             # Identification of interface residues
#             residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
#             # Select cavities if D residue belongs to them
#             cavity_selection = list()
#             for key, value in residues.items():
#                 for residue in value:
#                     if int(residue[0]) == D_res:
#                         cavity_selection.append(key)
#             # Detect cavity interface residues
#             residues_constitutional = {key:value for key, value in residues.items() if key in cavity_selection}
#             print(cavity_selection)
#             # Export results
#             pyKVFinder.export(f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao/{file}/{file}_output.pdb", cavities, None, vertices, selection=cavity_selection)
#             pyKVFinder.write_results(f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao/{file}/{file}_output.toml", input=f"{input_directory}/aligned_PDB/{kinase}", ligand=None, output=f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao/{file}/{file}_output.pdb", residues=residues_constitutional)
#
#             # # Calculate occurrence if number of cavities = 1
#             # if occurrence is None and len(cavity_selection) == 1:
#             #     occurrence = (cavities > 1).astype(int)
#             # elif occurrence is not None and len(cavity_selection) == 1:
#             #     occurrence += (cavities > 1).astype(int)
#             # else:
#             #     print(kinase, len(cavity_selection))
#
#             # Calculate occurrence if number of cavities = 1
#             if occurrence is None and len(cavity_selection) == 1:
#                 if cavity_selection ==['KAA']:
#                     occurrence = (cavities == 2).astype(int)
#                 if cavity_selection == ['KAB']:
#                     occurrence = (cavities == 3).astype(int)
#             elif occurrence is not None and len(cavity_selection) == 1:
#                 if cavity_selection ==['KAA']:
#                     occurrence += (cavities == 2).astype(int)
#                 if cavity_selection == ['KAB']:
#                     occurrence += (cavities == 3).astype(int)
#             else:
#                 print(kinase, len(cavity_selection))
#
#         percentage = (occurrence/len(set)) * 100
#         cavities = ((occurrence > 0).astype('int32'))*2
#         pyKVFinder.export(fn=f"{input_directory}/KV_Files_{volume_cutoff}_D_occurrence_qualificacao/grid_occurrence_{set}.pdb", cavities=cavities, surface=None, vertices=vertices, B=percentage)
#
#     print("END: Detect cavities and Calculate ocurrence")

def cavities_analysis (input_directory: str,
                       volume_cutoff_list: int,
                       detection_residues: list):
    """
    Analyse number of cavities and residues of these cavities
    """

    print("Start cavity analysis")

    kinases_names = sorted([f.replace('.pdb', '') for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])
    consensus_df = pd.DataFrame(columns=['kinases', 'Number of cavities', 'detection method', 'volume cutoff'])

    # Plot bar graph for each ligand radius
    for residue in detection_residues:
        for volume_cutoff in volume_cutoff_list:
            # Calculate number of cavities
            cavities_number_df = pd.DataFrame(columns=['kinases', 'Number of cavities', 'detection method', 'volume cutoff'])
            cavities_dict = dict()
            for file in kinases_names:
                file = file.replace("_fixed", "")
                file = file.replace("_", "")
                try:
                    kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{residue}/{file}/{file}_output.toml")
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':int(len(kv_finder_result['RESULTS']['RESIDUES'].keys())), 'detection method':residue, 'volume cutoff':volume_cutoff}, index=[0])
                except:
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':0, 'detection method':residue, 'volume cutoff':volume_cutoff}, index=[0])
                cavities_number_df = pd.concat([cavities_number_df, file_cavity_number], ignore_index=True)
                consensus_df = pd.concat([consensus_df, file_cavity_number], ignore_index=True)

            # Plot number of cavities for all kinases
            fig = plt.figure(dpi=300)
            fig, ax1 = plt.subplots()
            title = ax1.set_title('Plot of number of cavities')
            ax1.bar(x=cavities_number_df['kinases'], height=cavities_number_df['Number of cavities'], align='center', color='black')
            ax1.set_ylabel('Number of cavities')
            ax1.set_xlabel('kinases')
            plt.xticks(visible=False)
            plt.yticks(np.arange(min(cavities_number_df['Number of cavities']), max(cavities_number_df['Number of cavities'])+3, 1))
            plt.tight_layout()
            plt.savefig(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{residue}/number_of_cavities.png", dpi=300)

            # Plot percentage of quantity of cavities
            cavity_counter_0 = (cavities_number_df['Number of cavities'] == 0).sum()
            cavity_counter_1 = (cavities_number_df['Number of cavities'] == 1).sum()
            cavity_counter_2 = (cavities_number_df['Number of cavities'] == 2).sum()
            # cavity_counter_3 = (cavities_number_df['Number of cavities'] == 3).sum()
            # cavity_counter_4 = (cavities_number_df['Number of cavities'] == 4).sum()

            total_kinases = cavities_number_df.shape[0]

            fig2 = plt.figure(dpi=300)
            fig2, ax2 = plt.subplots()
            title = ax2.set_title('Percentage of number of cavities')
            # bars = ax2.bar(x=[0, 1, 2, 3, 4], height=[((cavity_counter_0 / total_kinases)*100), ((cavity_counter_1 / total_kinases)*100), ((cavity_counter_2 / total_kinases)*100), ((cavity_counter_3 / total_kinases)*100), ((cavity_counter_4 / total_kinases)*100)], align='center', color='black')
            bars = ax2.bar(x=[0, 1, 2], height=[((cavity_counter_0 / total_kinases)*100), ((cavity_counter_1 / total_kinases)*100), ((cavity_counter_2 / total_kinases)*100)], align='center', color='black')
            ax2.set_ylabel('%')
            ax2.set_xlabel('Number of cavities')
            plt.ylim(0, 100)
            plt.xticks([0,1,2], ["0", "1", "2"])
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plt.tight_layout()
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',  # Formatação para exibir apenas uma casa decimal
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords='offset points',
                             ha='center',
                             va='bottom')
            plt.savefig(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{residue}/%_of_cavities.png", dpi=300)

    # Plot graph for all volume cutoff and residues detection
    frequency_table = pd.crosstab(index=consensus_df['detection method'], columns=[consensus_df['volume cutoff'], consensus_df['Number of cavities']], margins=False)
    percentage_table = frequency_table.div(len(kinases_names), axis=0)
    percentage_table = percentage_table.mul(100, axis=0)
    for residue in detection_residues:
        unstacked_table = percentage_table.loc[residue].unstack(level=1)
        unstacked_table = unstacked_table.reset_index()
        unstacked_table = unstacked_table.fillna(0)
        unstacked_table.columns.name = None
        unstacked_table.set_index('volume cutoff', inplace=True)
        ax = unstacked_table.plot.bar(rot=0)
        ax.set_xlabel('Filtro de volume da cavidade (Å)', fontsize=8)
        ax.set_ylabel('Porcentagem', fontsize=8)
        ax.set_title(f'Número de cavidades utilizando o resíduo DFG-{residue} para detecção', pad=20, fontsize=9)
        ax.legend(title='Número de cavidades', fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
        plt.ylim(0, 100)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=8)
        plt.xticks(fontsize=8)
        plt.subplots_adjust(right=0.65)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=5)
        plt.savefig(f"{input_directory}/%_of_cavities_{residue}.png", dpi=300)


    print("End cavity analysis")

    return True

def cavities_analysis_ligand (input_directory: str,
                              radius_list: list):
    """
    Analyse number of cavities
    """

    print("Start cavity analysis")

    kinases_names = sorted([f.replace('.pdb', '') for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])

    # Open tsv file containing all human kinases structures
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')

    # Calculate number of cavities for DFG-In (ATP ligand) and DFG-Out (imatinib)
    cavities_number_df = pd.DataFrame(columns=['kinases', 'Number of cavities', 'ligand used', 'ligand radius'])
    cavities_dict = dict()
    for file in kinases_names:
        file = file.replace("_fixed", "")
        file = file.replace("_", "")
        kinase_conformation = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'SpatialLabel'].values[0]
        if kinase_conformation == "DFGin" or kinase_conformation == "DFGout":
            for ligand_radius in radius_list:
                try:
                    kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_300_ligand/{file}/{file}_{ligand_radius}_output.toml")
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':int(len(kv_finder_result['RESULTS']['RESIDUES'].keys())), 'ligand used':'ATP (DFG_in) and imatinib (DFG_out)', 'ligand radius':ligand_radius}, index=[0])
                except:
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':0, 'ligand used':'ATP (DFG_in) and imatinib (DFG_out)', 'ligand radius':ligand_radius}, index=[0])
                cavities_number_df = pd.concat([cavities_number_df, file_cavity_number], ignore_index=True)

    print(cavities_number_df)

    # Contagem de proteínas por 'ligand radius' e 'Number of cavities'
    cavities_count = cavities_number_df.groupby(['ligand radius', 'Number of cavities']).size().unstack(fill_value=0)
    cavities_count = cavities_count.drop(columns=[3])

    # Calculando o percentual em relação ao total de proteínas para cada 'ligand radius'
    total_per_radius = cavities_count.sum(axis=1)
    cavities_percent = (cavities_count.T / total_per_radius).T * 100

    # Plotando o gráfico de barras agrupadas
    cavities_percent.plot(kind='bar', stacked=False, figsize=(12, 7), color=['skyblue', 'orange', 'green'])

    plt.title("Percentual de proteínas por 'ligand radius' e 'Number of cavities' - Ligante ATP (DFG-In) e Imatinib (DFG-Out)")
    plt.xlabel("Ligand Radius")
    plt.ylabel("Percentual de Proteínas (%)")
    plt.xticks(rotation=45)
    # Colocando a legenda fora do gráfico
    plt.legend(title='Número de cavidades', loc='center left', bbox_to_anchor=(1, 0.5))

    # Adicionando rótulos de valor a cada barra
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f%%', fontsize=9)

    plt.savefig(f"{input_directory}/proteins_by_cavities_and_ligand_radius_ATP_imatinib.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate number of cavities for DFG-In and DFG-Out ((ATP ligand)
    cavities_number_df_1 = pd.DataFrame(columns=['kinases', 'Number of cavities', 'ligand used', 'ligand radius'])
    cavities_dict = dict()
    for file in kinases_names:
        file = file.replace("_fixed", "")
        file = file.replace("_", "")
        kinase_conformation = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'SpatialLabel'].values[0]
        if kinase_conformation == "DFGin" or kinase_conformation == "DFGout":
            for ligand_radius in radius_list:
                try:
                    kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_300_ligand/{file}/ATP_ligand/{file}_{ligand_radius}_output.toml")
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':int(len(kv_finder_result['RESULTS']['RESIDUES'].keys())), 'ligand used':'ATP (DFG_in and DFG_out)', 'ligand radius':ligand_radius}, index=[0])
                except:
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':0, 'ligand used':'ATP (DFG_in and DFG_out)', 'ligand radius':ligand_radius}, index=[0])
                cavities_number_df_1 = pd.concat([cavities_number_df_1, file_cavity_number], ignore_index=True)

    print(cavities_number_df_1)

    # Contagem de proteínas por 'ligand radius' e 'Number of cavities'
    cavities_count = cavities_number_df_1.groupby(['ligand radius', 'Number of cavities']).size().unstack(fill_value=0)

    # Calculando o percentual em relação ao total de proteínas para cada 'ligand radius'
    total_per_radius = cavities_count.sum(axis=1)
    cavities_percent = (cavities_count.T / total_per_radius).T * 100

    # Plotando o gráfico de barras agrupadas
    cavities_percent.plot(kind='bar', stacked=False, figsize=(12, 7), color=['skyblue', 'orange', 'green'])

    plt.title("Percentual de proteínas por 'ligand radius' e 'Number of cavities - Ligante ATP'")
    plt.xlabel("Ligand Radius")
    plt.ylabel("Percentual de Proteínas (%)")
    plt.xticks(rotation=45)
    # Colocando a legenda fora do gráfico
    plt.legend(title='Número de cavidades', loc='center left', bbox_to_anchor=(1, 0.5))

    # Adicionando rótulos de valor a cada barra
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f%%', fontsize=9)

    plt.savefig(f"{input_directory}/proteins_by_cavities_and_ligand_radius_ATP.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate number of cavities for DFG-In and DFG-Out ((imatinib ligand)
    cavities_number_df_2 = pd.DataFrame(columns=['kinases', 'Number of cavities', 'ligand used', 'ligand radius'])
    cavities_dict = dict()
    for file in kinases_names:
        file = file.replace("_fixed", "")
        file = file.replace("_", "")
        kinase_conformation = kincore_dataframe.loc[kincore_dataframe['PDB'] == file, 'SpatialLabel'].values[0]
        if kinase_conformation == "DFGin" or kinase_conformation == "DFGout":
            for ligand_radius in radius_list:
                try:
                    kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_300_ligand/{file}/imatinib_ligand/{file}_{ligand_radius}_output.toml")
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':int(len(kv_finder_result['RESULTS']['RESIDUES'].keys())), 'ligand used':'imatinib (DFG_in and DFG_out)', 'ligand radius':ligand_radius}, index=[0])
                except:
                    file_cavity_number = pd.DataFrame({'kinases':file, 'Number of cavities':0, 'ligand used':'imatinib (DFG_in and DFG_out)', 'ligand radius':ligand_radius}, index=[0])
                cavities_number_df_2 = pd.concat([cavities_number_df_2, file_cavity_number], ignore_index=True)

    print(cavities_number_df_2)

    # Contagem de proteínas por 'ligand radius' e 'Number of cavities'
    cavities_count = cavities_number_df_2.groupby(['ligand radius', 'Number of cavities']).size().unstack(fill_value=0)
    cavities_count = cavities_count.drop(columns=[3])

    # Calculando o percentual em relação ao total de proteínas para cada 'ligand radius'
    total_per_radius = cavities_count.sum(axis=1)
    cavities_percent = (cavities_count.T / total_per_radius).T * 100

    # Plotando o gráfico de barras agrupadas
    cavities_percent.plot(kind='bar', stacked=False, figsize=(12, 7), color=['skyblue', 'orange', 'green'])

    plt.title("Percentual de proteínas por 'ligand radius' e 'Number of cavities - Ligante Imatinib'")
    plt.xlabel("Ligand Radius")
    plt.ylabel("Percentual de Proteínas (%)")
    plt.xticks(rotation=45)
    # Colocando a legenda fora do gráfico
    plt.legend(title='Número de cavidades', loc='center left', bbox_to_anchor=(1, 0.5))

    # Adicionando rótulos de valor a cada barra
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f%%', fontsize=9)

    plt.savefig(f"{input_directory}/proteins_by_cavities_and_ligand_radius_imatinib.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("End cavity analysis")

    return True

def cavities_pymol (input_directory: str,
                reference_structure: str,
                volume_cutoff_list: int,
                detection_residues: list):
    """
    Plot cavities in pymol using gradient for grid occurrence
    """

    for volume_cutoff in volume_cutoff_list:
        cmd.load(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{detection_residues}/grid_occurrence.pdb")
        cmd.hide("everything", "b < 10")
        cmd.spectrum ("b", "blue_white_red")
        cmd.fetch('3qri')
        cmd.remove("solvent")
        cmd.remove("chain B")
        cmd.set("cartoon_transparency", 0.5)
        cmd.center()
        cmd.rotate([1,1,0], 135)
        cmd.rotate([0,1,0], 45)
        cmd.zoom()
        cmd.png(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{detection_residues}/grid_occurrence_spectrum.png", width=1200, dpi=300)
        cmd.save(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{detection_residues}/grid_occurrence_spectrum.pse", selection='(all)')
        cmd.reinitialize()

def cavities_pymol_ligand (input_directory: str,
                           reference_structure: str,
                           ligand_cutoff_list: int,
                           DFG_list: str):
    """
    Plot cavities in pymol using gradient for grid occurrence
    """

    DFG_list = ['in', 'out']
    # for ligand_cutoff in ligand_cutoff_list:
    #     for state in DFG_list:
    #         cmd.load(f"{input_directory}/fixed_KV_Files_300_ligand/grid_occurrence_{ligand_cutoff}_{state}.pdb")
    #         cmd.hide("everything", "b < 10")
    #         cmd.spectrum ("b", "blue_white_red")
    #         cmd.fetch('3qri')
    #         cmd.remove("solvent")
    #         cmd.remove("chain B")
    #         cmd.remove("NA")
    #         cmd.remove("organic")
    #         cmd.set("cartoon_transparency", 0.5)
    #         cmd.center()
    #         cmd.rotate([1,0,0], 180)
    #         cmd.rotate([1,1,0], 135)
    #         cmd.rotate([0,1,0], 45)
    #         cmd.zoom()
    #         cmd.png(f"{input_directory}/fixed_KV_Files_300_ligand/grid_occurrence_{ligand_cutoff}_{state}_spectrum.png", width=1200, dpi=300)
    #         cmd.save(f"{input_directory}/fixed_KV_Files_300_ligand/grid_occurrence_{ligand_cutoff}_{state}_spectrum.pse", selection='(all)')
    #         cmd.reinitialize()

    # for ligand_cutoff in ligand_cutoff_list:
    #     for state in DFG_list:
    #         cmd.load(f"{input_directory}/fixed_KV_Files_300_ligand/ATP_ligand/grid_occurrence_{ligand_cutoff}_{state}.pdb")
    #         cmd.hide("everything", "b < 10")
    #         cmd.spectrum ("b", "blue_white_red")
    #         cmd.fetch('3qri')
    #         cmd.remove("solvent")
    #         cmd.remove("chain B")
    #         cmd.remove("NA")
    #         cmd.remove("organic")
    #         cmd.set("cartoon_transparency", 0.5)
    #         cmd.center()
    #         cmd.rotate([1,0,0], 180)
    #         cmd.rotate([1,1,0], 135)
    #         cmd.rotate([0,1,0], 45)
    #         cmd.zoom()
    #         cmd.png(f"{input_directory}/fixed_KV_Files_300_ligand/ATP_ligand/grid_occurrence_{ligand_cutoff}_{state}_spectrum_ATP.png", width=1200, dpi=300)
    #         cmd.save(f"{input_directory}/fixed_KV_Files_300_ligand/ATP_ligand/grid_occurrence_{ligand_cutoff}_{state}_spectrum_ATP.pse", selection='(all)')
    #         cmd.reinitialize()

    for ligand_cutoff in ligand_cutoff_list:
        for state in DFG_list:
            cmd.load(f"{input_directory}/fixed_KV_Files_300_ligand/imatinib_ligand/grid_occurrence_{ligand_cutoff}_{state}.pdb")
            cmd.hide("everything", "b < 10")
            cmd.spectrum ("b", "blue_white_red")
            cmd.fetch('3qri')
            cmd.remove("solvent")
            cmd.remove("chain B")
            cmd.remove("NA")
            cmd.remove("organic")
            cmd.set("cartoon_transparency", 0.5)
            cmd.center()
            cmd.rotate([1,0,0], 180)
            cmd.rotate([1,1,0], 135)
            cmd.rotate([0,1,0], 45)
            cmd.zoom()
            cmd.png(f"{input_directory}/fixed_KV_Files_300_ligand/imatinib_ligand/grid_occurrence_{ligand_cutoff}_{state}_spectrum_imatinib.png", width=1200, dpi=300)
            cmd.save(f"{input_directory}/fixed_KV_Files_300_ligand/imatinib_ligand/grid_occurrence_{ligand_cutoff}_{state}_spectrum_imatinib.pse", selection='(all)')
            cmd.reinitialize()

def αC_extract_coords (input_directory: str,
                       volume_cutoff: int,
                       detection_residues: str,
                       verbose: bool = False) -> dict:
    """
    :Extract coordinates from cavity residues
    :param cavities_dir: Directory containing trajectory frames and KVFinderMD output
    :param verbose: provides additional details as to what the function is doing
    :type cavities_dir: str
    :type verbose: bool
    :return: dictionary containing coordinates from alpha carbons of cavity residues for all frames
    """

    # Get all kinases (files that endswith .pdb)
    kinases_names = sorted([f for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])

    # Create empty dictionary for cavity info of all frames
    kinases_coords = dict()

    print("START EXTRACT COORDS")

    # Extract C-alpha coordinates from .pdb files using BioPython only for kinases with one cavity
    for file in kinases_names:
        kinase = file.upper()
        kinase = kinase.replace("_FIXED", "")
        kinase = kinase.replace("_", "")
        kinase = kinase.replace(".PDB", "")
        try:
            kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{detection_residues}/{kinase}/{kinase}_output.toml")
            if len(kv_finder_result['RESULTS']['RESIDUES'].keys()) != 1:
                print(f"{kinase} has {len(kv_finder_result['RESULTS']['RESIDUES'].keys())} cavities")
            else:
                dict_coords = dict()
                parser = PDBParser(PERMISSIVE=1)
                structure = parser.get_structure(f"{file}", f"{input_directory}/aligned_PDB_fixed/{file}")
                for model in structure.get_list():
                    for chain in model.get_list():
                        for residue in chain.get_list():
                            if residue.has_id("CA"):
                                dict_coords["{}_{}".format(residue.get_id()[1], residue.get_full_id()[2])] = residue["CA"].get_coord()
                # Create dictionary for each cavity containing residue number, chain and C-alpha coordinates
                kv_info = dict()
                for kv_tag in kv_finder_result['RESULTS']['RESIDUES'].keys():
                    kv_info[kv_tag] = dict()
                    for residue_info in kv_finder_result['RESULTS']['RESIDUES'][kv_tag]:
                        residue_id = residue_info[0]+"_"+residue_info[1]
                        try:
                            kv_info[kv_tag][residue_id] = dict_coords[residue_id]
                        except:
                            print(f"{file} has no {residue_id} in pdb file")
                    kinases_coords[kinase] = kv_info
        except:
            print(f"There is no {kinase} file")

    if verbose:
        print(f"> xyz coordinates were parsed and a dictionary containing alpha carbon coordinates was created")

    print("END EXTRACT COORDS")

    return kinases_coords

def αC_extract_coords_ligand (input_directory: str,
                       volume_cutoff: int,
                       ligand_cutoff: int,
                       verbose: bool = False) -> dict:
    """
    :Extract coordinates from cavity residues
    :param cavities_dir: Directory containing trajectory frames and KVFinderMD output
    :param verbose: provides additional details as to what the function is doing
    :type cavities_dir: str
    :type verbose: bool
    :return: dictionary containing coordinates from alpha carbons of cavity residues for all frames
    """

    # Get all kinases (files that endswith .pdb)
    kinases_names = sorted([f for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])

    # Open tsv file containing all human kinases structures
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')

    # Create empty dictionary for cavity info of all frames
    kinases_coords = dict()

    print("START EXTRACT COORDS")

    # Extract C-alpha coordinates from .pdb files using BioPython only for kinases with one cavity
    for file in kinases_names:
        kinase = file.upper()
        kinase = kinase.replace("_FIXED", "")
        kinase = kinase.replace("_", "")
        kinase = kinase.replace(".PDB", "")
        kinase_conformation = kincore_dataframe.loc[kincore_dataframe['PDB'] == kinase, 'SpatialLabel'].values[0]
        if kinase_conformation == "DFGin" or kinase_conformation == "DFGout":
            try:
                kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{kinase}/ATP_ligand/{kinase}_{ligand_cutoff}_output.toml")
                if len(kv_finder_result['RESULTS']['RESIDUES'].keys()) != 1:
                    print(f"{kinase} has {len(kv_finder_result['RESULTS']['RESIDUES'].keys())} cavities")
                else:
                    dict_coords = dict()
                    parser = PDBParser(PERMISSIVE=1)
                    structure = parser.get_structure(f"{file}", f"{input_directory}/aligned_PDB_fixed/{file}")
                    for model in structure.get_list():
                        for chain in model.get_list():
                            for residue in chain.get_list():
                                if residue.has_id("CA"):
                                    dict_coords["{}_{}".format(residue.get_id()[1], residue.get_full_id()[2])] = residue["CA"].get_coord()

                # Create dictionary for each cavity containing residue number, chain and C-alpha coordinates
                    kv_info = dict()
                    for kv_tag in kv_finder_result['RESULTS']['RESIDUES'].keys():
                        kv_info[kv_tag] = dict()
                        for residue_info in kv_finder_result['RESULTS']['RESIDUES'][kv_tag]:
                            residue_id = residue_info[0]+"_"+residue_info[1]
                            try:
                                kv_info[kv_tag][residue_id] = dict_coords[residue_id]
                            except:
                                print(f"{file} has no {residue_id} in pdb file")
                        kinases_coords[kinase] = kv_info
            except:
                print(f"There is no {kinase} file")

    if verbose:
        print(f"> xyz coordinates were parsed and a dictionary containing alpha carbon coordinates was created")

    print("END EXTRACT COORDS")

    return kinases_coords

def αC_calc_distance (kinases_coords: dict,
                      verbose: bool = False) -> dict:
    """
    :Calculate Euclidian distance between C-alpha from cavity residues
    :param frames_coords: dictionary containing C-alpha coordinates from cavity residues for all frames
    :param verbose:provides additional details as to what the function is doing
    :type frames_coords: dict
    :type verbose: bool
    :return: dictionary containing Euclidian distance between C-alpha from cavity residues
    """

    print("START CALCULATE DISTANCE")

    # Create xyz coordinates array for residues
    for kv_file in kinases_coords.keys():
        for kv_tag in kinases_coords[kv_file].keys():
            coordinates_array = np.empty((0,3))
            for coordinates in kinases_coords[kv_file][kv_tag].values():
                coordinates_array = np.append(coordinates_array, np.array([coordinates]), axis=0)

            # Create combinations using coordinates and residues
            coordinates_combination = combinations(coordinates_array, 2)
            residues_combination = combinations(kinases_coords[kv_file][kv_tag].keys(), 2)

            # Calculate Euclidian distances between residues for each cavity
            residues_distance = dict()
            for coordinates_pair, residue_pair in zip(coordinates_combination, residues_combination):
                distance = np.linalg.norm(coordinates_pair[0]-coordinates_pair[1])
                residues_distance.update({residue_pair:distance})
            kinases_coords[kv_file][kv_tag]= residues_distance

    if verbose:
        print (f"> Distances between alpha carbons were successfully calculated")

    print("END CALCULATE DISTANCE")

    print (kinases_coords)

    return kinases_coords

def create_graphs (kinases_coords: dict,
                   αC_cutoff: float,
                   input_directory: str,
                   volume_cutoff: int,
                   detection_residues: str,
                   base_name: str,
                   verbose: bool = False) -> dict:
    """
    :Create graphs for representation of cavity residues
    :param kinases_coords: dictionary containing Euclidian distance between C-alpha from cavity residues
    :param input_directory: Directory containing trajectory frames and KVFinderMD output
    :param base_name: Base output file name
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type cavities_dir: str
    :type αC_cutoff: float
    :type base_name: str
    :type verbose: bool
    :return graphs saved in output_dir
    """

    # Create directory for graph saving
    try:
        os.mkdir(f"{input_directory}/fixed_graph_properties_{αC_cutoff}")
    except FileExistsError:
        pass

    try:
        os.mkdir(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/aC_graph")
    except FileExistsError:
        pass

    try:
        os.mkdir(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/aC_graph_attribute")
    except FileExistsError:
        pass

    for kv_file in kinases_coords.keys():
        try:
            os.mkdir(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/{kv_file}")
        except FileExistsError:
            pass

    print ("START GRAPH CREATION")

    # Create graph for each cavity containing all residues as vertices
    for kv_file in kinases_coords.keys():
        for kv_tag in kinases_coords[kv_file].keys():
            αC_graph = Graph()
            αC_graph_attribute = Graph()

            # Name graph according to the kv_tag
            αC_graph["name"] = kv_tag
            αC_graph_attribute["name"] = kv_tag

            # Add name vertex according to the residues
            kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_{detection_residues}/{kv_file}/{kv_file}_output.toml")
            for residue_info in kv_finder_result['RESULTS']['RESIDUES'][kv_tag]:
                residue_id = residue_info[0]+"_"+residue_info[1]
                αC_graph.add_vertex(name=residue_id)
                αC_graph_attribute.add_vertex(name=residue_id)

            # Add edges and attributes according to distance between alpha Carbons
            for residue_pair, distance in kinases_coords[kv_file][kv_tag].items():
                if distance < αC_cutoff:
                    αC_graph.add_edge(residue_pair[0], residue_pair[1], color="black", interaction="aC distance")
                    αC_graph_attribute.add_edge(residue_pair[0], residue_pair[1], aC_distance=distance, color="black", interaction="aC distance")

            # Save graph
            αC_graph.write_graphml(f=f"{input_directory}/fixed_graph_properties_{αC_cutoff}/{kv_file}/{base_name}_{kv_file}_{kv_tag}_aC_graph.graphml")
            αC_graph.write_leda(f=f"{input_directory}/fixed_graph_properties_{αC_cutoff}/aC_graph/{base_name}_{kv_file}_{kv_tag}_aC_graph.gw")
            αC_graph_attribute.write_graphml(f=f"{input_directory}/fixed_graph_properties_{αC_cutoff}/{kv_file}/{base_name}_{kv_file}_{kv_tag}_aC_graph_attribute.graphml")
            αC_graph_attribute.write_leda(f=f"{input_directory}/fixed_graph_properties_{αC_cutoff}/aC_graph_attribute/{base_name}_{kv_file}_{kv_tag}_aC_graph.gw", weights='aC_distance')

    if verbose:
        print(f"> graphs were created for all frames")

    print ("END GRAPH CREATION")

    return True

def create_graphs_ligand (kinases_coords: dict,
                   αC_cutoff: float,
                   input_directory: str,
                   volume_cutoff: int,
                   ligand_cutoff: int,
                   verbose: bool = False) -> dict:
    """
    :Create graphs for representation of cavity residues
    :param kinases_coords: dictionary containing Euclidian distance between C-alpha from cavity residues
    :param input_directory: Directory containing trajectory frames and KVFinderMD output
    :param base_name: Base output file name
    :param verbose: provides additional details as to what the function is doing
    :type frames_coords: dict
    :type cavities_dir: str
    :type αC_cutoff: float
    :type base_name: str
    :type verbose: bool
    :return graphs saved in output_dir
    """

    # Create directory for graph saving
    try:
        os.mkdir(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_ATP_ligand")
    except FileExistsError:
        pass

    try:
        os.mkdir(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_ATP_ligand/aC_graph")
    except FileExistsError:
        pass

    for kv_file in kinases_coords.keys():
        try:
            os.mkdir(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_ATP_ligand/{kv_file}")
        except FileExistsError:
            pass

    print ("START GRAPH CREATION")

    # Create graph for each cavity containing all residues as vertices
    for kv_file in kinases_coords.keys():
        for kv_tag in kinases_coords[kv_file].keys():
            αC_graph = Graph()

            # Name graph according to the kv_tag
            αC_graph["name"] = kv_tag

            # Add name vertex according to the residues
            kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_{volume_cutoff}_ligand/{kv_file}/ATP_ligand/{kv_file}_{ligand_cutoff}_output.toml")
            for residue_info in kv_finder_result['RESULTS']['RESIDUES'][kv_tag]:
                residue_id = residue_info[0]+"_"+residue_info[1]
                αC_graph.add_vertex(name=residue_id)

            # Add edges and attributes according to distance between alpha Carbons
            for residue_pair, distance in kinases_coords[kv_file][kv_tag].items():
                if distance < αC_cutoff:
                    αC_graph.add_edge(residue_pair[0], residue_pair[1], color="black", interaction="aC distance")

            # Save graph
            αC_graph.write_graphml(f=f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_ATP_ligand/{kv_file}/{kv_file}_{kv_tag}_aC_graph.graphml")
            αC_graph.write_leda(f=f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_ATP_ligand/aC_graph/{kv_file}_{kv_tag}_aC_graph.gw")

    if verbose:
        print(f"> graphs were created for all frames")

    print ("END GRAPH CREATION")

    return True

def add_volume_kincore_database (input_directory: str,
                                 ligand_cutoff_list: int):
    """ Mergeia os dados de volume de cavidade presentes nos arquivos .toml com a planilha baixada do kincore,
    que possui a conformação e o código pdb das proteinas
    """

    # Passo 1: Ler o arquivo .tab
    tabela = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')

    # Criar uma lista para armazenar os volumes
    pdb_volumes = dict()

    # Passo 2: Para cada código PDB na tabela, buscar o arquivo .toml correspondente
    for ligand_cutoff in ligand_cutoff_list:
        for pdb_code in tabela['PDB']:
            toml_file = f"{input_directory}/fixed_KV_Files_300_ligand/{pdb_code}/ATP_ligand/{pdb_code}_{ligand_cutoff}_output.toml"

            if os.path.exists(toml_file):
                # Ler o arquivo .toml
                toml_data = toml.load(toml_file)

                # Acessar a seção VOLUME em RESULTS
                volume_section = toml_data.get('RESULTS', {}).get('VOLUME', {})

                # Verificar se há apenas uma chave na seção VOLUME
                if len(volume_section) == 1:

                    # Se houver apenas uma cavidade, pegar o valor
                    volume = next(iter(volume_section.values()), None)
                else:
                    # Se houver mais de uma cavidade, não adiciona o volume (define como None)
                    volume = None

            else:
                volume = None  # Se o arquivo .toml não for encontrado, define como None

            print(pdb_code, volume)

            # Armazenar o volume no dicionário
            pdb_volumes[pdb_code] = volume

            # Passo 3: Criar uma nova coluna 'Volume_Cavidade' na tabela associando cada PDB ao volume correto
            tabela['Cavity_vol'] = tabela['PDB'].map(pdb_volumes)

        # Passo 4: Salvar a tabela atualizada em um novo arquivo .tab
        tabela.to_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_{ligand_cutoff}_ATP.tab", sep='\t', index=False)


def gcd_11_read (input_directory: str,
                 αC_cutoff: float):

    """
    Read gcd_11 output file and transform it in a Dataframe
    """
    print("START gcd_11.txt read")

    # Read txt file and rename columns
    gcd_df = pd.read_csv(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/aC_graph/gcd11.txt", delimiter="\t")
    column_index = gcd_df[gcd_df.columns[1:]].columns
    new_column_index = [col[6] for col in column_index.str.split("_")]
    new_column_index.insert(0, "kinase")
    gcd_df.columns = new_column_index

    # Rename rows
    new_row_label = [col[6] for col in gcd_df["kinase"].str.split("_")]
    gcd_df["kinase"] = new_row_label

    print(gcd_df)

    print("END gcd_11.txt read")

    return gcd_df

def agglomerative_clusterization (input_directory:str,
                                  gcd_df: pd.core.frame.DataFrame,
                                  blaminus_filter: bool,
                                  family_clusterization: bool,
                                  protein_clusterization: bool,
                                  linkage_type: str,
                                  αC_cutoff:float):

    """
    Perform agglomerative clusterization using gcd 11 distances and plot dendrogram - filter
    """

    print("START dendrogram creation for all kinases")

    import matplotlib.patches as mpatches

    # Transform protein names in index values
    gcd_df = gcd_df.set_index(gcd_df.columns[0])
    gcd_df.index.name = "Kinases"

    # indices_desejados = ['6S9WA', '3QKLA', '4GV1A', '3MV5A']
    #
    # # Método 1: Usando loc
    # df_filtrado = gcd_df.loc[indices_desejados, indices_desejados]
    # print(df_filtrado)

    # Add cavity volume label to proteins
    kinases_names = sorted([f.replace('.pdb', '') for f in os.listdir(f"{input_directory}/aligned_PDB_fixed") if f.endswith('.pdb')])
    cavities_volume_df = pd.DataFrame(columns=['PDB', 'Volume da cavidade'])
    for file in kinases_names:
        file = file.replace("_fixed", "")
        file = file.replace("_", "")
        try:
            kv_finder_result = toml.load(f"{input_directory}/fixed_KV_Files_300_D/{file}/{file}_output.toml")
            file_cavity_volume = pd.DataFrame({'PDB':file, 'Volume da cavidade':kv_finder_result['RESULTS']['VOLUME'].values()}, index=[0])
            cavities_volume_df = pd.concat([cavities_volume_df, file_cavity_volume], ignore_index=True)
        except:
            pass

    # Add labels to proteins
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All.tab", sep='\t')
    kincore_dataframe = kincore_dataframe[['Group', 'Gene', 'PDB', 'SpatialLabel', 'DihedralLabel','C-helix', 'Ligand', 'LigandType']]
    kincore_dataframe = kincore_dataframe.loc[kincore_dataframe['PDB'].isin(gcd_df.index)]
    kincore_dataframe = kincore_dataframe.loc[~kincore_dataframe['PDB'].duplicated()]
    kincore_dataframe = pd.merge(kincore_dataframe, cavities_volume_df, on='PDB')
    kincore_dataframe = kincore_dataframe.set_index('PDB')

    # Filter of BLAminus (active) and BBAminus (inactive) structures
    kincore_dataframe_blaminus = kincore_dataframe[kincore_dataframe['DihedralLabel'].isin(['BLAminus', 'BBAminus'])]
    if blaminus_filter == True:
        gcd_df = gcd_df[gcd_df.index.isin(kincore_dataframe_blaminus.index)]
        gcd_df = gcd_df.loc[:, kincore_dataframe_blaminus.index]
        reordered_columns = gcd_df.index
        gcd_df = gcd_df[reordered_columns]

    # Create linkage matrix for dendrogram construction (blaminus structures)
    condensed_matrix = squareform(gcd_df)
    linkage_matrix = linkage(condensed_matrix, method=linkage_type)

    # Create color labels for range values of Cavity Volume
    bins_labels = ['0-500', '501-1000', '1001-1500', '1501-2000', '>2001']
    bins = [0, 500, 1000, 1500, 2000, float('inf')]
    labels = ['#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000']
    row_colors_cv = pd.cut(kincore_dataframe['Volume da cavidade (Å3)'], bins=bins, labels=labels, right=False)

    # Create color labels for unique values of SpatialLabel
    sl = dict(zip(kincore_dataframe['SpatialLabel'].unique(), ["#00FFFF", "#008000", "#DDA0DD", "#FF0000"]))
    row_colors_sl = kincore_dataframe['SpatialLabel'].map(sl)
    # sl = dict(zip(['DFGin', 'DFGout'], ["#00FFFF", "#008000"]))
    # row_colors_sl = kincore_dataframe['SpatialLabel'].map(sl)

    # Create color labels for unique values of Group
    gr = dict(zip(kincore_dataframe['Group'].unique(), ["#069AF3", "#380282", "#FF796C", "#A9561E", '#C79FEF', '#000000', '#0000FF', '#C5C9C7', '#A52A2A']))
    row_colors_gr = kincore_dataframe['Group'].map(gr)

    # # Create color labels for unique values of C-helix
    # ch = dict(zip(kincore_dataframe['C-helix'].unique(), ["#7FFF00", "#C1F80A", "#008080"]))
    # row_colors_ch = kincore_dataframe['C-helix'].map(ch)

    # # Create color labels for unique values of LigandType
    # ligand_type_list = ['Type1', 'Type2', 'Type3', 'No_ligand']
    # kincore_dataframe['LigandType'] = np.where(kincore_dataframe['LigandType'].isin(ligand_type_list), kincore_dataframe['LigandType'], 'Other')
    # lt = dict(zip(kincore_dataframe['LigandType'].unique(), ["#ED0DD9", "#FF4500", "#FFA500", "#DC143C", "#FBDD7E"]))
    # row_colors_lt = kincore_dataframe['LigandType'].map(lt)

    # # Create color labels for unique values of DihedralLabel
    # dl = dict(zip(kincore_dataframe['DihedralLabel'].unique(), ["#029386", "#F97306", "#67E50E", "#6E750E", "#650021", "#E6DAA6", "#AAA662", "#FF81C0", "#DAA520"]))
    # row_colors_dl = kincore_dataframe['DihedralLabel'].map(dl)

    # Join dataframes
    joined_colors = pd.concat([row_colors_cv, row_colors_sl, row_colors_gr], axis=1)

    # Plot dendrogram using color bars to identify labels (without heatmap)
    sns.set_theme(color_codes=True)
    g = sns.clustermap(gcd_df, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, dendrogram_ratio=(0.05, 0.35), figsize=(12, 8), cmap="viridis_r")
    g.ax_col_dendrogram.set_title(f"Quinases - {αC_cutoff}Å cutoff", pad=-350, fontsize='large')
    # Remove heatmap
    g.ax_heatmap.remove()
    # Adjust fontsize of colorbar label
    g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_yticklabels(), fontsize=12)

    # Add legend - Cavity Volume
    patchList_cavvol = []
    for bin, label in zip(bins_labels, labels):
        data_key = mpatches.Patch(color=label, label=bin)
        patchList_cavvol.append(data_key)
    cavvol_legend = plt.gca().legend(title="Volume da cavidade", handles=patchList_cavvol, loc='lower right', bbox_to_anchor=(1.0, -4))
    plt.gca().add_artist(cavvol_legend)

    # Add legend - SpatialLabel
    patchList_spatiallabel = []
    for label in kincore_dataframe['SpatialLabel'].unique():
        data_key = mpatches.Patch(color=sl[label], label=label)
        patchList_spatiallabel.append(data_key)
    spatiallabel_legend = plt.gca().legend(title="Spatial Label", handles=patchList_spatiallabel, loc='lower center', bbox_to_anchor=(0.5, -4))
    plt.gca().add_artist(spatiallabel_legend)

    # # Add legend - C-helix
    # patchList_chelix = []
    # for label in kincore_dataframe['C-helix'].unique():
    #     data_key = mpatches.Patch(color=ch[label], label=label)
    #     patchList_chelix.append(data_key)
    # chelix_legend = plt.gca().legend(title="C-helix", handles=patchList_chelix, loc='lower center', bbox_to_anchor=(0.5, -4))
    # plt.gca().add_artist(chelix_legend)

    # # Add legend - LigandType
    # patchList_ligandtype = []
    # for label in kincore_dataframe['LigandType'].unique():
    #     data_key = mpatches.Patch(color=lt[label], label=label)
    #     patchList_ligan1000dtype.append(data_key)
    # ligandtype_legend = plt.gca().legend(title="LigandType", handles=patchList_ligandtype, loc='lower center', bbox_to_anchor=(0.7, -4))
    # plt.gca().add_artist(ligandtype_legend)

    # # Add legend - Rotamers
    # patchList_dihedral = []
    # for label in kincore_dataframe['DihedralLabel'].unique():
    #     data_key = mpatches.Patch(color=dl[label], label=label)
    #     patchList_dihedral.append(data_key)
    # dihedral_legend = plt.gca().legend(title="DihedralLabel", handles=patchList_dihedral, loc='lower right', bbox_to_anchor=(1.0, -4))
    # plt.gca().add_artist(dihedral_legend)

    # Add legend - Group
    patchList_group = []
    for label in kincore_dataframe['Group'].unique():
        data_key = mpatches.Patch(color=gr[label], label=label)
        patchList_group.append(data_key)
    plt.gca().legend(title="Grupo", handles=patchList_group, loc='lower left', bbox_to_anchor=(0, -4))

    if blaminus_filter == True:
        plt.savefig(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/fixed_DFGIn_Out_blaminus_dendrogram_{αC_cutoff}_{linkage_type}.png", dpi=300)

    else:
        plt.savefig(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/fixed_DFGIn_Out_dendrogram_{αC_cutoff}_{linkage_type}.png", dpi=300)

    print("END dendrogram creation for all kinases")

    # Generate one dendrogram per family
    if family_clusterization == True:
        for family in kincore_dataframe['Group'].unique():
            print(f"START dendrogram creation for {family}")

            if blaminus_filter == True:
                # Analyze only blaminus and bbaminus structures
                kincore_dataframe_family = kincore_dataframe_blaminus[kincore_dataframe_blaminus['Group'] == family]
            else:
                kincore_dataframe_family = kincore_dataframe[kincore_dataframe['Group'] == family]

            gcd_df_family = gcd_df[gcd_df.index.isin(kincore_dataframe_family.index)]
            gcd_df_family = gcd_df_family.loc[:, kincore_dataframe_family.index]
            reordered_columns = gcd_df_family.index
            gcd_df_family = gcd_df_family[reordered_columns]

            if gcd_df_family.empty:
                pass
            else:
                condensed_matrix = squareform(gcd_df_family)
                linkage_matrix = linkage(condensed_matrix, method=linkage_type)

                # Join dataframes
                joined_colors = pd.concat([row_colors_sl, row_colors_cv], axis=1)

                # Plot dendrogram using color bars to identify labels (without heatmap)
                sns.set_theme(color_codes=True)
                g = sns.clustermap(gcd_df_family, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, dendrogram_ratio=(0.05, 0.35), figsize=(12, 9), cmap="viridis_r")
                # Remove heatmap
                g.ax_heatmap.remove()
                g.ax_col_dendrogram.set_title(f"{family} Kinases - {αC_cutoff}Å cutoff", pad=-350, fontsize='large')
                # Adjust fontsize of colorbar label
                g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_yticklabels(), fontsize=12)

                # Add legend - SpatialLabel
                patchList_spatiallabel = []
                for label in kincore_dataframe['SpatialLabel'].unique():
                    data_key = mpatches.Patch(color=sl[label], label=label)
                    patchList_spatiallabel.append(data_key)
                spatiallabel_legend = plt.gca().legend(title="Spatial Label", handles=patchList_spatiallabel, loc='lower center', bbox_to_anchor=(0.4, -4))
                plt.gca().add_artist(spatiallabel_legend)

                # Add legend - Cavity Volume
                patchList_cavvol = []
                for bin, label in zip(bins_labels, labels):
                    data_key = mpatches.Patch(color=label, label=bin)
                    patchList_cavvol.append(data_key)
                cavvol_legend = plt.gca().legend(title="Volume da cavidade", handles=patchList_cavvol, loc='lower right', bbox_to_anchor=(0.8, -4))
                plt.gca().add_artist(cavvol_legend)

                # # Add legend - C-helix
                # patchList_chelix = []
                # for label in kincore_dataframe['C-helix'].unique():
                #     data_key = mpatches.Patch(color=ch[label], label=label)
                #     patchList_chelix.append(data_key)
                # chelix_legend = plt.gca().legend(title="C-helix", handles=patchList_chelix, loc='lower left', bbox_to_anchor=(0.0, -4))
                # plt.gca().add_artist(chelix_legend)

                # # Add legend - Rotamers
                # patchList_dihedral = []
                # for label in kincore_dataframe['DihedralLabel'].unique():
                #     data_key = mpatches.Patch(color=dl[label], label=label)
                #     patchList_dihedral.append(data_key)
                # dihedral_legend = plt.gca().legend(title="DihedralLabel", handles=patchList_dihedral, loc='lower center', bbox_to_anchor=(0.7, -4))
                # plt.gca().add_artist(dihedral_legend)

                # # Add legend - LigandType
                # patchList_ligandtype = []
                # for label in kincore_dataframe['LigandType'].unique():
                #     data_key = mpatches.Patch(color=lt[label], label=label)
                #     patchList_ligandtype.append(data_key)
                # ligandtype_legend = plt.gca().legend(title="LigandType", handles=patchList_ligandtype, loc='lower right', bbox_to_anchor=(1.0, -4))
                # plt.gca().add_artist(ligandtype_legend)

                if blaminus_filter == True:
                    plt.savefig(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/fixed_DFGIn_Out__blaminus_dendrogram_{family}_{αC_cutoff}_{linkage_type}.png", dpi=300)
                else:
                    plt.savefig(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/heatmap_fixed_DFGIn_Out_dendrogram_{family}_{αC_cutoff}_{linkage_type}.png", dpi=300)

                print(f"END dendrogram creation for {family}")

    # Plot dendrogram for specific proteins
    if protein_clusterization == True:
        for group in kincore_dataframe['Group'].unique():
            print(f"START dendrogram creation for {group}")
            # Filter kincore dataframe for groups
            if blaminus_filter == True:
                kincore_dataframe_family = kincore_dataframe_blaminus[kincore_dataframe_blaminus['Group'] == group]
            else:
                kincore_dataframe_family = kincore_dataframe[kincore_dataframe['Group'] == group]
            count_proteins = kincore_dataframe_family['Gene'].value_counts()

            # Filter kincore dataframe for proteins
            for protein in count_proteins.index:
                if count_proteins[protein] > 20:
                    kincore_dataframe_protein = kincore_dataframe_family[kincore_dataframe_family['Gene'] == protein]
                    if kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGout', 0) > 3:
                        print(f"{protein} has {kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGout', 0)} structures in DFG out state")
                        print(f"{protein} has {kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGin', 0)} structures in DFG in state")

                        # Filter and ordenate gcd dataframe per protein
                        gcd_df_protein = gcd_df[gcd_df.index.isin(kincore_dataframe_protein.index)]
                        gcd_df_protein = gcd_df_protein.loc[:, kincore_dataframe_protein.index]
                        reordered_columns = gcd_df_protein.index
                        gcd_df_protein = gcd_df_protein[reordered_columns]

                        # Create linkage matrix for dendrogram construction
                        condensed_matrix = squareform(gcd_df_protein)
                        linkage_matrix = linkage(condensed_matrix, method=linkage_type)

                        # Create list of labels for all leaves
                        labels_leaves = list(gcd_df_protein.index)

                        # Join dataframes
                        joined_colors = pd.concat([row_colors_sl, row_colors_cv], axis=1)

                        # Plot dendrogram using color bars to identify labels (without heatmap)
                        sns.set_theme(color_codes=True)
                        sns.set(font_scale=0.7)
                        # g = sns.clustermap(gcd_df_protein, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, dendrogram_ratio=(0.05, 0.35), figsize=(12, 8), xticklabels=labels_leaves, cmap="viridis_r")
                        g = sns.clustermap(gcd_df_protein, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, figsize=(15, 8), cmap="viridis_r")
                        g.ax_col_dendrogram.set_title(f"Kinase {protein} - {αC_cutoff}Å cutoff", pad=-350, fontsize='large')
                        # Remove heatmap
                        g.ax_heatmap.remove()
                        # Adjust fontsize of colorbar label
                        g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_yticklabels(), fontsize=12)

                        # Add legend - Cavity Volume
                        patchList_cavvol = []

                        for bin, label in zip(bins_labels, labels):
                            data_key = mpatches.Patch(color=label, label=bin)
                            patchList_cavvol.append(data_key)
                        cavvol_legend = plt.gca().legend(title="Volume da cavidade (Å3)", handles=patchList_cavvol, loc='lower right', bbox_to_anchor=(0.8, -4))
                        plt.gca().add_artist(cavvol_legend)

                        # Add legend - SpatialLabel
                        patchList_spatiallabel = []
                        for label in kincore_dataframe['SpatialLabel'].unique():
                            data_key = mpatches.Patch(color=sl[label], label=label)
                            patchList_spatiallabel.append(data_key)
                        spatiallabel_legend = plt.gca().legend(title="Spatial Label", handles=patchList_spatiallabel, loc='lower center', bbox_to_anchor=(0.4, -4))
                        plt.gca().add_artist(spatiallabel_legend)
                        plt.tight_layout()

                        # # Add legend - C-helix
                        # patchList_chelix = []6Gr5EUwS
                        # for label in kincore_dataframe['C-helix'].unique():
                        #     data_key = mpatches.Patch(color=ch[label], label=label)
                        #     patchList_chelix.append(data_key)
                        # chelix_legend = plt.gca().legend(title="C-helix", handles=patchList_chelix, loc='lower left', bbox_to_anchor=(0.0, -4))
                        # plt.gca().add_artist(chelix_legend)

                        # # Add legend - Rotamers
                        # patchList_dihedral = []
                        # for label in kincore_dataframe['DihedralLabel'].unique():
                        #     data_key = mpatches.Patch(color=dl[label], label=label)
                        #     patchList_dihedral.append(data_key)
                        # dihedral_legend = plt.gca().legend(title="DihedralLabel", handles=patchList_dihedral, loc='lower center', bbox_to_anchor=(0.7, -4))
                        # plt.gca().add_artist(dihedral_legend)

                        # # Add legend - LigandType
                        # patchList_ligandtype = []
                        # for label in kincore_dataframe['LigandType'].unique():
                        #     data_key = mpatches.Patch(color=lt[label], label=label)
                        #     patchList_ligandtype.append(data_key)
                        # ligandtype_legend = plt.gca().legend(title="LigandType", handles=patchList_ligandtype, loc='lower right', bbox_to_anchor=(1.0, -4))
                        # plt.gca().add_artist(ligandtype_legend)

                        if blaminus_filter == True:
                            plt.savefig(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/fixed_{protein}__Blaminus_DFGIn_Out_dendrogram_{αC_cutoff}_{linkage_type}.png", dpi=300)
                        else:
                            plt.savefig(f"{input_directory}/fixed_graph_properties_{αC_cutoff}/fixed_{protein}_DFGIn_Out_dendrogram_{αC_cutoff}_{linkage_type}.png", dpi=300)
                    else:
                        # print(f"DISCARDED {protein} has {kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGout', 0)} structures in DFG out state")
                        pass

                else:
                    # print(f"DISCARDED {protein} has less than 20 structures in kincore database")
                    pass

            print(f"END dendrogram creation for {group}")

def gcd_11_read_ligand (input_directory: str,
                 ligand_cutoff: float):

    """
    Read gcd_11 output file and transform it in a Dataframe
    """
    print("START gcd_11.txt read")

    # Read txt file
    # gcd_df = pd.read_csv(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib_ligand/aC_graph/gcd11.txt", delimiter="\t")
    gcd_df = pd.read_csv(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib/gcd11.txt", delimiter="\t")

    # Renomear colunas (mantendo a primeira coluna como 'kinase')
    column_index = gcd_df[gcd_df.columns[1:]].columns
    new_column_index = [col.split('/')[-1].split('_')[0] for col in column_index]
    new_column_index.insert(0, "kinase")
    gcd_df.columns = new_column_index

    # Renomear as linhas da coluna 'kinase' (mantendo a correspondência linha x coluna)
    gcd_df['kinase'] = [col.split('/')[-1].split('_')[0] for col in gcd_df['kinase']]

    # # Filtrar para as proteinas estudadas ABL1:
    # # Valores que você deseja filtrar
    # kinases_to_keep = ["3UE4B", "1OPLB", "2G1TB", "2HZ0A", "3CS9D"]
    #
    # # Filtrar a coluna 'kinase' para conter apenas esses valores
    # filtered_df = gcd_df[gcd_df['kinase'].isin(kinases_to_keep)]
    #
    # # Manter apenas as colunas que correspondem aos valores filtrados
    # columns_to_keep = ['kinase'] + kinases_to_keep  # 'kinase' + as colunas referentes aos valores desejados
    # filtered_df = filtered_df[columns_to_keep]
    #
    # print(filtered_df)

    print("END gcd_11.txt read")

    return gcd_df


def agglomerative_clusterization_ligand (input_directory:str,
                                  gcd_df: pd.core.frame.DataFrame,
                                  blaminus_filter: bool,
                                  family_clusterization: bool,
                                  protein_clusterization: bool,
                                  linkage_type: str,
                                  ligand_cutoff:int):

    """
    Perform agglomerative clusterization using gcd 11 distances and plot dendrogram - filter
    """

    print("START dendrogram creation for all kinases")

    import matplotlib.patches as mpatches
    from scipy.cluster.hierarchy import fcluster

    pd.set_option('display.max_rows', None)  # Exibe todas as linhas

    # Transform protein names in index values
    gcd_df = gcd_df.set_index(gcd_df.columns[0])
    gcd_df.index.name = "Kinases"

    # Add labels to proteins
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_{ligand_cutoff}_imatinib.tab", sep='\t')
    kincore_dataframe = kincore_dataframe[['Group', 'Gene', 'PDB', 'SpatialLabel', 'DihedralLabel','C-helix', 'Ligand', 'LigandType', 'Cavity_vol']]
    kincore_dataframe = kincore_dataframe.loc[kincore_dataframe['PDB'].isin(gcd_df.index)]
    kincore_dataframe = kincore_dataframe.loc[~kincore_dataframe['PDB'].duplicated()]
    kincore_dataframe = kincore_dataframe.set_index('PDB')

    # Filter of BLAminus (active) and BBAminus (inactive) structures
    kincore_dataframe_blaminus = kincore_dataframe[kincore_dataframe['DihedralLabel'].isin(['BLAminus', 'BBAminus'])]
    if blaminus_filter == True:
        gcd_df = gcd_df[gcd_df.index.isin(kincore_dataframe_blaminus.index)]
        gcd_df = gcd_df.loc[:, kincore_dataframe_blaminus.index]
        reordered_columns = gcd_df.index
        gcd_df = gcd_df[reordered_columns]

    # Create linkage matrix for dendrogram construction
    condensed_matrix = squareform(gcd_df)
    linkage_matrix = linkage(condensed_matrix, method=linkage_type)
    #
    # # Silhouete calculation
    # cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
    # try:
    #     silhouette_avg = silhouette_score(gcd_df, cluster_labels, metric="precomputed")
    #     print(f"Índice de Silhueta médio: {silhouette_avg:.4f}")
    # except ValueError as e:
    #     print(f"Erro ao calcular o índice de silhueta: {e}")

    # Identification of leaves label in a DataFrame
    # # Definindo a distância máxima para o corte (ajuste conforme necessário)
    # max_d = 6  # Distância correspondente ao terceiro nível
    # # Identificar os clusters no terceiro nível
    # clusters = fcluster(linkage_matrix, max_d, criterion='distance')
    # # Adicionar a coluna de cluster ao dataframe kincore_dataframe
    # kincore_dataframe['Cluster'] = clusters
    # # Visualizar os rótulos das folhas e seus respectivos ramos (clusters)
    # labels_with_clusters = kincore_dataframe[['Gene', 'Cluster']]  # Pode trocar 'Gene' por outra coluna de interesse
    # print(labels_with_clusters)

    # # Identificar as folhas dos três ramos principais
    # # Definir o número de clusters como 3
    # num_clusters = 3
    # clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    # # Adicionar os clusters ao DataFrame para análise
    # kincore_dataframe['Cluster'] = clusters
    # # Obter a ordem das folhas
    # leaf_order = leaves_list(linkage_matrix)
    # # Mapear a ordem para os índices do dataframe
    # ordered_leaves = gcd_df.index[leaf_order]
    # # Criar um mapeamento da ordem para cada índice
    # leaf_order_mapping = {label: i for i, label in enumerate(ordered_leaves)}
    # # Adicionar a ordem ao dataframe
    # kincore_dataframe['Order'] = kincore_dataframe.index.map(leaf_order_mapping)
    # # Adicionar a informação de volume da cavidade
    # kincore_dataframe_vol = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_5_imatinib.tab", sep="\t")
    # kincore_dataframe_unified = pd.merge(kincore_dataframe, kincore_dataframe_vol, on=["Gene", "PDB"], how="inner")
    # kincore_dataframe_unified.to_csv(f"{input_directory}/kinases_clustered_imatinib_5.tab", index=True)
    # print(kincore_dataframe_unified)

    # Create color labels for range values of Cavity Volume
    bins_labels = ['0-250', '251-500', '501-750', '751-1000', '>1001']
    bins = [0, 250, 500, 750, 1000, float('inf')]
    labels = ['#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000']
    row_colors_cv = pd.cut(kincore_dataframe['Cavity_vol'], bins=bins, labels=labels, right=False)

    # Create color labels for unique values of SpatialLabel
    sl = dict(zip(kincore_dataframe['SpatialLabel'].unique(), ["#00FFFF", "#008000", "#DDA0DD", "#FF0000"]))
    row_colors_sl = kincore_dataframe['SpatialLabel'].map(sl)

    # Create color labels for unique values of Group
    gr = dict(zip(kincore_dataframe['Group'].unique(), ["#069AF3", "#380282", "#FF796C", "#A9561E", '#C79FEF', '#000000', '#0000FF', '#C5C9C7', '#A52A2A']))
    row_colors_gr = kincore_dataframe['Group'].map(gr)

    # Join dataframes
    joined_colors = pd.concat([row_colors_cv, row_colors_sl, row_colors_gr], axis=1)

    # Plot dendrogram using color bars to identify labels (without heatmap)
    sns.set_theme(color_codes=True)
    g = sns.clustermap(gcd_df, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, dendrogram_ratio=(0.05, 0.35), figsize=(12, 8), cmap="viridis_r")
    g.ax_col_dendrogram.set_title(f"Quinases - {ligand_cutoff}Å cutoff", pad=-350, fontsize='large')
    # Remove heatmap
    # g.ax_heatmap.remove()
    # Adjust fontsize of colorbar label
    g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_yticklabels(), fontsize=12)

    # Add legend - Cavity Volume
    patchList_cavvol = []
    for bin, label in zip(bins_labels, labels):
        data_key = mpatches.Patch(color=label, label=bin)
        patchList_cavvol.append(data_key)
    cavvol_legend = plt.gca().legend(title="Volume da cavidade (Å3)", handles=patchList_cavvol, loc='lower right', bbox_to_anchor=(1.0, -4))
    plt.gca().add_artist(cavvol_legend)

    # Add legend - SpatialLabel
    patchList_spatiallabel = []
    for label in kincore_dataframe['SpatialLabel'].unique():
        data_key = mpatches.Patch(color=sl[label], label=label)
        patchList_spatiallabel.append(data_key)
    spatiallabel_legend = plt.gca().legend(title="Spatial Label", handles=patchList_spatiallabel, loc='lower center', bbox_to_anchor=(0.5, -4))
    plt.gca().add_artist(spatiallabel_legend)

    # Add legend - Group
    patchList_group = []
    for label in kincore_dataframe['Group'].unique():
        data_key = mpatches.Patch(color=gr[label], label=label)
        patchList_group.append(data_key)
    plt.gca().legend(title="Grupo", handles=patchList_group, loc='lower left', bbox_to_anchor=(0, -4))

    if blaminus_filter == True:
        plt.savefig(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib_ligand/fixed_DFGIn_Out_blaminus_dendrogram_{ligand_cutoff}_{linkage_type}.png", dpi=300)

    else:
        plt.savefig(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib_ligand/small_heatmap_fixed_DFGIn_Out_dendrogram_{ligand_cutoff}_{linkage_type}.png", dpi=300)

    print("END dendrogram creation for all kinases")

    # Generate one dendrogram per family
    if family_clusterization == True:
        for family in kincore_dataframe['Group'].unique():
            print(f"START dendrogram creation for {family}")

            if blaminus_filter == True:
                # Analyze only blaminus and bbaminus structures
                kincore_dataframe_family = kincore_dataframe_blaminus[kincore_dataframe_blaminus['Group'] == family]
            else:
                kincore_dataframe_family = kincore_dataframe[kincore_dataframe['Group'] == family]

            gcd_df_family = gcd_df[gcd_df.index.isin(kincore_dataframe_family.index)]
            gcd_df_family = gcd_df_family.loc[:, kincore_dataframe_family.index]
            reordered_columns = gcd_df_family.index
            gcd_df_family = gcd_df_family[reordered_columns]

            if gcd_df_family.empty:
                pass
            else:
                condensed_matrix = squareform(gcd_df_family)
                linkage_matrix = linkage(condensed_matrix, method=linkage_type)

                # Join dataframes
                joined_colors = pd.concat([row_colors_sl, row_colors_cv], axis=1)

                # Plot dendrogram using color bars to identify labels (without heatmap)
                sns.set_theme(color_codes=True)
                g = sns.clustermap(gcd_df_family, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, dendrogram_ratio=(0.05, 0.35), figsize=(12, 9), cmap="viridis_r")
                # Remove heatmap
                # g.ax_heatmap.remove()
                g.ax_col_dendrogram.set_title(f"{family} Kinases - {ligand_cutoff}Å cutoff", pad=-350, fontsize='large')
                # Adjust fontsize of colorbar label
                g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_yticklabels(), fontsize=12)

                # Add legend - SpatialLabel
                patchList_spatiallabel = []
                for label in kincore_dataframe['SpatialLabel'].unique():
                    data_key = mpatches.Patch(color=sl[label], label=label)
                    patchList_spatiallabel.append(data_key)
                spatiallabel_legend = plt.gca().legend(title="Spatial Label", handles=patchList_spatiallabel, loc='lower center', bbox_to_anchor=(0.4, -4))
                plt.gca().add_artist(spatiallabel_legend)

                # Add legend - Cavity Volume
                patchList_cavvol = []
                for bin, label in zip(bins_labels, labels):
                    data_key = mpatches.Patch(color=label, label=bin)
                    patchList_cavvol.append(data_key)
                cavvol_legend = plt.gca().legend(title="Volume da cavidade (Å3)", handles=patchList_cavvol, loc='lower right', bbox_to_anchor=(0.8, -4))
                plt.gca().add_artist(cavvol_legend)

                if blaminus_filter == True:
                    plt.savefig(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib_ligand/fixed_DFGIn_Out__blaminus_dendrogram_{family}_{ligand_cutoff}_{linkage_type}.png", dpi=300)
                else:
                    plt.savefig(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib_ligand/heatmap_fixed_DFGIn_Out_dendrogram_{family}_{ligand_cutoff}_{linkage_type}.png", dpi=300)

                print(f"END dendrogram creation for {family}")

    # Plot dendrogram for specific proteins
    if protein_clusterization == True:
        for group in kincore_dataframe['Group'].unique():
            print(f"START dendrogram creation for {group}")
            # Filter kincore dataframe for groups
            if blaminus_filter == True:
                kincore_dataframe_family = kincore_dataframe_blaminus[kincore_dataframe_blaminus['Group'] == group]
            else:
                kincore_dataframe_family = kincore_dataframe[kincore_dataframe['Group'] == group]
            count_proteins = kincore_dataframe_family['Gene'].value_counts()

            # Filter kincore dataframe for proteins
            for protein in count_proteins.index:
                if count_proteins[protein] > 20:
                    kincore_dataframe_protein = kincore_dataframe_family[kincore_dataframe_family['Gene'] == protein]
                    if kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGout', 0) > 3:
                        print(f"{protein} has {kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGout', 0)} structures in DFG out state")
                        print(f"{protein} has {kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGin', 0)} structures in DFG in state")

                        # Filter and ordenate gcd dataframe per protein
                        gcd_df_protein = gcd_df[gcd_df.index.isin(kincore_dataframe_protein.index)]
                        gcd_df_protein = gcd_df_protein.loc[:, kincore_dataframe_protein.index]
                        reordered_columns = gcd_df_protein.index
                        gcd_df_protein = gcd_df_protein[reordered_columns]

                        # Create linkage matrix for dendrogram construction
                        condensed_matrix = squareform(gcd_df_protein)
                        linkage_matrix = linkage(condensed_matrix, method=linkage_type)

                        # Create list of labels for all leaves
                        labels_leaves = list(gcd_df_protein.index)

                        # Join dataframes
                        joined_colors = pd.concat([row_colors_sl, row_colors_cv], axis=1)

                        # Plot dendrogram using color bars to identify labels (without heatmap)
                        sns.set_theme(color_codes=True)
                        sns.set(font_scale=0.7)
                        # g = sns.clustermap(gcd_df_protein, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, dendrogram_ratio=(0.05, 0.35), figsize=(12, 8), xticklabels=labels_leaves, cmap="viridis_r")
                        g = sns.clustermap(gcd_df_protein, row_linkage=linkage_matrix, col_linkage=linkage_matrix, col_colors=joined_colors, cbar_pos=None, row_cluster=False, col_cluster=True, figsize=(15, 8), cmap="viridis_r")
                        g.ax_col_dendrogram.set_title(f"Kinase {protein} - {ligand_cutoff}Å cutoff", pad=-350, fontsize='large')
                        # Remove heatmap
                        # g.ax_heatmap.remove()
                        # Adjust fontsize of colorbar label
                        g.ax_col_colors.set_yticklabels(g.ax_col_colors.get_yticklabels(), fontsize=12)

                        # Add legend - Cavity Volume
                        patchList_cavvol = []

                        for bin, label in zip(bins_labels, labels):
                            data_key = mpatches.Patch(color=label, label=bin)
                            patchList_cavvol.append(data_key)
                        cavvol_legend = plt.gca().legend(title="Volume da cavidade (Å3)", handles=patchList_cavvol, loc='lower right', bbox_to_anchor=(0.8, -4))
                        plt.gca().add_artist(cavvol_legend)

                        # Add legend - SpatialLabel
                        patchList_spatiallabel = []
                        for label in kincore_dataframe['SpatialLabel'].unique():
                            data_key = mpatches.Patch(color=sl[label], label=label)
                            patchList_spatiallabel.append(data_key)
                        spatiallabel_legend = plt.gca().legend(title="Spatial Label", handles=patchList_spatiallabel, loc='lower center', bbox_to_anchor=(0.4, -4))
                        plt.gca().add_artist(spatiallabel_legend)
                        plt.tight_layout()

                        if blaminus_filter == True:
                            plt.savefig(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}/fixed_{protein}__Blaminus_DFGIn_Out_dendrogram_{ligand_cutoff}_{linkage_type}.png", dpi=300)
                        else:
                            plt.savefig(f"{input_directory}/ligand_fixed_graph_properties_{ligand_cutoff}_imatinib_ligand/heatmap_fixed_{protein}_DFGIn_Out_dendrogram_{ligand_cutoff}_{linkage_type}.png", dpi=300)
                    else:
                        # print(f"DISCARDED {protein} has {kincore_dataframe_protein['SpatialLabel'].value_counts().get('DFGout', 0)} structures in DFG out state")
                        pass

                else:
                    # print(f"DISCARDED {protein} has less than 20 structures in kincore database")
                    pass

            print(f"END dendrogram creation for {group}")

def merge_dataframes (input_directory = "input/kinases/kincore_database"):

    # Adicionar a informação de volume da cavidade
    kincore_dataframe = pd.read_csv("input/kinases/kincore_database/ligand_fixed_graph_properties_5_imatinib/kinases_clustered_imatinib_5.tab", sep=",")
    print(kincore_dataframe)
    kincore_dataframe_vol = pd.read_csv("input/kinases/kincore_database/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_5_imatinib.tab", sep=",")
    print(kincore_dataframe_vol)
    kincore_dataframe_unified = pd.merge(kincore_dataframe, kincore_dataframe_vol, on=["PDB"], how="inner")
    columns_to_keep = ["Gene", "PDB", "Cluster", "Volume"]  # Customize conforme necessário
    kincore_dataframe_unified = kincore_dataframe_unified.loc[:, ~kincore_dataframe_unified.columns.duplicated()]
    print(kincore_dataframe_unified)
    kincore_dataframe_unified.to_csv("input/kinases/kincore_database/kinases_clustered_imatinib_5.tab", index=True)

def pykvfinder_run (input_directory: str):

    # kinases_list = ["7KXZA", "5HIBA", "6G33C", "4F0IA", "7AW4A", "6HEXA", "4Z9LA", "4U79A", "4PDOB", "4F6SA", "8GDSD", "4F6UA", "3FSFA", "3EFKB", "2W05A", "7MGJA", "4R5YA", "6GQKB", "8H6PA", "5JSMB", "1PXNA", "3O96A", "6YZ4A", "4ITJA", "4YR8C", "1UV5A", "4F7NA", "6QE1A", "5UQ1A", "4EOSA", "3MTLA", "6YLCA", "6GESA", "2C68A", "6CNHA", "3HV4B", "6NYBA", "6V2UA", "3Q96A", "5HI2A", "4ZTNA", "3G2FB", "4ITIA", "7YDXB", "5K7GD", "2OIBA", "6C3EB", "4TPTA", "6E6EA", "1KSWA", "2RF9A", "6D1YA", "7LTXC", "2RFEC", "6HEWA", "6NPVA", "4J99A", "4Z16A", "6GQJA", "8AN8A"]
    # kinases_list = ["7KSKA", "4B99A", "3RK5A", "IZYJA", "3MPAA", "8CURA", "5G6VB", "3MH3A", "3PG3A", "3KQ7A", "6QE1A", "2BAJA"]
    # kinases_list = ["6OBFA", "6VNCB", "3D7TA", "4ZOGB", "4GFMA", "4QQJA", "6BLNA", "5C01B", "7APGD", "4K33A", "5D7VC", "6XE4A", "3UE4A", "7AW4B", "5FXRA", "3EFKB", "6TFWB", "3F5PK", "5BVOA", "3QRIA", "6PL1A"]
    kinases_list = ["6IG8A"]
    import pyKVFinder

    # Set vertices for detection
    vertices_load = toml.load(f"{input_directory}/kinases_vertices_2.toml")
    vertices = vertices_load["box"]

    for kinase in kinases_list:
        file = f"{kinase[:4]}_{kinase[4]}.pdb_fixed.pdb"
        print(f"Processing: {file}")
        kinase_output_dir = f"{input_directory}/fixed_KV_Files_300_ligand/{kinase}/imatinib_ligand"

        # Criar diretório de saída
        try:
            os.makedirs(kinase_output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {kinase_output_dir}: {e}")
            continue

        try:
            # Ler o ligante e a proteína
            latomic = pyKVFinder.read_pdb(f"{input_directory}/ligante_2hyy.pdb")
            atomic = pyKVFinder.read_pdb(f"{input_directory}/aligned_PDB_fixed/{file}")

            # Detectar cavidades
            ncav, cavities = pyKVFinder.detect(atomic, vertices, probe_out=6.0, latomic=latomic,volume_cutoff=300, ligand_cutoff=5)

            # Identificar resíduos da interface e volumes
            residues = pyKVFinder.constitutional(cavities, atomic, vertices, ignore_backbone=False)
            surface, volume, area = pyKVFinder.spatial(cavities)

            # Exportar resultados
            output_pdb = f"{kinase_output_dir}/{kinase}_5_output.pdb"
            pyKVFinder.export(output_pdb, cavities, None, vertices)
            pyKVFinder.write_results(
                f"{kinase_output_dir}/{kinase}_5_output.toml",
                input=f"{input_directory}/aligned_PDB_fixed/{file}",
                ligand=f"{input_directory}/ligante_2hyy.pdb",
                output=output_pdb,
                residues=residues, volume=volume
            )
            print(f"Cavities identified for {kinase}")

        except Exception as e:
            print(f"Error processing {kinase}: {e}")

def agglomerative_clusterization_order (input_directory:str,
                                  gcd_df: pd.core.frame.DataFrame,
                                  blaminus_filter: bool,
                                  family_clusterization: bool,
                                  protein_clusterization: bool,
                                  linkage_type: str,
                                  ligand_cutoff:int):

    """
    Perform agglomerative clusterization using gcd 11 distances and plot dendrogram - filter
    """

    print("START dendrogram creation for all kinases")

    import matplotlib.patches as mpatches
    from scipy.cluster.hierarchy import fcluster

    pd.set_option('display.max_rows', None)  # Exibe todas as linhas

    # Transform protein names in index values
    gcd_df = gcd_df.set_index(gcd_df.columns[0])
    gcd_df.index.name = "Kinases"

    # kinases_list = ["7KXZA", "4L44A", "4OGRE", "4F0IA", "1OPLB", "6NPTA", "8H6PA", "5JSMB", "1PXNA", "2HZIB", "6YZ4A", "7TNHA", "2B55A", "7M0YB", "3VVHC", "7MGJA", "4R5YA", "6EIMB", "4PDOB", "4U79A", "7O7KA", "8GDSD", "4F6UA"]
    # filtered_distances = gcd_df.loc[kinases_list, kinases_list]
    #
    # # Mostrar o resultado
    # print(filtered_distances)

    # Add labels to proteins
    kincore_dataframe = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_{ligand_cutoff}_imatinib.tab", sep=',')
    kincore_dataframe = kincore_dataframe[['Group', 'Gene', 'PDB', 'SpatialLabel', 'DihedralLabel','C-helix', 'Ligand', 'LigandType', 'Cavity_vol']]
    kincore_dataframe = kincore_dataframe.loc[kincore_dataframe['PDB'].isin(gcd_df.index)]
    kincore_dataframe = kincore_dataframe.loc[~kincore_dataframe['PDB'].duplicated()]
    kincore_dataframe = kincore_dataframe.set_index('PDB')

    # # Create linkage matrix for dendrogram construction
    # condensed_matrix = squareform(gcd_df)
    # linkage_matrix = linkage(condensed_matrix, method=linkage_type)
    #
    # # Identificar as folhas dos três ramos principais
    # # Definir o número de clusters como 3
    # num_clusters = 3
    # clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    # # Adicionar os clusters ao DataFrame para análise
    # kincore_dataframe['Cluster'] = clusters
    # # Obter a ordem das folhas
    # leaf_order = leaves_list(linkage_matrix)
    # # Mapear a ordem para os índices do dataframe
    # ordered_leaves = gcd_df.index[leaf_order]
    # # Criar um mapeamento da ordem para cada índice
    # leaf_order_mapping = {label: i for i, label in enumerate(ordered_leaves)}
    # # Adicionar a ordem ao dataframe
    # kincore_dataframe['Order'] = kincore_dataframe.index.map(leaf_order_mapping)
    # # Adicionar a informação de volume da cavidade
    # kincore_dataframe_vol = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_5_imatinib.tab", sep="\t")
    # kincore_dataframe_unified = pd.merge(kincore_dataframe, kincore_dataframe_vol, on=["Gene", "PDB"], how="inner")
    # # kincore_dataframe_unified.to_csv(f"{input_directory}/kinases_clustered_imatinib_5.tab", index=True)
    # # print(kincore_dataframe_unified)

    print("END dendrogram creation for all kinases")

    # Generate one dendrogram per family
    if family_clusterization == True:
        for family in kincore_dataframe['Group'].unique():
            print(f"START dendrogram creation for {family}")

            if blaminus_filter == True:
                # Analyze only blaminus and bbaminus structures
                kincore_dataframe_family = kincore_dataframe_blaminus[kincore_dataframe_blaminus['Group'] == family]
            else:
                kincore_dataframe_family = kincore_dataframe[kincore_dataframe['Group'] == family]

            gcd_df_family = gcd_df[gcd_df.index.isin(kincore_dataframe_family.index)]
            gcd_df_family = gcd_df_family.loc[:, kincore_dataframe_family.index]
            reordered_columns = gcd_df_family.index
            gcd_df_family = gcd_df_family[reordered_columns]

            if gcd_df_family.empty:
                pass
            else:
                condensed_matrix = squareform(gcd_df_family)
                linkage_matrix = linkage(condensed_matrix, method=linkage_type)

                # Identificar as folhas dos três ramos principais
                # Definir o número de clusters como 3
                # num_clusters = 3
                # clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
                # Adicionar os clusters ao DataFrame para análise
                # kincore_dataframe_family['Cluster'] = clusters
                # Obter a ordem das folhas
                leaf_order = leaves_list(linkage_matrix)
                # Mapear a ordem para os índices do dataframe
                ordered_leaves = gcd_df_family.index[leaf_order]
                # Criar um mapeamento da ordem para cada índice
                leaf_order_mapping = {label: i for i, label in enumerate(ordered_leaves)}
                # Adicionar a ordem ao dataframe
                kincore_dataframe_family['Order'] = kincore_dataframe_family.index.map(leaf_order_mapping)
                # Adicionar a informação de volume da cavidade
                kincore_dataframe_vol = pd.read_csv(f"{input_directory}/Human_Allgroups_Allspatials_Alldihedrals_All_CAVVOL_5_imatinib.tab", sep="\t")
                # kincore_dataframe_unified = pd.merge(kincore_dataframe_family, kincore_dataframe_vol, on=["Gene", "PDB"], how="inner")
                kincore_dataframe_family.to_csv(f"{input_directory}/kinases_clustered_imatinib_5_{family}.tab", index=True)
                print(kincore_dataframe_family)

                print(f"END dendrogram creation for {family}")

if "__main__"== __name__:
    input_directory = "/home/up.msimoes1/kincore_database"
    local_directory = "input/kinases/kincore_database"
    test_gcd11= "input/kinases/kincore_database"
    test_cluster = "/home/up.msimoes1/kincore_database/kinases_kincore_chainA_teste_pyKVFinder"
    test = "input/kinases/kinases-kincore-chainA-teste-pyKVFinder"
    # kincore_pdb = pdb_download(input_directory=input_directory)
    # pdb_fix = fix_pdb(input_directory=input_directory)
    # pdb_split = split_chains(input_directory=local_directory)
    # alignment = structure_alignment (input_directory=local_directory, reference_structure="3qri")
    # test_pykvfinder = pykvfinder_test(input_directory=local_directory)
    # run_pykvfinder_ligand = pykvfinder_ligand(input_directory=input_directory, volume_cutoff=300)
    # run_pyKVFinder = pyKVFinder (input_directory=test, reference_structure="3qri", volume_cutoff=300, detection_residues=["D", "F"])
    # run_occurrence = occurrence_protein (input_directory=local_directory, reference_structure="2hyy", volume_cutoff=300)
    # pyKVFinder_analysis = cavities_analysis (input_directory=input_directory, volume_cutoff_list=[100,150,200,250,300,350,400], detection_residues=["D", "F"])
    # pyKVFinder_analysis_ligand = cavities_analysis_ligand (input_directory=input_directory, radius_list=[5, 6, 7, 8, 9, 10])
    # pymol = cavities_pymol (input_directory=local_directory, reference_structure="3qri", volume_cutoff_list=[300], detection_residues="D")
    # pymol_ligand = cavities_pymol_ligand (input_directory=local_directory, reference_structure="3qri", ligand_cutoff_list=[5, 6, 7, 8, 9, 10], DFG_list=["in", "out"])
    # αC_coords = αC_extract_coords(input_directory=test, volume_cutoff=300, detection_residues="D", verbose="False")
    # αC_coords_ligand = αC_extract_coords_ligand(input_directory=input_directory, volume_cutoff=300, ligand_cutoff=10, verbose="True")
    # αC_distance = αC_calc_distance(kinases_coords=αC_coords_ligand, verbose="True")
    # graph = create_graphs(input_directory=test, kinases_coords=αC_distance, αC_cutoff=5, volume_cutoff=300, detection_residues="D", base_name="5A", verbose="True")
    # graph_ligand = create_graphs_ligand(input_directory=input_directory, kinases_coords=αC_distance, αC_cutoff=10, volume_cutoff=300, ligand_cutoff=10, verbose="True")
    # add_volume_kincore = add_volume_kincore_database(input_directory=input_directory, ligand_cutoff_list=[5, 6, 7, 8, 9, 10])
    # read_gcd_file = gcd_11_read(input_directory=local_directory, αC_cutoff=5)
    # clusterization_agglomerative = agglomerative_clusterization(input_directory=input_directory, gcd_df=read_gcd_file, αC_cutoff=5, blaminus_filter=False, family_clusterization=True, protein_clusterization=True, linkage_type="ward")
    read_gcd_file_ligand = gcd_11_read_ligand(input_directory=local_directory, ligand_cutoff=5)
    # clusterization_agglomerative_ligand = agglomerative_clusterization_ligand(input_directory=input_directory, gcd_df=read_gcd_file_ligand, ligand_cutoff=5, blaminus_filter=False, family_clusterization=True, protein_clusterization=True, linkage_type="ward")
    # dataframe_merge = merge_dataframes()
    # pykvfinder = pykvfinder_run(input_directory=local_directory)
    clusterization_agglomerative_ordering = agglomerative_clusterization_order(input_directory=local_directory, gcd_df=read_gcd_file_ligand, ligand_cutoff=5, blaminus_filter=False, family_clusterization=True, protein_clusterization=True, linkage_type="ward")
