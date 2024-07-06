# -*- coding: utf-8 -*-
# "C:\\Python312\\python" "process_md5.py" --init Init_MDS_Euc.pickle   --mapping Mapping_REFINED.pickle  --evolution REFINED_Evolve.csv --num 5

import pickle
import numpy as np
from itertools import product
import time
import datetime
import argparse
from scipy.stats import pearsonr
from scipy.spatial import distance
import pandas as pd

# Definir funciones
def universial_corr(dist_matr, mapping_in_int):
    Nn = dist_matr.shape[0]
    coord = np.array([[item[0] for item in np.where(mapping_in_int == ii)] for ii in range(Nn)])
    pixel_dist = distance.pdist(coord)
    pixel_dist = pixel_dist.reshape(len(pixel_dist), 1)
    feature_dist = distance.squareform(dist_matr)
    feature_dist = feature_dist.reshape(len(feature_dist), 1)
    L2_Norm = np.sqrt(sum((pixel_dist - feature_dist) ** 2) / sum(feature_dist ** 2))
    return L2_Norm

def evaluate_swap(coord1, coord2, dist_matr, mapping_in_int, original_corr=-2):
    the_map = mapping_in_int.copy()
    if coord1[0] < 0 or coord1[1] < 0 or coord2[0] < 0 or coord2[1] < 0:
        return np.nan
    if coord1[0] >= the_map.shape[0] or coord1[1] >= the_map.shape[0] or coord2[0] >= the_map.shape[0] or coord2[1] >= the_map.shape[0]:
        return np.nan
    if original_corr < -1 or original_corr > 1:
        original_corr = universial_corr(dist_matr, the_map)
    try:
        temp = the_map[coord1[0], coord1[1]]
        the_map[coord1[0], coord1[1]] = the_map[coord2[0], coord2[1]]
        the_map[coord2[0], coord2[1]] = temp
        changed_corr = universial_corr(dist_matr, the_map)
        return changed_corr - original_corr
    except IndexError:
        raise Warning("Swap index:", coord1, coord2, "Index error. Check the coordination.")
        return np.nan

def evaluate_centroid(centroid, dist_matr, mapping_in_int):
    original_corr = universial_corr(dist_matr, mapping_in_int)
    results = [100000]  # just to skip the 0 position
    for each_direc in product([-1, 0, 1], repeat=2):
        swap_coord = [centroid[0] + each_direc[0], centroid[1] + each_direc[1]]
        try:
            evaluation = evaluate_swap(centroid, swap_coord, dist_matr, mapping_in_int, original_corr)
            if not isinstance(evaluation, (int, float)):
                raise ValueError("Evaluation must be a scalar value")
        except ValueError:
            evaluation = 0
        results.append(evaluation)
    results_array = np.array(results)
    best_swap_direc = np.where(results_array == np.nanmin(results_array))[0][0]
    return best_swap_direc

def evaluate_centroids_in_list(centroids_list, dist_matr, mapping_in_int):
    results = dict()
    for each_centr in centroids_list:
        each_centr = tuple(each_centr)
        evaluation = evaluate_centroid(each_centr, dist_matr, mapping_in_int)
        results.update({each_centr: evaluation})
    return results

def execute_coordination_swap(coord1, coord2, mapping_in_int):
    the_map = mapping_in_int
    if coord1[0] < 0 or coord1[1] < 0 or coord2[0] < 0 or coord2[1] < 0:
        raise Warning("Swapping failed:", coord1, coord2, "-- Negative coordination.")
        return the_map
    if coord1[0] > the_map.shape[0] or coord1[1] > the_map.shape[0] or coord2[0] > the_map.shape[0] or coord2[1] > the_map.shape[0]:
        raise Warning("Swapping failed:", coord1, coord2, "-- Coordination out of bound.")
        return the_map

    temp = the_map[coord1[0], coord1[1]]
    the_map[coord1[0], coord1[1]] = the_map[coord2[0], coord2[1]]
    the_map[coord2[0], coord2[1]] = temp
    return the_map

def execute_direction_swap(centroid, mapping_in_int, direction=5):
    coord1 = list(centroid)
    coord2 = list(centroid)
    if direction not in range(1, 10):
        raise ValueError("Invalid swapping direction.")
    if direction == 5:
        return mapping_in_int

    if direction in [1, 4, 7]:
        coord2[1] -= 1
    elif direction in [3, 6, 9]:
        coord2[1] += 1

    if direction in [1, 2, 3]:
        coord2[0] -= 1
    elif direction in [7, 8, 9]:
        coord2[0] += 1

    the_map = execute_coordination_swap(coord1, coord2, mapping_in_int)
    return the_map

def execute_dict_swap(swapping_dict, mapping_in_int):
    for each_key in swapping_dict:
        execute_direction_swap(each_key, mapping_in_int, direction=swapping_dict[each_key])
    return mapping_in_int

# Lectura de parámetros desde la terminal
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--init', type=str, required=True, help="Initial data file")
parser.add_argument('--num', type=int, required=True, help="Number of iterations")
parser.add_argument('--mapping', type=str, required=True, help="Output mapping file")
parser.add_argument('--evolution', type=str, required=True, help="Output evolution file")
args = parser.parse_args()

# Carga de datos
print("Loading data...")
print("Time now at the beginning: ", datetime.datetime.now())
with open(args.init, 'rb') as file:
    gene_names, dist_matr, init_map = pickle.load(file)

Nn = len(gene_names)
kk = args.num

print("Data loaded.")
print("Actual time: ", datetime.datetime.now())
if init_map.shape[0] != init_map.shape[1]:
    raise ValueError("For now only square images are considered.")
nn = init_map.shape[0]
init_map = np.char.strip(init_map.astype(str), 'F').astype(int)
map_in_int = init_map

corr_evol = []
print("Initial corr: >>>", universial_corr(dist_matr, map_in_int))
print("Start hill climbing...", datetime.datetime.now())
print("Number of iterations:", kk)
print("Number of features:", Nn)
for n_iter in range(kk):
    init_coords = [x for x in product([0, 1, 2], repeat=2)] # 9 initial coordinates.
    for init_coord in init_coords: # Update the mapping.
        centr_list = [x for x in product([init_coord[0] + i * 3 for i in range(int(nn / 3) + 1) if (init_coord[0] + i * 3) < nn], [init_coord[1] + i * 3 for i in range(int(nn / 3) + 1) if (init_coord[1] + i * 3) < nn])]
        swap_dict = evaluate_centroids_in_list(centr_list, dist_matr, map_in_int)
        map_in_int = execute_dict_swap(swap_dict, map_in_int)
        print(">", init_coord, "Corr:", universial_corr(dist_matr, map_in_int))
        print("Time now: ", datetime.datetime.now())

    print(">>>", n_iter, "Corr:", universial_corr(dist_matr, map_in_int))
    corr_evol.append(universial_corr(dist_matr, map_in_int))

coords = np.array([[item[0] for item in np.where(map_in_int == ii)] for ii in range(Nn)])
with open(args.mapping, 'wb') as file:
    pickle.dump([gene_names, coords, map_in_int], file)
Endtime = datetime.datetime.now()
print("Time now at the end: ", Endtime)
pd.Series(corr_evol).to_csv(args.evolution)
