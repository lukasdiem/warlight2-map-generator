from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph
from matplotlib import cm
from more_itertools.recipes import unique_everseen
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union


def randint_tuple(in_tuple):
    return np.random.randint(in_tuple[0], in_tuple[1])


def voronoi_non_border_region(voronoi_graph: Voronoi):
    nr_regions = voronoi_graph.points.shape[0]
    non_border_region = np.zeros((nr_regions, 1), dtype=np.bool)
    for idx in range(nr_regions):
        region = voronoi_graph.regions[voronoi_graph.point_region[idx]]
        if -1 not in region:
            non_border_region[idx] = True

    return non_border_region


def find_seed_regions(voronoi_graph: 'VoronoiGraph', map_dict):
    # non_border = voronoi_non_border_region(voronoi_graph)
    non_border = np.zeros((voronoi_graph.vor.points.shape[0], 1), dtype=np.bool)
    non_border[voronoi_graph.filtered_pts_idx] = True
    non_blocked = map_dict['country_idx'] == 0
    possible_idx = np.flatnonzero(np.logical_and(non_blocked, non_border))

    return possible_idx[np.random.randint(len(possible_idx))]


def init_map_dict(voronoi_graph):
    nr_fields = voronoi_graph.points.shape[0]
    map_dict = {
        'country_idx': np.zeros((nr_fields, 1), dtype=np.uint16),
        'continent_idx': np.zeros((nr_fields, 1), dtype=np.uint16),
        'neighbors': voronoi_neighbors(voronoi_graph)
    }

    return map_dict


def find_neighbors(tri, pt_idx):
    return tri.vertex_neighbor_vertices[1][
           tri.vertex_neighbor_vertices[0][pt_idx]:tri.vertex_neighbor_vertices[0][pt_idx + 1]]


def voronoi_neighbors(voronoi_graph: Voronoi):
    tri = Delaunay(voronoi_graph.points)
    nr_pts = voronoi_graph.points.shape[0]
    neighbors = [find_neighbors(tri, idx) for idx in range(nr_pts)]

    return neighbors


def create_land(map_dict, voronoi_graph, start_idx, land_size, continent_idx, country_idx, max_iter=10000):
    remaining_size = land_size
    neighbors = map_dict['neighbors']
    queue = [start_idx]

    iter = 0
    # not_blocked = voronoi_non_border_region(voronoi_graph)
    not_blocked = np.zeros((voronoi_graph.vor.points.shape[0], 1), dtype=np.bool)
    not_blocked[voronoi_graph.filtered_pts_idx] = True
    not_blocked[map_dict['continent_idx'] > 0] = False

    while remaining_size > 0 and len(queue) > 0 and iter < max_iter:
        reg_idx = queue.pop(0)

        # append if not already done
        if map_dict['continent_idx'][reg_idx] == 0 and not_blocked[reg_idx]:
            not_blocked[reg_idx] = False
            neighbor_idx = neighbors[reg_idx]
            neighbor_idx = neighbor_idx[not_blocked[neighbor_idx].reshape(-1)]

            # append all region neighbors
            queue.extend(neighbor_idx)
            queue = list(unique_everseen(queue))

            map_dict['continent_idx'][reg_idx] = continent_idx + 1
            map_dict['country_idx'][reg_idx] = country_idx + 1
            remaining_size -= 1

        iter += 1

    return queue


def create_map(in_graph: 'VoronoiGraph', continent_range: tuple, country_range: tuple,
               country_size_range: tuple):
    nr_continents = randint_tuple(continent_range)

    print('#Continents: {}'.format(nr_continents))
    voronoi_graph = in_graph.vor

    map_dict = init_map_dict(voronoi_graph)
    for c_idx in range(nr_continents):
        seed_idx = find_seed_regions(in_graph, map_dict)
        nr_countries = randint_tuple(country_range)

        for country_idx in range(nr_countries):
            country_size = randint_tuple(country_size_range)

            print('Continent: {}, Country: {}/{}, Country Size: {}'.format(c_idx, country_idx,
                                                                           nr_countries, country_size))

            neighbor_idx = create_land(map_dict, in_graph, seed_idx, country_size,
                                       continent_idx=c_idx, country_idx=country_idx)

            # update the seed index to a neighbor
            if len(neighbor_idx) > 0:
                seed_idx = neighbor_idx.pop()  # [np.random.randint(len(neighbor_idx))]
            else:
                print('break')
                pass

    return map_dict


def polygons_from_map(vor, map_dict):
    """
    Get the polygons from the map_dict and voronoi graph

    :param vor: the voronoi graph
    :param map_dict: the map dict containing `country_idx` and `continent_idx`
    :return: (tuple)
        poly_continents (list): The continent polygons represented as list of Nx2 np.ndarray's.
        poly_countries (list): The country polygons represented as list of Nx2 np.ndarray's.
    """
    poly_countries = []
    poly_continents = []

    # map polygons to countries and continents
    region_map = dict(Regions=[], SuperRegions=[])
    polys = defaultdict(lambda: defaultdict(list))

    for idx, pt in enumerate(vor.points):
        region = vor.regions[vor.point_region[idx]]
        continent_idx = map_dict['continent_idx'][idx][0]
        country_idx = map_dict['country_idx'][idx][0]

        if continent_idx > 0 and -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            polys[continent_idx][country_idx].append(polygon)

    # map (continent_idx, country_idx) -> region_id
    cc_idx = np.column_stack((map_dict['continent_idx'], map_dict['country_idx']))
    cc_idx_unique = np.unique(cc_idx, axis=0)
    # remove 0 indices => sea
    cc_idx_unique = np.delete(cc_idx_unique, np.where(np.any(cc_idx_unique == 0, axis=1)), axis=0)
    map_dict['continent_country_tup'] = cc_idx_unique
    region_ids = np.zeros((cc_idx.shape[0], 1), dtype=np.int32)
    for r_id, idx in enumerate(cc_idx_unique):
        region_ids[np.all(cc_idx == idx, axis=1)] = r_id + 1
    map_dict['region_id'] = region_ids

    # correctly map the neighbor indices
    nr_continents = np.max(map_dict['continent_idx'])
    nr_countries = np.max(region_ids)
    continent_neighbors = []
    country_neighbors = [set() for _ in range(nr_countries)]
    for continent_idx in range(1, nr_continents + 1):
        reg_indices = np.flatnonzero(map_dict['continent_idx'] == continent_idx)
        cur_continent_neigh = set([])
        for reg_idx in reg_indices:
            neigh_indices = map_dict['neighbors'][reg_idx]
            cur_continent_neigh.update(map_dict['continent_idx'][neigh_indices].flatten())

            all_country_neigh = map_dict['region_id'][neigh_indices].flatten()
            all_country_neigh = all_country_neigh[np.all((all_country_neigh > 0,
                                                          all_country_neigh != map_dict['region_id'][reg_idx]), axis=0)]
            country_neighbors[map_dict['region_id'][reg_idx][0] - 1].update(all_country_neigh)

        continent_neighbors.append(list(cur_continent_neigh))

    map_dict['continent_neighbors'] = continent_neighbors
    map_dict['country_neighbors'] = [list(s) for s in country_neighbors]

    # convert the small polygons to the bigger country and continent regions
    for continent_idx, poly_cont in enumerate(polys.values()):
        continent_poly_list = []
        for country_idx, country_pts in enumerate(poly_cont.values()):
            country_poly = cascaded_union([Polygon(pts) for pts in country_pts])
            poly_countries.append(np.asarray(country_poly.exterior.coords.xy).T.reshape(-1, 2))
            continent_poly_list.append(country_poly)

            region_id = int(np.where(np.all(cc_idx_unique == (continent_idx + 1, country_idx + 1), axis=1))[0] + 1)
            region_map['Regions'].append(dict(
                id=region_id,
                superRegion=continent_idx + 1,
                neighbors=map_dict['country_neighbors'][region_id - 1]
            ))

        continent_poly = cascaded_union(continent_poly_list)
        poly_continents.append(np.asarray(continent_poly.exterior.coords.xy).T.reshape(-1, 2))

        nr_countries = len(continent_poly_list)
        bonus = min(1, nr_countries + np.random.randint(-1, 1))
        region_map['SuperRegions'].append(dict(id=continent_idx + 1, bonus=bonus))

    return poly_continents, poly_countries, region_map


def neighbors_to_adjacency(neighbor_list):
    n = len(neighbor_list)
    mat = np.zeros((n, n), np.int32)

    for idx, neigh in enumerate(neighbor_list):
        neigh = np.asarray(neigh)
        neigh = neigh[neigh != idx+1]  # remove loops
        neigh = neigh[neigh > 0]  # remove 0 (sea)

        mat[idx, np.asarray(neigh)-1] = 1

    return mat


def map_to_json(vor, map_dict, visualize=False):
    poly_continents, poly_countries, region_map = polygons_from_map(vor, map_dict)

    # add sea bridges as neighbors to the region map
    cent_countries = np.asarray([np.mean(p, axis=0) for p in poly_countries])
    tri = Delaunay(cent_countries)
    neigh_countries = [find_neighbors(tri, idx) + 1 for idx in range(len(cent_countries))]

    # create the sea bridges

    # compute graph trees
    continent_neigh = map_dict['continent_neighbors']
    adj_mat = neighbors_to_adjacency(continent_neigh)
    g = csgraph.csgraph_from_dense(adj_mat)
    nr_comp, labels = csgraph.connected_components(g, directed=False)

    country_lut = np.zeros((len(region_map['Regions']), 1), np.int32)
    for idx, reg in enumerate(region_map['Regions']):
        country_lut[idx] = reg['superRegion']-1

    if nr_comp > 1:
        while len(np.unique(labels)) > 1:
            (lbl_idx1, lbl_idx2) = np.random.permutation(np.unique(labels))[0:2]
            # select random continent and region
            c_idx1 = np.random.permutation(np.where(labels == lbl_idx1)[0])[0]
            r_idx1 = np.random.permutation(np.where(country_lut == c_idx1)[0])[0]

            c_idx2 = np.random.permutation(np.where(labels == lbl_idx2)[0])[0]
            r_idx2 = np.random.permutation(np.where(country_lut == c_idx2)[0])[0]

            # append the neighbors
            region_map['Regions'][r_idx1]['neighbors'].append(r_idx2+1)
            region_map['Regions'][r_idx2]['neighbors'].append(r_idx1+1)
            # update the connected components
            labels[labels == lbl_idx2] = lbl_idx1

    if visualize:
        cmap = cm.get_cmap('Set3')
        for reg_idx, region in enumerate(region_map['Regions']):
            reg_pt = cent_countries[reg_idx, :]
            neigh_pts = cent_countries[np.asarray(region['neighbors']) - 1, :]
            nr_neigh = neigh_pts.shape[0]

            x_coords = np.column_stack((neigh_pts[:, 0], np.repeat(reg_pt[0], nr_neigh, axis=0)))
            y_coords = np.column_stack((neigh_pts[:, 1], np.repeat(reg_pt[1], nr_neigh, axis=0)))

            plt.plot(x_coords.T, y_coords.T, 'o-', cmap.colors[np.mod(reg_idx, len(cmap.colors))])

        #plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices.copy())
        # plt.plot(tri.points[:, 0], tri.points[:, 1], 'o')
        for idx, pt in enumerate(cent_countries):
            text = '{}: {}'.format(idx + 1, sorted(region_map['Regions'][idx]['neighbors']))
            plt.text(pt[0], pt[1], text)

    return region_map
