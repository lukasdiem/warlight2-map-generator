import argparse
import json
import time

import cv2
import numpy as np
from path import Path

from wl2generator.utils import JSONNumpyEncoder
from wl2generator.voronoi_graph import VoronoiGraph
from wl2generator.voronoi_map import create_map, polygons_from_map, map_to_json


def create_image(out_path, file_prefix, map_dict, vor, graph_size, padding=50):
    # initialize the output path
    out_path = Path(out_path)
    out_path.makedirs_p()
    # collect the needed infos
    poly_continents, poly_countries, _ = polygons_from_map(vor, map_dict)
    # get the image size
    img_size = [graph_size[1] + 2 * padding, graph_size[0] + 2 * padding, 4]

    # write one image per country
    border_image = np.zeros(img_size, dtype=np.uint8)
    country_border_thickness = 2
    continent_border_thickness = 6
    for idx, poly in enumerate(poly_countries):
        image = np.zeros(img_size, dtype=np.uint8)

        out_poly = poly.copy().astype(np.int32) + padding
        cv2.fillPoly(image, [out_poly], (255, 255, 255, 255))
        cv2.polylines(border_image, [out_poly], isClosed=True, color=(0, 0, 0, 255), thickness=country_border_thickness)

        out_file = out_path / file_prefix + '_country_{:03}.png'.format(idx)
        cv2.imwrite(out_file, image)

    # write the continent borders
    for poly in poly_continents:
        out_poly = poly.copy().astype(np.int32) + padding
        cv2.polylines(border_image, [out_poly], isClosed=True, color=(0, 0, 0, 255),
                      thickness=continent_border_thickness)

    out_file = out_path / file_prefix + '_borders.png'
    cv2.imwrite(out_file, border_image)

    # write out the sea bridges
    sea_bridge_image = np.zeros(img_size, dtype=np.uint8)
    bridge_indices = map_dict['sea_bridges']
    country_centroids = map_dict['country_centroids']
    pts1 = country_centroids[bridge_indices[:, 0] - 1, :]
    pts2 = country_centroids[bridge_indices[:, 1] - 1, :]
    for idx in range(pts1.shape[0]):
        cv2.line(sea_bridge_image, tuple(pts1[idx, :].astype(np.int)),
                 tuple(pts2[idx, :].astype(np.int)), color=(255, 255, 255, 255), thickness=4)
    out_file = out_path / file_prefix + '_sea_bridges.png'
    cv2.imwrite(out_file, sea_bridge_image)


def check_size_type(value):
    val_list = value.lower().split('x')
    if len(val_list) != 2:
        raise argparse.ArgumentTypeError('{} is not a valid size. Please use the format "[w]x[h]", where w is the width'
                                         'and h is the height.'.format(value))
    return int(val_list[0]), int(val_list[1])


def check_range_type(value, dtype=int):
    val_list = value.lower().split('-')
    if len(val_list) != 2:
        raise argparse.ArgumentTypeError('{} is not a valid range. Please use the format "[l]x[u]", where l is the'
                                         'lower and u is the upper bound of the range.'.format(value))
    return dtype(val_list[0]), dtype(val_list[1])


def parse_input():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_path', help='Output directory of the map files. The directory structure'
                                                    'will be created if it does not exist.')
    parser.add_argument('-n', '--nr_maps', type=int, help='Number of maps that are created.', default=1)
    parser.add_argument('-p', '--prefix', default='map', help='File prefix that will be appended to all output files')
    parser.add_argument('-gs', '--graph_size', default='1300x1000', type=check_size_type,
                        help='The width and height of the graph (e.g. 1300x1000)')
    parser.add_argument('-gg', '--graph_granularity', default=4000, type=int,
                        help='The granularity of the graph. Sets the number of points that are initialized for '
                             'the voronoi graph.', )
    parser.add_argument('-mr', '--map_regions', default='3-8', type=check_range_type,
                        help='Lower and upper limit of regions (countries) per "super region". '
                             'Just use a string formatted like this: "2-4".')
    parser.add_argument('-ms', '--map_super_regions', default='3-5', type=check_range_type,
                        help='Lower and upper limit of "super regions" (continents). '
                             'Just use a string formatted like this: "2-4".')
    parser.add_argument('-rs', '--region_size', default='20-70', type=check_range_type,
                        help='Lower and upper limit of the region (country) size. '
                             'Just use a string formatted like this: "20-70".')

    in_args = parser.parse_args()
    return vars(in_args)


def main(params):
    graph_size = params.get('graph_size', (1300, 1000))
    out_path = Path(params.get('output_path', ''))
    prefix = params.get('prefix', 'map')

    for idx in range(params.get('nr_maps', 1)):
        ts = time.time()
        vg = VoronoiGraph(dimensions=graph_size,
                          granularity=params.get('graph_granularity', 4000))
        vg.relax_points(times=2)
        vor = vg.vor

        map_dict = create_map(vg, params.get('map_super_regions', (2, 4)),
                              params.get('map_regions', (3, 8)),
                              params.get('region_size', (20, 70)))

        print('Map creation took: {:.4f}s'.format(time.time() - ts))

        json_map = map_to_json(vor, map_dict)
        json_file = out_path / prefix + '_{:03d}.json'.format(idx)
        with open(json_file, 'w') as fp:
            json.dump(json_map, fp, cls=JSONNumpyEncoder)

        create_image(out_path, prefix + '_{:03d}'.format(idx),
                     map_dict, vor, graph_size)


if __name__ == '__main__':
    params = parse_input()
    main(params)
