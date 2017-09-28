import time

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import PolyCollection
from path import Path

from wl2generator.voronoi_graph import VoronoiGraph
from wl2generator.voronoi_map import create_map, polygons_from_map, map_to_json


def create_plot(map_dict, vor, graph_size):
    poly_continents, poly_countries, _ = polygons_from_map(vor, map_dict)

    plt.figure()
    off = 100
    #plt.imshow(Image.open('./data/bg_1920.jpg'), extent=[-off, graph_size[0] + off, -off, graph_size[1] + off])
    axes = plt.gca()
    pc_countries = PolyCollection(poly_countries, edgecolors='k', facecolors=(0.95, 0.85, 0.7), linewidths=1.0)
    pc_continents = PolyCollection(poly_continents, edgecolors='k', facecolors='none', linewidths=3.0)
    axes.add_artist(pc_countries)
    axes.add_artist(pc_continents)
    axes.set_xlim(-off, graph_size[0] + off)
    axes.set_ylim(-off, graph_size[1] + off)

    axes.set_frame_on(False)
    plt.axis('off')
    plt.tight_layout()
    # plt.savefig('map.svg')
    # plt.show()


if __name__ == '__main__':
    graph_size = (1300, 1000)
    out_path = Path('./maps_big_tmp')
    out_path.makedirs_p()

    for idx in range(1):
        ts = time.time()
        vg = VoronoiGraph(dimensions=graph_size, granularity=4000)
        vg_pts = vg.relax_points(times=2)
        vor = vg.vor

        #map_dict = create_map(vg, (2, 8), (10, 15), (20, 70))
        map_dict = create_map(vg, (3, 6), (3, 8), (20, 70))
        print('Map creation took: {:.4f}s'.format(time.time() - ts))

        create_plot(map_dict, vor, graph_size)
        map_to_json(vor, map_dict, visualize=True)
        plt.show()


        # plt.savefig(out_path / 'map_{:03d}.png'.format(idx), dpi=300, transparent=True,
        #             bbox_inches='tight', pad_inches=0, facecolor=(0, 0.4, 0.55, 0.8))
        # plt.savefig(out_path / 'map_{:03d}.svg'.format(idx), dpi=300, transparent=True,
        #             bbox_inches='tight', pad_inches=0, facecolor=(0, 0.4, 0.55, 0.8))
        # plt.close()
