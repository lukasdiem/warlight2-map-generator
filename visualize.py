import time

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.spatial._plotutils import voronoi_plot_2d

from wl2generator.voronoi_graph import VoronoiGraph
from wl2generator.voronoi_map import create_map

ts = time.time()
vg = VoronoiGraph(dimensions=(1024, 768), granularity=3000)
vg_pts = vg.relax_points(times=2)
vor = vg.vor

map_dict = create_map(vg, (4, 6), (3, 7), (50, 100))
print('Map creation took: {:.4f}s'.format(time.time() - ts))

mapper = cm.ScalarMappable(norm=Normalize(0, max(map_dict['continent_idx']) + 1), cmap='Set3')

axes = plt.gca()
voronoi_plot_2d(vor, axes, show_points=False, show_vertices=False, s=1)

for idx, pt in enumerate(vor.points):
    region = vor.regions[vor.point_region[idx]]
    continent_idx = map_dict['continent_idx'][idx][0]
    country_idx = map_dict['country_idx'][idx][0]
    if continent_idx > 0 and -1 not in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(continent_idx))

        # text = '{}/{}'.format(map['continent_idx'][idx], map['country_idx'][idx])
        text = '{:d}'.format(country_idx)
        plt.text(pt[0], pt[1], text, ha='center')

        # cent = vg._region_centroid(region)
        # plt.plot(cent[0], cent[1], 'r*')

plt.show()
