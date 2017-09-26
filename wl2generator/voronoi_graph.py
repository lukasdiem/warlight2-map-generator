import sys

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial._plotutils import voronoi_plot_2d


class VoronoiGraph(object):
    """
    Creates a voronoi object, relaxes it as needed, and returns it as a layout.

    Implements Fortune's Algorithm (https://en.wikipedia.org/wiki/Fortune%27s_algorithm) in python.

    1. Takes in a 2D numpy array (or generates points with given boundaries)

    2. Creates a tuple of two graphs (with networkx) representing the Delaunay triangulation (https://en.wikipedia.org/wiki/Delaunay_triangulation)

    3. Relaxes the points through Lloyd's Algorithm (https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)

    4. Returns a Voronoi diagram (http://www.voronoi.com/wiki/index.php?title=Main_Page)

    Code written by Peixian Wang
    From: https://github.com/peixian/kingfisher/blob/master/atlas.py
    """

    def __init__(self, points=(), dimensions=(None, None), granularity=None):
        """
        Creates the Voronoi object

        :param points: predefined points
        :type points: numpy array of shape (w, 2) where w is the number of points [x, y] style, default None
        :param dimensions: dimensions of the points, from [w, 2] where w is the highest value, this *cannot* be None if points is None
        :type dimensions: tuple of ints, maximum (x,y) dimensions, default None
        :param granularity: how many points to create, must be given if dimensions are given
        :type granularity: int, default None
        """
        if len(points) == 0 and dimensions == (None, None):
            raise ValueError('You can\'t have both points and dimensions be empty, '
                             'try passing in some points or dimensions and granularity.')
        if len(points) == 0 and dimensions != None and granularity == None:
            raise ValueError('Granularity can\'t be none if dimensions are passed in, try passing in a granularity.')
        if len(points) != 0:
            self.points = points
        else:
            points = np.random.random((granularity, 2))
            points = list(map(lambda x: np.array([x[0] * dimensions[0], x[1] * dimensions[1]]), points))
            self.points = np.array(points)
        self.bounding_region = [min(self.points[:, 0]), max(self.points[:, 0]), min(self.points[:, 1]),
                                max(self.points[:, 1])]
        self.vor = None
        self.filtered_regions = None
        self.filtered_pts_idx = None
        self.filtered_reg_idx = None

    @staticmethod
    def _eu_distance(p1, p2):
        """
        Calculates the Euclidian distance between two points

        :param p1: (x,y) position for the first point
        :type p1: tuple (or list) of floats
        :param p2: (x,y) position for the second point
        :type p2: tuple (or list) of floats

        :return: the euclidian distance
        :rtype: float
        """
        return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

    def generate_voronoi(self):
        """
        Uses scipy.spatial.Voronoi to generate a voronoi diagram.
        Filters viable regions and stashes them in filtered_regions, see https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells

        :return: A voronoi diagram based on the points
        :rtype: scipy.spatial.Voronoi
        """
        eps = sys.float_info.epsilon
        self.vor = Voronoi(self.points)
        self.filtered_regions = []
        self.filtered_pts_idx = []
        self.filtered_reg_idx = []
        for reg_idx, region in enumerate(self.vor.regions):
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = self.vor.vertices[index, 0]
                    y = self.vor.vertices[index, 1]
                    if not (self.bounding_region[0] - eps <= x and x <= self.bounding_region[1] + eps and
                                        self.bounding_region[2] - eps <= y and y <= self.bounding_region[3] + eps):
                        flag = False
                        break
            if region != [] and flag:
                self.filtered_regions.append(region)
                self.filtered_pts_idx.append(np.where(self.vor.point_region == reg_idx)[0][0])
                self.filtered_reg_idx.append(reg_idx)
        return self.vor

    @staticmethod
    def polygon_area(pts):
        x = pts[0]
        y = pts[1]

        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _region_centroid(self, vertices):
        """
        Finds the centroid of the voronoi region bounded by given vertices
        See: https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon

        :param vertices: list of vertices that bound the region
        :type vertices: numpy array of vertices from the scipy.spatial.Voronoi.regions (e.g. vor.vertices[region + [region[0]], :])

        :return: list of centroids
        :rtype: np.array of centroids
        """
        signed_area = 0
        C_x = 0
        C_y = 0
        for i in range(len(vertices) - 1):
            step = (vertices[i, 0] * vertices[i + 1, 1]) - (vertices[i + 1, 0] * vertices[i, 1])
            signed_area += step
            C_x += (vertices[i, 0] + vertices[i + 1, 0]) * step
            C_y += (vertices[i, 1] + vertices[i + 1, 1]) * step
        signed_area *= 0.5
        C_x *= (1.0 / (6.0 * signed_area))
        C_y *= (1.0 / (6.0 * signed_area))
        return np.array([[C_x, C_y]])

    def relax_points(self, times=1):
        """
        Relaxes the points after an initial Voronoi is created to refine the graph.
        See: https://stackoverflow.com/questions/17637244/voronoi-and-lloyd-relaxation-using-python-scipy

        :param times: Number of times to relax, default is 1
        :type times: int

        :return: the final voronoi diagrama
        :rtype: scipy.spatial.Voronoi
        """
        if self.vor is None:
            self.generate_voronoi()

        for i in range(times):
            centroids = []
            for region in self.filtered_regions:
                vertices = self.vor.vertices[region + [region[0]], :]
                centroid = self._region_centroid(vertices)
                centroids.append(list(centroid[0, :]))
            self.points = centroids
            self.generate_voronoi()
        return self.vor

    def draw(self, canvas):
        if self.vor is None:
            return

        voronoi_plot_2d(self.vor, canvas)
