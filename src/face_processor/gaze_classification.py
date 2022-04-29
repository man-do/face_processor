#!/usr/bin/env python3
from dis import dis
from unicodedata import name
import rospy
import numpy as np
from face_processor.msg import GazeCovariance
from collections import deque
from face_processor.msg import PixelCoords
from geometry_msgs.msg import Quaternion
from collections import namedtuple
from face_processor.gaze_utils import *
from shapely.geometry import Polygon
from scipy.spatial import distance


class GazeClassifier():
    """
        Uses the gaze covariance matrix and the center of gaze to determine
        which gui element the human's gaze is focused upon
    """

    def __init__(self, elements=None) -> None:
        self._gui_elements = elements
        self._gaze_coords = [0, 0]
        self._radius_v = 0
        self._radius_h = 0
        self._rot_angle = 0
        self.gaze_ellipse_poly = None

    def update(self, cov_mat: GazeCovariance) -> None:
        """ Using CovarianceMatrix msg updates the state """
        self._gaze_coords[0] = cov_mat.center[0]
        self._gaze_coords[1] = cov_mat.center[1]
        self._radius_v, self._radius_h, self._rot_angle = covariance_ell_radius_angle(
            cov_mat.matrix)
        self.gaze_ellipse_poly = Polygon(poly_oval(self._gaze_coords[0]-self._radius_v, self._gaze_coords[1]-self._radius_h,
                                                   self._gaze_coords[0]+self._radius_v, self._gaze_coords[1]+self._radius_h, rotation=self._rot_angle))

    def intersection_classify(self) -> list:
        """ 
            Returns id of selected element based on which has 
            the largest intersection area with the ellipse
        """
        biggest_area = 0
        # result_coords = [0, 0]
        result = None
        for name, el in self._gui_elements.items():
            area = self.gaze_ellipse_poly.intersection(el).area
            if area > biggest_area:
                area = biggest_area
                # result_coords[0] = el.centroid.x
                # result_coords[1] = el.centroid.y
                result = name
        return result

    def distance_classify(self) -> list:
        """ 
            Returns id of selected element based on which element
            is closer to the gaze point
        """
        smallest_dist = 1000
        result = None
        for name, el in self._gui_elements.items():
            el_center = el.centroid
            gaze_center = self._gaze_coords
            dist = distance.euclidean(
                np.array(el_center), np.array(gaze_center))
            if dist < smallest_dist:
                smallest_dist = dist
                result = name
        return result

    def get_statics_poly_tk(self) -> list:
        """
            Yields poly points usable by tkinter for static gui elements
        """
        for name, el in self._gui_elements.items():
            points = []
            x, y = el.exterior.xy
            for i in range(len(x)):
                points.append(int(x[i]))
                points.append(int(y[i]))
                yield points

    def get_ellipse_poly_tk(self) -> list:
        """
            Returns poly points usable by tkinter for gaze_ellipse
        """
        for el in self._gui_elements:
            points = []
            x, y = self.gaze_ellipse_poly.exterior.xy
            for i in range(len(x)):
                points.append(int(x[i]))
                points.append(int(y[i]))
        return points
