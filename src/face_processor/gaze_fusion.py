#!/usr/bin/env python3  
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped, Point
from face_processor.msg import CovarianceMatrix
from collections import deque
from face_processor.msg import PixelCoords
from tf import TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf import transformations
from geometry_msgs.msg import Quaternion
from math import pi
from scipy.interpolate import interp2d
from face_processor.gaze_utils import get_calib_points


class GazeFusion():
    def __init__(self, undistort_gaze=False) -> None:
        self._gaze_intersection = np.zeros(2)
        self._head_intersection = np.zeros(2)
        self._gaze_seq = deque(maxlen=5)
        self._fused_seq = deque(maxlen=5)
        self._gaze_intersection_weight = 2
        self._head_intersection_weight = 0
        self._tfl = TransformListener()
        self._cov_mat = CovarianceMatrix()
        self.right_eye_visible = True
        self.left_eye_visible = True
        self._undistort_gaze = undistort_gaze
        if self._undistort_gaze:
            self._res_w = rospy.get_param("/screen_resolution/width")
            self._res_h = rospy.get_param("/screen_resolution/height")
            self._undistortion_deltas = self._load_deltas()

    def _load_deltas(self) -> np.array:
        path = rospy.get_param("gaze_fusion/calib")
        with open(path, 'rb') as f:
            return np.load(f)

    def _undistort(self, x, y) -> tuple:
        rects_pos = get_calib_points(self._res_w, self._res_h)
        labels_x = self._undistortion_deltas[:, 0]
        labels_y = self._undistortion_deltas[:, 1]
        x_f = interp2d(
            rects_pos[:, 0], rects_pos[:, 1], labels_x)
        y_f = interp2d(
            rects_pos[:, 0], rects_pos[:, 1], labels_y)
        return x-x_f(x, y), y-y_f(x, y)

    def _set_weights(self, head_weight=0.0, gaze_weight=0.0) -> None:
        self._gaze_intersection_weight = gaze_weight
        self._head_intersection_weight = head_weight

    def _update_weights(self) -> None:
        # change into a mathematical formula
        if not self.right_eye_visible and not self.left_eye_visible:
            self._head_intersection_weight = 2
            self._gaze_intersection_weight = 0
        elif not self.right_eye_visible or not self.left_eye_visible:
            self._head_intersection_weight = 1
            self._gaze_intersection_weight = 1
        else:
            self._head_intersection_weight = 0
            self._gaze_intersection_weight = 2

    def append_gaze_coord(self, point) -> None:
        x, y = point[0], point[1]
        if self._undistort_gaze:
            x, y = self._undistort(x, y)
        self._gaze_intersection[0], self._gaze_intersection[1] = x, y
        self._gaze_seq.append(self._gaze_intersection)

    def append_head_coord(self, point) -> None:
        self._head_intersection[0], self._head_intersection[1] = point[0], point[1]

    def get_gaze_point(self) -> CovarianceMatrix:
        self._update_weights()
        fused_coord = (self._head_intersection_weight * self._head_intersection +
                       self._gaze_intersection_weight * self._gaze_intersection)//2
        self._fused_seq.append(fused_coord)
        if self._fused_seq:
            data = np.array(self._fused_seq).T
            self._cov_mat.center = fused_coord
            self._cov_mat.matrix = np.cov(data, bias=True).flatten()

        return self._cov_mat
