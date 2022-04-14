#!/usr/bin/env python3
from torch import pixel_shuffle
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


class GazeFusion():
    def __init__(self) -> None:
        self._gaze_intersection = np.zeros(2)
        self._head_intersection = np.zeros(2)
        self._gaze_seq = deque(maxlen=10)
        self._fused_seq = deque(maxlen=10)
        self._gaze_intersection_weight = 2
        self._head_intersection_weight = 0
        self._tfl = TransformListener()
        self._cov_mat = CovarianceMatrix()
        self._head_orientation = Quaternion()

    def _set_weights(self, head_weight=0.0, gaze_weight=0.0) -> None:
        self._gaze_intersection_weight = gaze_weight
        self._head_intersection_weight = head_weight

    def _update_weights(self) -> None:
        try:
            _, self._head_orientation = self._tfl.lookupTransform(
                'world', 'head_pose', rospy.Time(0))
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass
        rot_quat = transformations.quaternion_from_euler(0, 0, np.radians(-90))
        self._head_orientation = transformations.quaternion_multiply(self._head_orientation, rot_quat)
        _, pitch, yaw = transformations.euler_from_quaternion(
            self._head_orientation)
        yaw = np.degrees(yaw)
        # here we set the weights
        # based on occlussions
        print(yaw)

    def append_gaze_coord(self, point) -> None:
        self._gaze_intersection[0], self._gaze_intersection[1] = point[0], point[1]
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
