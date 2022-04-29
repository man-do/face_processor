#!/usr/bin/env python3
import rospy
import numpy as np
from tkinter import *
from face_processor.msg import CovarianceMatrix
import math
from numpy.linalg import eig


def covariance_ell_radius_angle(flat_cov_mat: CovarianceMatrix) -> tuple:
    """ Returns radiuses and rotation angle of ellipse from covariance matrix """
    cov_mat = np.reshape(flat_cov_mat, (2, 2))
    w, v = eig(cov_mat)
    rotation_angle = np.arctan2(*v[0]) * 180 / np.pi
    # 3rd standard deviation
    radius_h_metric = 2*np.sqrt(w[0]*5.991)
    radius_v_metric = 2*np.sqrt(w[1]*5.991)
    return radius_v_metric, radius_h_metric, rotation_angle


def poly_oval(x0, y0, x1, y1, steps=20, rotation=0) -> list:
    """ Returns ellipse poly points for shapely """
    rotation = rotation * math.pi / 180.0
    a = (x1 - x0) / 2.0
    b = (y1 - y0) / 2.0
    xc = x0 + a
    yc = y0 + b

    point_list = []
    for i in range(steps):
        theta = (math.pi * 2) * (float(i) / steps)

        x1 = a * math.cos(theta)
        y1 = b * math.sin(theta)

        x = (x1 * math.cos(rotation)) + (y1 * math.sin(rotation))
        y = (y1 * math.cos(rotation)) - (x1 * math.sin(rotation))

        point_list.append([round(x + xc), round(y + yc)])

    return point_list


def flatten_list(list) -> list:
    """ Reduces dimensions of list to one """
    return [el for subl in list for el in subl]
