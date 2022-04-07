#!/usr/bin/env python3
import rospy
import numpy as np
from tkinter import *
from face_processor.msg import CovarianceMatrix
import math
from numpy.linalg import eig


def point_cb(msg):
    x = (msg.center[0] + size_w/2)*coef_horizontal
    y = (msg.center[1] + size_h/2)*coef_vertical
    y = res_h - y
    cov_mat = np.array(msg.matrix)
    cov_mat = np.reshape(cov_mat, (2, 2))
    w, v = eig(cov_mat)
    rotation_angle = np.arctan2(*v[0]) * 180 / np.pi - 90
    radius_h_metric = 2*np.sqrt(w[1]*5.991)
    radius_w_metric = 2*np.sqrt(w[0]*5.991)
    radius_h = (radius_h_metric + size_w/2)*200
    radius_w = (radius_w_metric + size_w/2)*200
    # print(f"{x},{y}")
    myCanvas.delete('all')
    myCanvas.create_polygon(
        tuple(poly_oval(x-radius_w, y-radius_h, x+radius_w, y+radius_h, rotation=rotation_angle)), fill='red', outline='black')
    myCanvas.after(10, lambda: myCanvas.create_circle(x, y, 40))

    # print(rotation_angle)


def poly_oval(x0, y0, x1, y1, steps=20, rotation=0):
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

        point_list.append(round(x + xc))
        point_list.append(round(y + yc))

    return point_list


rospy.init_node("gaze_point_window")

size_w = rospy.get_param("/screen_size/width")
size_h = rospy.get_param("/screen_size/height")
res_w = rospy.get_param("/screen_resolution/width")
res_h = rospy.get_param("/screen_resolution/height")

coef_horizontal = res_w/size_w
coef_vertical = res_h/size_h

root = Tk()

root.geometry(f"{res_w}x{res_h}")

root.wait_visibility(root)

root.attributes('-alpha', 0.3)

myCanvas = Canvas(root, width=res_w, height=res_h)
myCanvas.pack()

sub = rospy.Subscriber("gaze_covariance/center_matrix",
                       CovarianceMatrix, point_cb)

root.mainloop()
