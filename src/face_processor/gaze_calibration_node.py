#!/usr/bin/env python3
from mimetypes import init
import tkinter
import rospy
import numpy as np
from face_processor.msg import GazeCovariance
from face_processor.gaze_classification import GazeClassifier
from std_msgs.msg import String
from face_processor.gaze_utils import *
from tkinter import Tk
from shapely.geometry import box
from functools import partial
from face_processor.msg import PixelCoords

gaze_point = [0, 0]
calib_points = []


def gaze_cb(msg):
    global gaze_point
    gaze_point[0] = msg.x_pos
    gaze_point[1] = msg.y_pos


rospy.init_node("calibration_node")

rospy.Subscriber(
    "intersection_points/gaze_point_pixel_coordinates", PixelCoords, gaze_cb)

size_w = rospy.get_param("/screen_size/width")
size_h = rospy.get_param("/screen_size/height")
res_w = rospy.get_param("/screen_resolution/width")
res_h = rospy.get_param("/screen_resolution/height")

sq_sz = 20
sq_cl = "green"
rects_pos = [(20, 20), (940, 20), (1880, 20),
             (20, 520), (940, 520), (1880, 520),
             (20, 1040), (940, 1040), (1880, 1040)]

calib_couples = []


def getorigin(eventorigin):
    global x, y
    x = eventorigin.x
    y = eventorigin.y
    print(x, y)


root = Tk()
root.attributes('-fullscreen', True)
# root.geometry(f"{res_w}x{res_h}")
root.wait_visibility(root)
root.attributes('-alpha', 0.3)
myCanvas = Canvas(root, width=res_w, height=res_h)


def on_click(item, event=None):
    global gaze_point
    global calib_points
    item_coords = np.array(rects_pos[item-1])
    gaze_point = np.array(gaze_point)
    delta = gaze_point - item_coords
    calib_points.append(delta)
    myCanvas.delete(f"{item}")
    if not myCanvas.find_all():
        calib_points = np.array(calib_points)
        with open("src/face_processor/src/face_processor/data.npy", 'wb') as f:
            np.save(f, calib_points)


def rectangle(x, y):
    item_id = myCanvas.create_rectangle(
        x - sq_sz//2, y - sq_sz//2, x + sq_sz//2, y + sq_sz//2, fill=sq_cl)
    myCanvas.tag_bind(item_id, '<Button-1>', partial(on_click, item_id))


for pos in rects_pos:
    rectangle(*pos)


myCanvas.pack()
# rospy.spin()

root.mainloop()
