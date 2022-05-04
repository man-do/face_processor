#!/usr/bin/env python3
import rospy
import numpy as np
from face_processor.msg import GazeCovariance
from face_processor.gaze_classification import GazeClassifier
from std_msgs.msg import String
from face_processor.gaze_utils import *
from tkinter import Tk
from shapely.geometry import box

size_w = rospy.get_param("/screen_size/width")
size_h = rospy.get_param("/screen_size/height")
res_w = rospy.get_param("/screen_resolution/width")
res_h = rospy.get_param("/screen_resolution/height")
el_names = rospy.get_param("gui_elements/element_names")

root = Tk()
root.attributes('-fullscreen', True)
root.wait_visibility(root)
root.attributes('-alpha', 0.3)
myCanvas = Canvas(root, width=res_w, height=res_h)
myCanvas.pack()


def gaze_cb(msg):
    classifier.update(msg)
    myCanvas.delete('ellipse')
    myCanvas.delete('result')
    myCanvas.create_polygon(classifier.get_ellipse_poly_tk(),
                            fill='', outline='red', width=10, tag='ellipse')


def intersection_cb(msg):
    # just search for the key
    for name, el in classifier._gui_elements.items():
        if msg.data == name:
            cx = el.centroid.x
            cy = el.centroid.y
            myCanvas.create_rectangle(
                cx-10, cy-10, cx+10, cy+10, fill='green', tags="result")


def distance_cb(msg):
    pass


rospy.init_node("gaze_gui_element")
rospy.Subscriber("gaze_fusion/center_matrix", GazeCovariance, gaze_cb)
inter_selected_sub = rospy.Subscriber(
    "gaze_classification/intersection_selected", String, intersection_cb)
dsit_selected_sub = rospy.Subscriber(
    "gaze_classification/distance_selected", String, distance_cb)

elements = {}
for name in el_names:
    center_x = rospy.get_param(f"gui_elements/{name}/center/x")
    center_y = rospy.get_param(f"gui_elements/{name}/center/y")
    width = rospy.get_param(f"gui_elements/{name}/width")
    height = rospy.get_param(f"gui_elements/{name}/height")
    minx = center_x - width//2
    miny = center_y - height//2
    maxx = center_x + width//2
    maxy = center_y + height//2
    b = box(minx, miny, maxx, maxy)
    elements.update({name: b})

classifier = GazeClassifier(elements)

for el in classifier.get_statics_poly_tk():
    myCanvas.create_polygon(el, fill='', outline='blue', width=10)

root.mainloop()
