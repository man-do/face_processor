#!/usr/bin/env python3
import rospy
import numpy as np
from face_processor.msg import GazeCovariance
from face_processor.gaze_classification import GazeClassifier
from std_msgs.msg import String
from face_processor.gaze_utils import *
from shapely.geometry import box, Polygon

size_w = rospy.get_param("/screen_size/width")
size_h = rospy.get_param("/screen_size/height")
res_w = rospy.get_param("/screen_resolution/width")
res_h = rospy.get_param("/screen_resolution/height")
el_names = rospy.get_param("gui_elements/element_names")


def gaze_cb(msg):
    try:
        classifier.update(msg)
        result_dist = classifier.distance_classify()
        result_inter = classifier.intersection_classify()
        inter_selected_el_pub.publish(result_inter)
        dist_selected_el_pub.publish(result_dist)
    except ValueError:
        pass


rospy.init_node("gaze_classification")
rospy.Subscriber("gaze_fusion/center_matrix", GazeCovariance, gaze_cb)
inter_selected_el_pub = rospy.Publisher(
    "gaze_classification/intersection_selected", String, queue_size=1)
dist_selected_el_pub = rospy.Publisher(
    "gaze_classification/distance_selected", String, queue_size=1)

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
rospy.spin()
