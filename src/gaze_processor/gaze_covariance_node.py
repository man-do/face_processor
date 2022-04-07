#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped, Point
from face_processor.msg import CovarianceMatrix
from collections import deque

pub_center = rospy.Publisher('gaze_covariance/center', Point, queue_size=1)
pub_mat = rospy.Publisher('gaze_covariance/center_matrix',
                          CovarianceMatrix, queue_size=1)

"""    
Keep the last 20 gaze points and calculate covariance 
"""
gaze_point_buffer = deque(maxlen=10)
mat_array = CovarianceMatrix()


def point_cb(msg):
    # From 3D to 2D in screen plane
    x = msg.point.y
    y = msg.point.z
    gaze_point_buffer.append([x, y])
    if len(gaze_point_buffer) >= 10:
        data = np.array(gaze_point_buffer).T
        mat_array.matrix = np.cov(data, bias=True).flatten()
        center = [x, y]
        mat_array.center = center
        pub_mat.publish(mat_array)


def gaze_covariance_node():
    rospy.init_node("gaze_covariance")
    rospy.Subscriber("/gaze_processor/point_marker", PointStamped, point_cb)
    rospy.spin()


if __name__ == "__main__":
    try:
        gaze_covariance_node()
    except rospy.ROSInterruptException:
        pass
