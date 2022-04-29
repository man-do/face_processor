#!/usr/bin/env python3
from torch import pixel_shuffle
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped, Point
from face_processor.msg import GazeCovariance
from collections import deque
from face_processor.msg import PixelCoords
from face_processor.gaze_fusion import GazeFusion
from std_msgs.msg import Bool

rospy.init_node("gaze_covariance")
pub_mat = rospy.Publisher('gaze_fusion/center_matrix',
                          GazeCovariance, queue_size=1)

fuser = GazeFusion()


def gaze_point_cb(msg):
    fuser.append_gaze_coord([msg.x_pos, msg.y_pos])
    # x = msg.x_pos
    # y = msg.y_pos
    # gaze_point_buffer.append([x, y])
    # if gaze_point_buffer:
    #     data = np.array(gaze_point_buffer).T
    #     mat_array.matrix = np.cov(data, bias=True).flatten()
    #     center = fuser.get_gaze_point()
    #     mat_array.center = center
    #     pub_mat.publish(mat_array)
    pub_mat.publish(fuser.get_gaze_point())


def head_point_cb(msg):
    fuser.append_head_coord([msg.x_pos, msg.y_pos])


def left_visible_cb(msg):
    fuser.left_eye_visible = msg.data


def right_visible_cb(msg):
    fuser.right_eye_visible = msg.data


def gaze_covariance_node():
    rospy.Subscriber("/intersection_points/gaze_point_pixel_coordinates",
                     PixelCoords, gaze_point_cb)
    rospy.Subscriber("/intersection_points/head_point_pixel_coordinates",
                     PixelCoords, head_point_cb)
    rospy.Subscriber("gaze_pose/left_eye_visible", Bool, left_visible_cb)
    rospy.Subscriber("gaze_pose/left_eye_visible", Bool, right_visible_cb)
    rospy.spin()


if __name__ == "__main__":
    try:
        gaze_covariance_node()
    except rospy.ROSInterruptException:
        pass
