#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PointStamped, TransformStamped, PoseStamped, Pose
# dont use custom message use a 2d point or vector prebuilt message
###
##
from face_processor.msg import PixelCoords
import numpy as np
import tf
import tf2_msgs.msg
from face_processor.gaze_utils import qv_mult, LinePlaneCollision
import sys

debug = sys.argv[1] == "True"
rospy.init_node("intersection_points")
tfl = tf.TransformListener()
tfb = tf.TransformBroadcaster()


def get_screen_marker(position, orientation, size_w, size_h):
    screen = Marker()
    screen.header.frame_id = 'camera'
    screen.id = 1
    screen.type = 1
    screen.action = 0
    screen.pose.position.x = position[0]
    screen.pose.position.y = position[1]
    screen.pose.position.z = position[2]
    screen.pose.orientation.x = orientation[0]
    screen.pose.orientation.y = orientation[1]
    screen.pose.orientation.z = orientation[2]
    screen.pose.orientation.w = orientation[3]
    screen.color.r = 0.0
    screen.color.g = 0.5
    screen.color.b = 0.5
    screen.color.a = 0.5
    screen.scale.x = 0.01
    screen.scale.y = size_w
    screen.scale.z = size_h

    return screen


def get_tfm(position, orientation):
    screen_transform = TransformStamped()
    screen_transform.header.stamp = rospy.Time.now()
    screen_transform.header.frame_id = "camera"
    screen_transform.child_frame_id = "screen"
    screen_transform.transform.translation.x = position[0]
    screen_transform.transform.translation.y = position[1]
    screen_transform.transform.translation.z = position[2]
    screen_transform.transform.rotation.x = orientation[0]
    screen_transform.transform.rotation.y = orientation[1]
    screen_transform.transform.rotation.z = orientation[2]
    screen_transform.transform.rotation.w = orientation[3]
    tfm = tf2_msgs.msg.TFMessage([screen_transform])

    return tfm


def gaze_processor_node():
    rate = rospy.Rate(10.0)

    screen_marker_p = rospy.Publisher(
        "/intersection_points/screen_marker_rviz", Marker, queue_size=1)

    gaze_coords = rospy.Publisher(
        "/intersection_points/gaze_point_pixel_coordinates", PixelCoords, queue_size=1)

    gaze_point_p = rospy.Publisher(
        "/intersection_points/gaze_point_marker_rviz", PointStamped, queue_size=1)
    head_coords = rospy.Publisher(
        "/intersection_points/head_point_pixel_coordinates", PixelCoords, queue_size=1)

    head_point_p = rospy.Publisher(
        "/intersection_points/head_point_marker_rviz", PointStamped, queue_size=1)

    tf_p = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)

    roll = rospy.get_param("/screen_pose/rotation/r")
    pitch = rospy.get_param("/screen_pose/rotation/p")
    yaw = rospy.get_param("/screen_pose/rotation/y")

    size_w = rospy.get_param("/screen_size/width")
    size_h = rospy.get_param("/screen_size/height")

    res_w = rospy.get_param("/screen_resolution/width")
    res_h = rospy.get_param("/screen_resolution/height")

    x = rospy.get_param("/screen_pose/translation/x")
    y = rospy.get_param("/screen_pose/translation/y")
    z = rospy.get_param("/screen_pose/translation/z")

    coef_hori = res_w/size_w
    coef_vert = res_h/size_h

    roll, pitch, yaw = np.radians((roll, pitch, yaw))
    screen_orient = tf.transformations.quaternion_from_euler(
        roll, pitch, yaw)

    vec_to_rotate = np.array([1, 0, 0])
    screen_normal = qv_mult(screen_orient, vec_to_rotate)
    screen_point = np.array([x, y, z])

    result_point = PointStamped()
    pixel_coords = PixelCoords()

    while not rospy.is_shutdown():
        screen_marker = get_screen_marker(
            [x, y, z], screen_orient, size_w, size_h)

        tfm = get_tfm([x, y, z], screen_orient)

        tf_p.publish(tfm)

        try:
            (head_translation, head_orientation) = tfl.lookupTransform(
                'world', 'head_pose', rospy.Time(0))
            (gaze_translation, gaze_orientation) = tfl.lookupTransform(
                'world', 'gaze_pose', rospy.Time(0))
            (screen_translation, screen_orient) = tfl.lookupTransform(
                'world', 'screen', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        eyes_point = np.array(
            [gaze_translation[0], gaze_translation[1], gaze_translation[2]])
        screen_point = np.array(
            [screen_translation[0], screen_translation[1], screen_translation[2]])
        gaze_direction = qv_mult(gaze_orientation, vec_to_rotate)
        gaze_screen_intersection = LinePlaneCollision(
            screen_normal, screen_point, gaze_direction, eyes_point)

        head_point = np.array(
            [head_translation[0], head_translation[1], head_translation[2]])
        head_direction = qv_mult(head_orientation, vec_to_rotate)
        head_screen_intersection = LinePlaneCollision(
            screen_normal, screen_point, head_direction, head_point)

        result_point.header.frame_id = "/world"
        result_point.point.x = gaze_screen_intersection[0]
        result_point.point.y = gaze_screen_intersection[1]
        result_point.point.z = gaze_screen_intersection[2]

        result_point = tfl.transformPoint("screen", result_point)
        gaze_point_p.publish(result_point)

        x_pix = (result_point.point.y + size_w/2)*coef_hori
        y_pix = (result_point.point.z + size_h/2)*coef_vert
        y_pix = res_h - y_pix
        pixel_coords.x_pos = int(x_pix)
        pixel_coords.y_pos = int(y_pix)
        gaze_coords.publish(pixel_coords)

        result_point.header.frame_id = "/world"
        result_point.point.x = head_screen_intersection[0]
        result_point.point.y = head_screen_intersection[1]
        result_point.point.z = head_screen_intersection[2]

        result_point = tfl.transformPoint("screen", result_point)
        head_point_p.publish(result_point)

        x_pix = (result_point.point.y + size_w/2)*coef_hori
        y_pix = (result_point.point.z + size_h/2)*coef_vert
        y_pix = res_h - y_pix
        pixel_coords.x_pos = int(x_pix)
        pixel_coords.y_pos = int(y_pix)
        head_coords.publish(pixel_coords)

        global debug
        if debug:
            screen_marker_p.publish(screen_marker)

        rate.sleep()


if __name__ == '__main__':
    try:
        gaze_processor_node()
    except rospy.ROSInterruptException:
        pass
