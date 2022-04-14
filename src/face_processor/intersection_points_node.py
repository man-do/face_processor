#!/usr/bin/env python3
import tkinter
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PointStamped, TransformStamped, PoseStamped, Pose
# dont use custom message use a 2d point or vector prebuilt message
###
##
from face_processor.msg import PixelCoords
from tf import transformations
import numpy as np
import tf
import tf.transformations
import tf2_msgs.msg
from tkinter import *

rospy.init_node("intersection_points")
tfl = tf.TransformListener()
tfb = tf.TransformBroadcaster()


def gaze_processor_node():
    rate = rospy.Rate(10.0)
    marker_pub = rospy.Publisher(
        "/intersection_points/screen_marker_rviz", Marker, queue_size=1)
    pixel_gaze_coords_pub = rospy.Publisher(
        "/intersection_points/gaze_point_pixel_coordinates", PixelCoords, queue_size=1)
    gaze_point_pub = rospy.Publisher(
        "/intersection_points/gaze_point_marker_rviz", PointStamped, queue_size=1)
    pixel_head_coords_pub = rospy.Publisher(
        "/intersection_points/head_point_pixel_coordinates", PixelCoords, queue_size=1)
    head_point_pub = rospy.Publisher(
        "/intersection_points/head_point_marker_rviz", PointStamped, queue_size=1)
    tf_pub = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
    marker = Marker()

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

    coef_horizontal = res_w/size_w
    coef_vertical = res_h/size_h

    roll, pitch, yaw = np.radians((roll, pitch, yaw))
    pose_quat = transformations.quaternion_from_euler(roll, pitch, yaw)

    # From this vector we get the normal and
    # direction vector transforming it with quaternion
    vec_to_rotate = np.array([1, 0, 0])
    screen_orientation = pose_quat

    screen_normal = qv_mult(screen_orientation, vec_to_rotate)
    screen_point = np.array([x, y, z])

    result_point = PointStamped()
    screen_transform = TransformStamped()
    pixel_coords = PixelCoords()

    while not rospy.is_shutdown():
        # Publish marker which models screen's size and position
        marker.header.frame_id = 'camera'
        marker.id = 1
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = pose_quat[0]
        marker.pose.orientation.y = pose_quat[1]
        marker.pose.orientation.z = pose_quat[2]
        marker.pose.orientation.w = pose_quat[3]
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.5
        marker.scale.x = 0.01
        marker.scale.y = size_w
        marker.scale.z = size_h
        marker_pub.publish(marker)

        # Publish frame for center of screen
        screen_transform.header.stamp = rospy.Time.now()
        screen_transform.header.frame_id = "camera"
        screen_transform.child_frame_id = "screen"
        screen_transform.transform.translation.x = x
        screen_transform.transform.translation.y = y
        screen_transform.transform.translation.z = z
        screen_transform.transform.rotation.x = pose_quat[0]
        screen_transform.transform.rotation.y = pose_quat[1]
        screen_transform.transform.rotation.z = pose_quat[2]
        screen_transform.transform.rotation.w = pose_quat[3]
        tfm = tf2_msgs.msg.TFMessage([screen_transform])
        tf_pub.publish(tfm)

        try:
            (head_translation, head_orientation) = tfl.lookupTransform(
                'world', 'head_pose', rospy.Time(0))
            (gaze_translation, gaze_orientation) = tfl.lookupTransform(
                'world', 'gaze_pose', rospy.Time(0))
            (screen_translation, screen_orientation) = tfl.lookupTransform(
                'world', 'screen', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        # Calc intersection of screen and gaze
        eyes_point = np.array(
            [gaze_translation[0], gaze_translation[1], gaze_translation[2]])
        screen_point = np.array(
            [screen_translation[0], screen_translation[1], screen_translation[2]])
        gaze_direction = qv_mult(gaze_orientation, vec_to_rotate)
        gaze_screen_intersection = LinePlaneCollision(
            screen_normal, screen_point, gaze_direction, eyes_point)

        # Calc intersection of screen and head orientation vector
        head_point = np.array(
            [head_translation[0], head_translation[1], head_translation[2]])
        head_direction = qv_mult(head_orientation, vec_to_rotate)
        head_screen_intersection = LinePlaneCollision(
            screen_normal, screen_point, head_direction, head_point)

        # Publish metric coordinates of gaze point
        result_point.header.frame_id = "/world"
        result_point.point.x = gaze_screen_intersection[0]
        result_point.point.y = gaze_screen_intersection[1]
        result_point.point.z = gaze_screen_intersection[2]

        # Gaze point relative to screen frame, not world frame
        result_point = tfl.transformPoint("screen", result_point)
        gaze_point_pub.publish(result_point)

        # Pub pixel coordinates of gaze point
        x_pix = (result_point.point.y + size_w/2)*coef_horizontal
        y_pix = (result_point.point.z + size_h/2)*coef_vertical
        y_pix = res_h - y_pix
        pixel_coords.x_pos = int(x_pix)
        pixel_coords.y_pos = int(y_pix)
        pixel_gaze_coords_pub.publish(pixel_coords)

        # Publish metric coordinates of head point
        result_point.header.frame_id = "/world"
        result_point.point.x = head_screen_intersection[0]
        result_point.point.y = head_screen_intersection[1]
        result_point.point.z = head_screen_intersection[2]

        # Head point relative to screen frame, not world frame
        result_point = tfl.transformPoint("screen", result_point)
        head_point_pub.publish(result_point)

        # Pub pixel coordinates of head point
        x_pix = (result_point.point.y + size_w/2)*coef_horizontal
        y_pix = (result_point.point.z + size_h/2)*coef_vertical
        y_pix = res_h - y_pix
        pixel_coords.x_pos = int(x_pix)
        pixel_coords.y_pos = int(y_pix)
        pixel_head_coords_pub.publish(pixel_coords)

        rate.sleep()


# Move these files to a utils file


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise rospy.loginfo("No intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def qv_mult(q1, v1):
    """ Rotate vector v1 by quaternion q1 """
    #v1 = tf.transformations.unit_vector(v1)
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2),
        tf.transformations.quaternion_conjugate(q1)
    )[:3]


if __name__ == '__main__':
    try:
        gaze_processor_node()
    except rospy.ROSInterruptException:
        pass
