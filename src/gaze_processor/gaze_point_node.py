#!/usr/bin/env python3
import tkinter
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Vector3, PointStamped, TransformStamped, PoseStamped, Pose
from tf import transformations
import numpy as np
import tf
import tf.transformations
import tf2_msgs.msg
from tkinter import *


def pose_cb(msg):
    gaze_pose = msg.pose


rospy.init_node("gaze_processor")
rospy.Subscriber("gaze_tracking/pose", PoseStamped, pose_cb)
tfl = tf.TransformListener()
tfb = tf.TransformBroadcaster()

gaze_pose = Pose()


def gaze_processor_node():
    rate = rospy.Rate(10.0)
    marker_pub = rospy.Publisher(
        "/gaze_processor/screen_marker", Marker, queue_size=1)
    point_pub = rospy.Publisher(
        "/gaze_processor/point_marker", PointStamped, queue_size=1)
    tf_pub = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
    marker = Marker()

    roll = rospy.get_param("/screen_pose/rotation/r")
    pitch = rospy.get_param("/screen_pose/rotation/p")
    yaw = rospy.get_param("/screen_pose/rotation/y")

    size_w = rospy.get_param("/screen_size/width")
    size_h = rospy.get_param("/screen_size/height")

    x = rospy.get_param("/screen_pose/translation/x")
    y = rospy.get_param("/screen_pose/translation/y")
    z = rospy.get_param("/screen_pose/translation/z")

    roll, pitch, yaw = np.radians((roll, pitch, yaw))

    pose_quat = transformations.quaternion_from_euler(roll, pitch, yaw)

    vec_to_rotate = np.array([1, 0, 0])

    screen_orientation = pose_quat

    screen_normal = qv_mult(screen_orientation, vec_to_rotate)

    screen_point = np.array([x, y, z])

    result_point = PointStamped()

    screen_transform = TransformStamped()

    while not rospy.is_shutdown():
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
            (gaze_translation, gaze_orientation) = tfl.lookupTransform(
                'world', 'gaze_pose', rospy.Time(0))
            (screen_translation, screen_orientation) = tfl.lookupTransform(
                'world', 'screen', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        head_point = np.array(
            [gaze_translation[0], gaze_translation[1], gaze_translation[2]])
        screen_point = np.array(
            [screen_translation[0], screen_translation[1], screen_translation[2]])

        gaze_normal = qv_mult(gaze_orientation, vec_to_rotate)

        result = LinePlaneCollision(
            screen_normal, screen_point, gaze_normal, head_point)

        #result_point.header.stamp = rospy.Time.now()
        result_point.header.frame_id = "/world"
        result_point.point.x = result[0]
        result_point.point.y = result[1]
        result_point.point.z = result[2]

        # point_pub.publish(result_point)

        result = tfl.transformPoint("screen", result_point)

        # print(result)

        point_pub.publish(result)

        #print(f"Screen normal: {screen_normal}")
        #print(f"Result: {result.point.y},{result.point.z}")
        #print(f"Gaze point: {result}")

        rate.sleep()


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise rospy.loginfo("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def qv_mult(q1, v1):
    # rotate vector v1 by quaternion q1
    # comment this out if v1 doesn't need to be a unit vector
    v1 = tf.transformations.unit_vector(v1)
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
