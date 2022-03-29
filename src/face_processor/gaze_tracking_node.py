#!/usr/bin/env python3
import rospy
import tf
from face_processor.gaze_tracking import GazeTracker
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from face_processor.msg import EyeFrames
import message_filters


gaze_tracker = GazeTracker()
rospy.init_node("gaze_tracking")
le_pose_pub = rospy.Publisher(
    "gaze_tracking/left_eye_pose", PoseStamped, queue_size=1)
re_pose_pub = rospy.Publisher(
    "gaze_tracking/right_eye_pose", PoseStamped, queue_size=1)
debug_img_pub = rospy.Publisher(
    "gaze_tracking/debug_frame", Image, queue_size=1)
tf_listener = tf.TransformListener()


def frame_cb(eye_frames):
    le_p, re_p = gaze_tracker.process_frame(eye_frames, tf_listener)
    le_pose_pub.publish(le_p)
    re_pose_pub.publish(re_p)
    # debug_img_pub.publish(debug_img)


def gaze_tracking_node():
    rate = rospy.Rate(60)
    rospy.Subscriber(
        'pose_tracking/eye_frames', EyeFrames, frame_cb)
    rospy.spin()


if __name__ == '__main__':
    try:
        gaze_tracking_node()
    except rospy.ROSInterruptException:
        pass
