#!/usr/bin/env python3
import rospy
import tf
from face_processor.gaze_tracking import GazeTracker
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from face_processor.msg import EyeFrames
from tf2_msgs.msg import TFMessage


gaze_tracker = GazeTracker()
rospy.init_node("gaze_tracking")
pose_pub = rospy.Publisher(
    "gaze_tracking/pose", PoseStamped, queue_size=1)
pose_tf_pub = rospy.Publisher(
    "/tf", TFMessage, queue_size=1)
tf_listener = tf.TransformListener()


def frame_cb(eye_frames):
    p, tfm = gaze_tracker.process_frame(eye_frames, tf_listener)
    pose_pub.publish(p)
    pose_tf_pub.publish(tfm)


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
