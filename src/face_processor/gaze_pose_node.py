#!/usr/bin/env python3
import rospy
import tf
from face_processor.gaze_pose import GazePose
from geometry_msgs.msg import PoseStamped
from face_processor.msg import EyeFrames
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool

gaze_tracker = GazePose()
rospy.init_node("gaze_pose")
pose_pub = rospy.Publisher(
    "gaze_pose/pose", PoseStamped, queue_size=1)
pose_tf_pub = rospy.Publisher(
    "/tf", TFMessage, queue_size=1)
left_eye_visibility_pub = rospy.Publisher(
    "gaze_pose/left_eye_visible", Bool, queue_size=1)
right_eye_visibility_pub = rospy.Publisher(
    "gaze_pose/right_eye_visible", Bool, queue_size=1)

tf_listener = tf.TransformListener()


def eye_frames_cb(eye_frames):
    try:
        p, tfm, le_vis, re_vis = gaze_tracker.process_frame(
            eye_frames, tf_listener)
        pose_pub.publish(p)
        pose_tf_pub.publish(tfm)
        left_eye_visibility_pub.publish(le_vis)
        right_eye_visibility_pub.publish(re_vis)
    except ValueError:
        rospy.loginfo("Could not infer gaze pose.")


def gaze_tracking_node():
    rate = rospy.Rate(60)
    rospy.Subscriber(
        'head_pose/eye_frames', EyeFrames, eye_frames_cb)
    rospy.spin()


if __name__ == '__main__':
    try:
        gaze_tracking_node()
    except rospy.ROSInterruptException:
        pass
