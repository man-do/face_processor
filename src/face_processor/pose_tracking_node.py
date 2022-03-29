#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from face_processor.pose_tracking import PoseTracker
import tf2_msgs.msg
from face_processor.msg import EyeFrames

pose_tracker = PoseTracker()
debug_img_pub = rospy.Publisher(
    "pose_tracking/debug_frame", Image, queue_size=1)
eyes_img_pub = rospy.Publisher(
    'pose_tracking/eye_frames', EyeFrames, queue_size=1)
tf_pub = rospy.Publisher("tf", tf2_msgs.msg.TFMessage, queue_size=1)


def frame_cb(img):
    tfm, frames, debug = pose_tracker.process_frame(img)
    debug_img_pub.publish(debug)
    eyes_img_pub.publish(frames)
    tf_pub.publish(tfm)


def pose_tracking_node():
    rospy.init_node("pose_tracking", anonymous=True)
    rospy.Subscriber("cam_node/frame", Image, frame_cb)
    rospy.Rate(60)
    rospy.spin()


if __name__ == '__main__':
    try:
        pose_tracking_node()
    except rospy.ROSInterruptException:
        pass
