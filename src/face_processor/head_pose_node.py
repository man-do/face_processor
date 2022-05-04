#! /usr/bin/env python3
import rospy
import traceback
from sensor_msgs.msg import Image
from face_processor.head_pose import HeadPose
import tf2_msgs.msg
from face_processor.msg import EyeFrames
import sys

debug = sys.argv[1] == 'True'
pose = HeadPose(debug_img=debug)

if debug:
    debug_img_pub = rospy.Publisher(
        "head_pose/debug_frame", Image, queue_size=1)
eye_frames_pub = rospy.Publisher(
    'head_pose/eye_frames', EyeFrames, queue_size=1)
tf_pub = rospy.Publisher("tf", tf2_msgs.msg.TFMessage, queue_size=1)


def frame_cb(img):
    try:
        if debug:
            tfm, frames, debug_img = pose.process_frame(img)
            debug_img_pub.publish(debug_img)
        else:
            tfm, frames = pose.process_frame(img)
        eye_frames_pub.publish(frames)
        tf_pub.publish(tfm)
    except TypeError:
        rospy.loginfo("/head_pose: No face detected")
    except:
        rospy.loginfo(traceback.format_exc())
        pass


def pose_tracking_node():
    rospy.init_node("head_pose", anonymous=True)
    rospy.Subscriber("cam_node/frame", Image, frame_cb)
    rospy.spin()


if __name__ == '__main__':
    try:
        pose_tracking_node()
    except rospy.ROSInterruptException:
        pass
