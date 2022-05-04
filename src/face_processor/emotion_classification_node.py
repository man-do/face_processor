#!/usr/bin/env python3
from face_processor.emotion_classification import EmotionClassifier
import rospy
import traceback
from sensor_msgs.msg import Image
from face_processor.msg import Emotions
import sys

debug = sys.argv[1] == "True"
rospy.init_node("emotion_classification")
emotion_classifier = EmotionClassifier()
emotion_pub = rospy.Publisher(
    "emotion_classification/emotion", Emotions, queue_size=1)

if debug:
    debug_frame_pub = rospy.Publisher(
        "emotion_classification/debug_frame", Image, queue_size=1)


def callback(imgmsg_in):
    try:
        emotions = emotion_classifier.process_frame(imgmsg_in)
        emotion_pub.publish(emotions)
        if debug:
            debug_frame = emotion_classifier.get_debug_frame(
                imgmsg_in, emotions.Dominant_emotion)
            debug_frame_pub.publish(debug_frame)
    except TypeError:
        rospy.loginfo("/emotion_classification: No face detected")
    except:
        rospy.loginfo(traceback.format_exc())
        pass


def emotion_classification_node():
    rospy.Subscriber("cam_node/frame", Image, callback)
    rospy.spin()


if __name__ == '__main__':
    try:
        emotion_classification_node()
    except rospy.ROSInterruptException:
        pass
