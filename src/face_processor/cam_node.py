#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

bridge = CvBridge()


def usb_cam():
    rospy.init_node("cam_node", anonymous=True)
    frame_pub = rospy.Publisher("cam_node/frame", Image, queue_size=1)
    id = rospy.get_param("/cam_node/cam_id")
    cap = cv2.VideoCapture(id)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_msg = bridge.cv2_to_imgmsg(frame, 'rgb8')
                frame_pub.publish(frame_msg)
            except CvBridgeError as e:
                rospy.loginfo(e)
                rospy.loginfo(f"Camera at index: {id} not found.")
                cap.release()
        else:
            cap.release()


if __name__ == '__main__':
    try:
        usb_cam()
    except rospy.ROSInterruptException:
        pass
