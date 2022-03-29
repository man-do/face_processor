from logging import debug
import cv2
import numpy as np
import rospy
from tf import transformations
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
import tf2_msgs.msg
from PIL import Image
from openvino.inference_engine import IECore
from pathlib import Path
import math

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# fix for none returns


class GazeTracker():
    def __init__(self):
        self.bridge = CvBridge()

    def process_frame(self, eye_frames, tf_listener):
        #frame = self.bridge.imgmsg_to_cv2(cam_img, 'rgb8')
        left_eye_img = eye_frames.left_eye_frame
        rigth_eye_img = eye_frames.right_eye_frame
        le_input = self.bridge.imgmsg_to_cv2(left_eye_img, 'rgb8')
        re_input = self.bridge.imgmsg_to_cv2(rigth_eye_img, 'rgb8')
        le_input = np.transpose(le_input, (2, 0, 1))
        re_input = np.transpose(re_input, (2, 0, 1))
        le_input = np.reshape(le_input, (1, 3, 60, 60))
        re_input = np.reshape(re_input, (1, 3, 60, 60))

        angles_input = tf_listener.lookupTransform(
            '/camera', '/head_pose', rospy.Time(0))
        angles_input = list(transformations.euler_from_quaternion(
            angles_input[1]))
        angles_input = np.degrees(angles_input)
        roll = angles_input[0]
        angles_input[2] = 180 - angles_input[2]
        angles_input[2] = angles_input[2] - \
            360 if angles_input[2] > 180 else angles_input[2]
        angles_input[0], angles_input[1], angles_input[2] = angles_input[2], angles_input[1], angles_input[0]

        ie = IECore()
        model_path = Path(rospy.get_param("gaze_tracking/model_path"))
        net = ie.read_network(model_path.with_suffix(".xml"))
        net_exec = ie.load_network(network=net, device_name='CPU')
        output = net_exec.infer(
            inputs={"head_pose_angles": angles_input,
                    "left_eye_image": le_input,
                    "right_eye_image": re_input})

        gaze_vector = output['gaze_vector']
        gaze_vector_n = gaze_vector / np.linalg.norm(gaze_vector)
        vcos = math.cos(math.radians(roll))
        vsin = math.sin(math.radians(roll))
        x = gaze_vector_n[0][0]*vcos + gaze_vector_n[0][1]*vsin
        y = -gaze_vector_n[0][0]*vsin + gaze_vector_n[0][1]*vcos
        gaze_vector_n = gaze_vector / np.linalg.norm(gaze_vector)
        quaternion = transformations.quaternion_from_euler(
            roll, -y, x)

        le_p = geometry_msgs.msg.PoseStamped()
        le_p.header.frame_id = "head_pose"
        le_p.header.stamp = rospy.Time.now()
        le_p.pose.position.x = 0
        le_p.pose.position.y = 0.05
        le_p.pose.position.z = 0.05
        le_p.pose.orientation.x = quaternion[0]
        le_p.pose.orientation.y = quaternion[1]
        le_p.pose.orientation.z = quaternion[2]
        le_p.pose.orientation.w = quaternion[3]

        re_p = geometry_msgs.msg.PoseStamped()
        re_p.header.frame_id = "head_pose"
        re_p.header.stamp = rospy.Time.now()
        re_p.pose.position.x = 0
        re_p.pose.position.y = -0.05
        re_p.pose.position.z = 0.05
        re_p.pose.orientation.x = quaternion[0]
        re_p.pose.orientation.y = quaternion[1]
        re_p.pose.orientation.z = quaternion[2]
        re_p.pose.orientation.w = quaternion[3]

        #debug_img = self.bridge.cv2_to_imgmsg(cam_img, 'rgb8')

        return le_p, re_p  # , debug_img