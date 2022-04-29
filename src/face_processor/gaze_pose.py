from logging import debug
from turtle import right
import cv2
import numpy as np
from numpy import eye
from torch import softmax, true_divide
import rospy
from tf import transformations
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
import tf2_msgs.msg
from PIL import Image
from openvino.inference_engine import IECore
from pathlib import Path
import math
from scipy.special import softmax

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# fix for none returns


class GazeTracker():
    def __init__(self):
        self.bridge = CvBridge()
        self._left_eye_visible = Bool(True)
        self._right_eye_visible = Bool(True)

    def process_frame(self, eye_frames, tf_listener):
        #frame = self.bridge.imgmsg_to_cv2(cam_img, 'rgb8')
        left_eye_img = eye_frames.left_eye_frame
        right_eye_img = eye_frames.right_eye_frame
        left_in = self.bridge.imgmsg_to_cv2(left_eye_img, 'rgb8')
        left_eye_img = left_in.copy()
        right_in = self.bridge.imgmsg_to_cv2(right_eye_img, 'rgb8')
        right_eye_img = right_in.copy()
        left_in = np.transpose(left_in, (2, 0, 1))
        right_in = np.transpose(right_in, (2, 0, 1))
        left_in = np.reshape(left_in, (1, 3, 60, 60))
        right_in = np.reshape(right_in, (1, 3, 60, 60))

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

        # Create net executor once

        ie = IECore()
        gaze_estimation_model_path = Path(
            rospy.get_param("gaze_pose/gaze_estimation_model_path"))
        net = ie.read_network(gaze_estimation_model_path.with_suffix(".xml"))
        net_exec = ie.load_network(network=net, device_name='CPU')
        output = net_exec.infer(
            inputs={"head_pose_angles": angles_input,
                    "left_eye_image": left_in,
                    "right_eye_image": right_in})

        gaze_vector = output['gaze_vector']
        gaze_vector_n = gaze_vector / np.linalg.norm(gaze_vector)
        vcos = math.cos(math.radians(roll))
        vsin = math.sin(math.radians(roll))
        x = gaze_vector_n[0][0]*vcos + gaze_vector_n[0][1]*vsin
        y = -gaze_vector_n[0][0]*vsin + gaze_vector_n[0][1]*vcos
        gaze_vector_n = gaze_vector / np.linalg.norm(gaze_vector)
        quaternion = transformations.quaternion_from_euler(
            roll, -y, x)

        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = "head_pose"
        p.header.stamp = rospy.Time.now()
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = 0.05
        p.pose.orientation.x = quaternion[0]
        p.pose.orientation.y = quaternion[1]
        p.pose.orientation.z = quaternion[2]
        p.pose.orientation.w = quaternion[3]

        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "head_pose"
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "gaze_pose"
        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0.05
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        tfm = tf2_msgs.msg.TFMessage([t])

        # add file location as param
        # eyes_cascade_name = '/home/maverick/upwork/face_tracking_ros/src/face_processor/src/face_processor/haarcascade_eye_tree_eyeglasses.xml'
        # eyes_cascade = cv2.CascadeClassifier()
        # file = cv2.samples.findFile(eyes_cascade_name)
        # eyes_cascade.load(file)

        # left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_RGB2GRAY)
        # right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_RGB2GRAY)

        # left_eye = eyes_cascade.detectMultiScale(
        #     left_eye_img, 1.04, 1, minSize=(2, 2), maxSize=(50, 50))
        # right_eye = eyes_cascade.detectMultiScale(
        #     right_eye_img, 1.04, 1, minSize=(2, 2), maxSize=(50, 50))

        # try:
        #     left_eye[0]
        #     self._left_eye_visible.data = True
        # except IndexError:
        #     self._left_eye_visible.data = True

        # try:
        #     right_eye[0]
        #     self._right_eye_visible.data = True
        # except IndexError:
        #     self._right_eye_visible.data = True

        # for (x2, y2, w2, h2) in left_eye:
        #     eye_center = (int(x + x2 + w2//2), int(y + y2 + h2//2))
        #     radius = int(round((w2 + h2)*0.25))
        #     debug_img = cv2.circle(left_eye_img,
        #                            eye_center, radius, (255, 0, 0), 3)

        # debug_out = self.bridge.cv2_to_imgmsg(debug_img, 'rgb8')

        return p, tfm, True, True  # , debug_out
