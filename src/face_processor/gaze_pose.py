import tensorflow as tf
import cv2
import math
from pathlib import Path
from openvino.inference_engine import IECore
import tf2_msgs.msg
import geometry_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from tf import transformations
import rospy

import numpy as np

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class GazePose():
    def __init__(self, debug=False):
        self._debug = debug
        self.bridge = CvBridge()
        self._left_eye_visible = Bool(True)
        self._right_eye_visible = Bool(True)
        self.ie = IECore()
        self.gaze_estimation_model_path = Path(
            rospy.get_param("gaze_pose/gaze_estimation_model_path"))
        self.net = self.ie.read_network(
            self.gaze_estimation_model_path.with_suffix(".xml"))
        self.net_exec = self.ie.load_network(
            network=self.net, device_name='CPU')
        self.model = tf.keras.models.load_model(
            "/home/maverick/upwork/face_tracking_ros/src/face_processor/src/face_processor/public/eye_occlussions.h5", custom_objects=None, compile=True, options=None
        )

    def process_frame(self, eye_frames, tf_listener):
        left_eye_img = eye_frames.left_eye_frame
        right_eye_img = eye_frames.right_eye_frame
        left_in = self.bridge.imgmsg_to_cv2(left_eye_img, 'rgb8')
        le = left_in.copy()
        right_in = self.bridge.imgmsg_to_cv2(right_eye_img, 'rgb8')
        re = right_in.copy()
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

        output = self.net_exec.infer(
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

        left_eye_img = cv2.resize(le, (256, 256))
        left_eye_img = np.reshape(left_eye_img, (1, 256, 256, 3))
        right_eye_img = cv2.resize(re, (256, 256))
        right_eye_img = np.reshape(right_eye_img, (1, 256, 256, 3))

        le_occlu = np.argmax(self.model.predict(left_eye_img))
        re_occlu = np.argmax(self.model.predict(right_eye_img))
        le_occlu = bool(le_occlu)
        re_occlu = bool(re_occlu)

        if self._debug:
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

            return p, tfm, le_occlu, re_occlu

        return tfm, le_occlu, re_occlu
