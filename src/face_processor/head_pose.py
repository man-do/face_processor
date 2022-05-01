#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import mediapipe as mp
from tf import transformations
import math
from face_processor.msg import EyeFrames
from cv_bridge import CvBridge, CvBridgeError
from functools import partial
import geometry_msgs.msg
import tf2_msgs.msg
import time
from face_processor.geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)


class HeadPose:
    def __init__(self, debug_img=False) -> None:
        self._debug = debug_img
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=2)
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.points_idx = [33, 263, 61, 291, 199]
        self.points_idx = self.points_idx + \
            [key for (key, val) in procrustes_landmark_basis]
        self.points_idx = list(set(self.points_idx))
        self.points_idx.sort()
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.frame_height = rospy.get_param("cam_node/height")
        self.frame_width = rospy.get_param("cam_node/width")
        self.channels = rospy.get_param("cam_node/channels")
        self.focal_length = self.frame_width
        self.center = (self.frame_width / 2, self.frame_height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]], [
                0, self.focal_length, self.center[1]], [0, 0, 1]],
            dtype="double",
        )
        self.dist_coeff = np.zeros((4, 1))
        self.pcf = PCF(
            near=1,
            far=10000,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            fy=self.camera_matrix[1, 1],
        )
        self.bridge = CvBridge()

    def _pixel_coord_from_landmark(self, landmark_id, landmarks):
        landmark = landmarks.landmark[landmark_id]
        return self.mp_drawing._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, self.frame_width, self.frame_height)

    def _draw_landmarks(self, frame, landmarks) -> np.array:
        for face_landmarks in landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
        return frame

    def _draw_nose(self, frame, nose_pointer2D) -> np.array:
        nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
        return cv2.line(frame, nose_tip_2D,
                        nose_tip_2D_extended, (255, 0, 0), 2)

    def _draw_dist(self, frame,  distance) -> np.array:
        cv2.putText(
            frame, f"Distance: {distance}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def _get_eye_ROIs(self, image, face_landmarks, padding=15) -> tuple:
        pixel_coord = partial(
            self._pixel_coord_from_landmark, landmarks=face_landmarks)
        le = image[
            (pixel_coord(386)[1]-padding):
            (pixel_coord(374)[1]+padding),
            (pixel_coord(362)[0]-padding):
            (pixel_coord(263)[0]+padding)]
        re = image[
            (pixel_coord(159)[1]-padding):
            (pixel_coord(145)[1]+padding),
            (pixel_coord(33)[0]-padding):
            (pixel_coord(133)[0]+padding)]

        return le, re

    def process_frame(self, imgmsg_in) -> tuple:
        frame = self.bridge.imgmsg_to_cv2(imgmsg_in, 'rgb8')
        results = self.face_mesh.process(frame)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = multi_face_landmarks[0]

            le_window, re_window = self._get_eye_ROIs(frame, face_landmarks)

            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]
            )
            landmarks = landmarks.T
            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                landmarks.copy(), self.pcf
            )

            pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
            mp_rotation_vector, _ = cv2.Rodrigues(
                pose_transform_mat[:3, :3])
            mp_translation_vector = pose_transform_mat[:3, 3, None]

            mp_quaternion = transformations.quaternion_from_matrix(
                pose_transform_mat)
            transform_quaternion = transformations.quaternion_from_euler(
                math.pi, -math.pi, 0)
            mp_quaternion = transformations.quaternion_multiply(
                transform_quaternion, mp_quaternion)

            t = geometry_msgs.msg.TransformStamped()
            t.header.frame_id = "camera"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "head_pose"
            t.transform.translation.x = mp_translation_vector[2]/100
            t.transform.translation.y = -mp_translation_vector[0]/100
            t.transform.translation.z = -mp_translation_vector[1]/100
            t.transform.rotation.x = mp_quaternion[2]
            t.transform.rotation.y = mp_quaternion[0]
            t.transform.rotation.z = mp_quaternion[1]
            t.transform.rotation.w = mp_quaternion[3]
            tfm = tf2_msgs.msg.TFMessage([t])

            le_window = cv2.resize(le_window, (60, 60),
                                   interpolation=cv2.INTER_LANCZOS4)
            re_window = cv2.resize(re_window, (60, 60),
                                   interpolation=cv2.INTER_LANCZOS4)

            left_eye_out = self.bridge.cv2_to_imgmsg(le_window, 'rgb8')
            right_eye_out = self.bridge.cv2_to_imgmsg(re_window, 'rgb8')

            eye_frames = EyeFrames()
            eye_frames.header.stamp = rospy.Time.now()
            eye_frames.left_eye_frame = left_eye_out
            eye_frames.right_eye_frame = right_eye_out

            if self._debug:
                model_points = metric_landmarks[0:3, self.points_idx].T
                nose_tip = model_points[0]
                nose_tip_extended = 1.5 * model_points[0]
                (nose_pointer2D, jacobian) = cv2.projectPoints(
                    np.array([nose_tip, nose_tip_extended]),
                    mp_rotation_vector,
                    mp_translation_vector,
                    self.camera_matrix,
                    self.dist_coeff,
                )

                dist = np.round(mp_translation_vector[2][0]/100, 3)

                frame = self._draw_landmarks(frame, multi_face_landmarks)
                frame = self._draw_dist(frame, dist)
                frame = self._draw_nose(frame, nose_pointer2D)
                debug_out = self.bridge.cv2_to_imgmsg(frame, 'rgb8')

                return tfm, eye_frames, debug_out

            return tfm, eye_frames
