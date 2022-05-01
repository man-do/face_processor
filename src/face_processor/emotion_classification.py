#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import mediapipe as mp
import cv2
from openvino.inference_engine import IECore
from pathlib import Path
from face_processor.msg import Emotions


class EmotionClassifier():
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.frame_height = rospy.get_param("cam_node/height")
        self.frame_width = rospy.get_param("cam_node/width")
        self.channels = rospy.get_param("cam_node/channels")
        self.bridge = CvBridge()

    def process_frame(self, frame) -> Emotions:
        frame = self.bridge.imgmsg_to_cv2(frame, 'rgb8')
        results = self.face_detection.process(frame)
        if results.detections:
            for detection in results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                rect_start_point = self.mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, self.frame_width,
                    self.frame_height)
                rect_end_point = self.mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, self.frame_width,
                    self.frame_height)
                detected_face = frame[
                    rect_start_point[1]:rect_end_point[1],
                    rect_start_point[0]:rect_end_point[0]]

                detected_face = cv2.resize(detected_face, (64, 64))
                face_input = np.transpose(detected_face, (2, 0, 1))
                face_input = np.reshape(face_input, (1, 3, 64, 64))

                ie = IECore()
                model_path = Path(rospy.get_param(
                    "emotion_classification/model_path"))
                net = ie.read_network(model_path.with_suffix(".xml"))
                net_exec = ie.load_network(network=net, device_name='CPU')
                output = net_exec.infer(
                    inputs={"data": face_input})

                prob_emotion = output["prob_emotion"]
                prob_emotion = np.reshape(prob_emotion, (5))

                emotion_names = ["Neutral", "Happy",
                                 "Sad", "Surprise", "Anger"]
                dominant_emotion = emotion_names[np.argmax(prob_emotion)]
                emotions = Emotions()
                emotions.header.stamp = rospy.Time.now()
                emotions.Neutral = prob_emotion[0]
                emotions.Happy = prob_emotion[1]
                emotions.Sad = prob_emotion[2]
                emotions.Surprise = prob_emotion[3]
                emotions.Anger = prob_emotion[4]
                emotions.Dominant_emotion = dominant_emotion

            return emotions

    def get_debug_frame(self, frame, dominant_emotion) -> np.array:
        frame = self.bridge.imgmsg_to_cv2(frame, 'rgb8')
        frame = cv2.putText(
            frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            2, (255, 0, 0), 2, cv2.LINE_AA)
        return self.bridge.cv2_to_imgmsg(frame, 'rgb8')
