# face_processor
A ROS package which does face pose tracking, gaze tracking and emotion classification
It's intended use is to gather statistics about how humans interface with a certain GUI. Through a config file position and size of gui elements are defined and then statistics in terms of where the user it's concentrating it's gaze and the emotions it's feeling can be pulled out. Also the system gives the pose of the head.
![alt text](https://github.com/man-do/face_processor/blob/main/imgs/deepin-screen-recorder_Select%20area_20220224230342.gif "Face pose tracking")
The frames of the head screen and gaze direction on real-time.
![alt text](https://github.com/man-do/face_processor/blob/main/imgs/vokoscreen-2022-03-21_14-09-59.gif "")
![alt text](https://github.com/man-do/face_processor/blob/main/imgs/vokoscreen-2022-03-21_14-12-07.gif "")
![alt text](https://github.com/man-do/face_processor/blob/main/imgs/vokoscreen-2022-03-21_14-12-31.gif "")
For the package ROS Noetic was used and it was written in Python.

