#!/usr/bin/python3
# _*_ coding: UTF-8 _*_

import os
import cv2
import numpy as np

face_cascade_file = os.path.dirname(__file__) + os.sep + 'classifier' + os.sep + 'haarcascade_frontalface_default.xml'
eye_cascade_file = os.path.dirname(__file__) + os.sep + 'classifier' + os.sep + 'haarcascade_eye.xml'


def on_mouser_action_event(event, x, y, flag, param):

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
        print('左键按下')


class FrontalFaceDetection:
    """
    人脸、眼睛检测
    """

    def __init__(self):
        self.winname = '检测'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

    def launcher(self):
        """
        1. 打开摄像头创建窗口展示视频图像
        2. 人脸检测
        3. 眼睛检测
        :return:
        """

        # 创建视频展示窗口
        cv2.namedWindow(self.winname)
        # 打开摄像头
        capture = cv2.VideoCapture(0)

        while capture.isOpened():
            # 读取1帧图像数据
            status, mat = capture.read()
            # 灰度化图像,为检测人脸降低计算量
            mat_gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
            # 进行检测人脸
            faces = self.face_cascade.detectMultiScale(mat_gray, 1.3, 5)
            for (x, y, w, h) in faces:  # 可能检测到多张人脸，用for循环单独框出每张人脸
                # 进行人脸绘框
                mat_rect = cv2.rectangle(mat, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi_gray = mat_gray[y:y+h, x:x+w]
                mat_roi = mat_rect[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(mat_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.imshow(self.winname, mat)
            if cv2.waitKey(1) == ord('q'):
                break

        # 释放摄像头
        capture.release()
        # 销毁窗口
        cv2.destroyWindow(self.winname)

    def detection_substance(self):
        """
        物体检测
        :return:
        """
        cap = cv2.VideoCapture(0)

        # take first frame of the video
        ret, frame = cap.read()

        # setup initial location of window
        r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
        track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while (1):
            ret, frame = cap.read()
            if ret is True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # apply meanshift to get the new location
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)

                # Draw it on image
                x, y, w, h = track_window
                img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
                cv2.imshow('img2', img2)

                k = cv2.waitKey(60) & 0xff
                if k == 27:
                    break
                else:
                    cv2.imwrite(chr(k) + ".jpg", img2)

            else:
                break

        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
    ffd = FrontalFaceDetection()
    # ffd.launcher()
    ffd.detection_substance()




