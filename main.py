from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from kivy.app import App
from kivy.uix.widget import Widget
import numpy as np


class MainScreen(Widget):
    pass


class MyCameraApp(App):
    def build(self):
        return MainScreen()


class CameraPreview(Image):
    points = [(100, 100), (1100, 100), (100, 600), (1100, 600)]
    eye_centers = [0 for _ in range(len(points))]  # Placeholder for eye center offsets
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)

    #Drawing method to execute at intervals
    def update(self, dt):
        #Load frame
        ret, self.frame = self.capture.read()
        self.tracker()

        #Convert to Kivy Texture
        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

    def get_eye_center(self, roi_gray):
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            return (ex + ew // 2, ey + eh // 2)
        return None

    def tracker(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Детекция глаз
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in eyes:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Область глаза
            eye_roi = self.frame[y:y + h, x:x + w]

            # Нахождение самого темного пикселя внутри глаза
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY))

            # Отображение зрачка как красная точка
            cv2.circle(self.frame, (int(min_loc[0] + x), int(min_loc[1] + y)), 2, (0, 0, 255), 2)


if __name__ == '__main__':
    MyCameraApp().run()