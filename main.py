from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from kivy.app import App
from kivy.uix.widget import Widget
from gaze_tracking import GazeTracking
from scipy.linalg import solve
from kivy.core.window import Window
import numpy as np
import csv
from datetime import datetime

class MainScreen(Widget):
    pass

class MyCameraApp(App):
    def build(self):
        # Установить полноэкранный режим
        Window.fullscreen = 'auto'
        return MainScreen()

class CameraPreview(Image):
    is_calibration = True
    # Установим начальные координаты калибровочных точек, которые будут обновлены при инициализации
    points = [(30, 30), (Window.width - 30, 30), (30, Window.height - 30), (Window.width - 30, Window.height - 30)]
    w = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    h = [[], [], [], []]
    h1 = []
    d = []

    def save_boundary_points(self, file_name, boundary_points):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Границы', 'x', 'y', 'time'])
            for i, point in enumerate(boundary_points):
                writer.writerow([i, point[0], point[1], "---"])
            writer.writerow(['Точка', 'x', 'y', 'time'])

    def save_point_to_csv(self, x, y, file_name):
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['-', x, y, datetime.now()])


    def draw_calibration_point(self, f, point, r=10):
        cv2.circle(f, point, r, (0, 0, 255), -1)


    def transform_point(self, quadrilateral_points, rectangle_points, inner_point):
        A = np.array([
            [quadrilateral_points[0][0], quadrilateral_points[0][1], 1, 0, 0, 0],
            [0, 0, 0, quadrilateral_points[0][0], quadrilateral_points[0][1], 1],
            [quadrilateral_points[1][0], quadrilateral_points[1][1], 1, 0, 0, 0],
            [0, 0, 0, quadrilateral_points[1][0], quadrilateral_points[1][1], 1],
            [quadrilateral_points[3][0], quadrilateral_points[3][1], 1, 0, 0, 0],
            [0, 0, 0, quadrilateral_points[3][0], quadrilateral_points[3][1], 1],
        ])

        b = np.array([
            rectangle_points[0][0],
            rectangle_points[0][1],
            rectangle_points[1][0],
            rectangle_points[1][1],
            rectangle_points[3][0],
            rectangle_points[3][1]
        ])

        c = solve(A, b)

        x, y, z = c[0] * inner_point[0] + c[1] * inner_point[1] + c[2], c[3] * inner_point[0] + c[4] * inner_point[1] + c[5], 1

        new_x = x / z
        new_y = y / z
        return new_x, new_y

    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.gaze = GazeTracking()

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        # Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)

        self.file_name = rf"point_data\{datetime.now().date()}_{datetime.now().time().hour}-{datetime.now().time().minute}.csw"
        self.save_boundary_points(self.file_name, self.points)

        # Обновить координаты калибровочных точек в зависимости от размеров окна
        Window.bind(on_resize=self.update_points)

    def update_points(self, instance, width, height):
        self.points = [(30, 30), (width - 30, 30), (30, height - 30), (width - 30, height - 30)]


    def on_click(self):
        if self.is_calibration:
            p = self.w[0]
            t1 = self.gaze.horizontal_ratio()
            t2 = self.gaze.vertical_ratio()
            if t1 is None or t2 is None:
                return
            self.w.pop(0)
            self.h[p].append((t1, t2))
            print(p, self.gaze.horizontal_ratio(), self.gaze.vertical_ratio())

            if len(self.w) == 0:
                for i in range(4):
                    t3 = 0
                    t4 = 0
                    for l1, l2 in self.h[i]:
                        t3 += l1
                        t4 += l2
                    self.h1.append((t3 / len(self.h[i]), t4 / len(self.h[i])))
                self.d = []
                self.is_calibration = False


    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        self.on_click()


    def _on_touch_down(self):
        self.on_click()


    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def calibration(self, dt):
        self.draw_calibration_point(self.frame, self.points[self.w[0]])

    def main_action(self, dt):
        pi0 = (self.gaze.horizontal_ratio(), self.gaze.vertical_ratio())
        if pi0[0] is not None and pi0[1] is not None:
            self.d.append(pi0)
            if len(self.d) > 8:
                self.d.pop(0)
            pi = (sum(map(lambda x: x[0], self.d)) / len(self.d), sum(map(lambda x: x[1], self.d)) / len(self.d))
            x0, y0 = self.transform_point(self.h1, self.points, pi)
            self.save_point_to_csv(x0, y0, self.file_name)

            self.draw_calibration_point(self.frame, (int(x0), int(y0)), 25)

        # left_pupil = self.gaze.pupil_left_coords()
        # right_pupil = self.gaze.pupil_right_coords()
        # cv2.putText(self.frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(self.frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    def update(self, dt):
        _, self.frame = self.capture.read()
        self.gaze.refresh(self.frame)

        self.frame = self.gaze.annotated_frame()
        self.frame = cv2.resize(self.frame, (Window.width, Window.height))

        if self.is_calibration:
            self.calibration(dt)
        else:
            self.main_action(dt)

        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture

if __name__ == '__main__':
    MyCameraApp().run()
