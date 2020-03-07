# coding: utf-8
#
import numpy as np
import json
from socket import *
import select
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Line, Rectangle, Color
from kivy.clock import Clock
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty,
)
from kivy.vector import Vector
from kivy.core.window import Window
import math

HOST = gethostname()
PORT = 1113
BUFSIZE = 1024
ADDR = ("127.0.0.1", PORT)
USER = "Server"
INTERVAL = 0.01
VEROCITY = 100
LIFETIME = 1000
# Window.fullscreen=True


class SoundSourceTimeline(Widget):
    x = NumericProperty(0)
    y = NumericProperty(0)
    color = ListProperty([1, 1, 1, 1])

    def build(self):
        self.pos = Vector(self.x, self.y)
        self.color = [1, 1, 1, 1]

    def update(self, dt):
        self.pos = Vector(self.x + dt * VEROCITY, self.y)


class ColorSelector(BoxLayout):
    picker = ObjectProperty(None)
    pass


class SourcesArea(RelativeLayout):
    pass


class SourceButton(Button):
    marker_color = ListProperty([1, 1, 1, 1])
    label_index = 0

    def __init__(self, **kwargs):
        super(SourceButton, self).__init__(**kwargs)


class SourceWidget(BoxLayout):
    sources_area = ObjectProperty(None)
    label_area = ObjectProperty(None)
    sources = ListProperty()
    label_buttons = ListProperty()
    clr_selector = None
    selected_btn = None
    label_colors = []
    label_texts = []

    def start(self):
        Clock.schedule_interval(self.callback, INTERVAL)
        self.udpServSock = socket(AF_INET, SOCK_DGRAM)  # IPv4/UDP
        self.udpServSock.bind(ADDR)
        self.udpServSock.setblocking(0)
        self.init()

    def on_color(self, instance, value):
        print(value)

    def on_color_ok(self, e):
        c = self.clr_selector.picker.color
        idx = self.selected_btn.label_index
        self.label_colors[idx] = Color(*c)
        self.selected_btn.marker_color = c
        self.remove_widget(self.clr_selector)
        self.clr_selector = None
        self.selected_btn = None
        print(type(c))

    def on_press(self, e):
        if self.clr_selector is None:
            e.marker_color = [1, 1, 1, 1]
            self.selected_btn = e
            self.clr_selector = ColorSelector()
            # self.clr_selector.picker.bind(color=self.on_color)
            self.clr_selector.button_ok.bind(on_press=self.on_color_ok)
            self.add_widget(self.clr_selector)

    def init(self):
        enabled_labels = [0, 1, 2]
        labels = {"0": "aaa", "1": "bbb", "2": "ccc"}
        h = 0
        print(self.label_area)
        for index, k in enumerate(enabled_labels):
            label = labels[str(k)]
            c = Color(h, 1, 1, mode="hsv")
            self.label_colors.append(c)
            self.label_texts.append(label)
            h += 0.3
            b = SourceButton(text=label)
            b.marker_color = c.rgba
            b.label_index = index
            b.bind(on_press=self.on_press)
            self.label_buttons.append(b)
            print(self.label_area)
            self.label_area.add_widget(b)

    def callback(self, dt):
        data = None
        draw_data = {}
        try:
            while True:
                data, addr = self.udpServSock.recvfrom(BUFSIZE)
                if data is not None and len(data) > 0:
                    print("...received from and returned to:", addr)
                    print(len(data))
                    while len(data) >= 6:
                        idx = int.from_bytes(data[0:1], byteorder="little")
                        l = int.from_bytes(data[1:2], byteorder="little")
                        arr = np.frombuffer(data[2:6], dtype=np.float32)
                        data = data[6:]
                        # l=arr[0]
                        y = arr[0]
                        draw_data[l] = y
                        print(idx, l, y)
        except:
            pass
        if len(draw_data) > 0:
            for l, y in draw_data.items():
                ss = SoundSourceTimeline(y=int(y))
                # set label color
                label = int(l)
                if label < len(self.label_colors):
                    c = self.label_colors[label]
                    ss.color = c.rgba
                #
                self.sources.append(ss)
                self.sources_area.add_widget(ss)
        for s in self.sources:
            s.update(dt)
            if s.pos[0] > self.size[0]:
                self.sources_area.remove_widget(s)
                self.sources.remove(s)

    def stop(self):
        self.udpServSock.close()


class SourceApp(App):
    def build(self):
        widget = SourceWidget()
        widget.start()
        return widget


SourceApp().run()
