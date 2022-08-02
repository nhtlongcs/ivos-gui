"""
Based on https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN 
(which is based on https://github.com/seoungwugoh/ivs-demo)

This version is much simplified. 
In this repo, we don't have
- local control
- fusion module
- undo
- timers

but with XMem as the backbone and is more memory (for both CPU and GPU) friendly
"""

import functools

import os
import cv2

# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import numpy as np
import torch

from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QComboBox,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QSpinBox,
    QFileDialog,
    QPlainTextEdit,
    QVBoxLayout,
    QSizePolicy,
    QButtonGroup,
    QSlider,
    QShortcut,
    QRadioButton,
)

from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor, QIcon
from PyQt5.QtCore import Qt, QTimer

from model.network import XMem

from inference.inference_core import InferenceCore
from .s2m_controller import S2MController
from .reference_controller import ReferenceController

from .interactive_utils import *
from .interaction import *
from .resource_manager import ResourceManager
from .gui_utils import *
from .classnames import FLARE22_CLASSNAMES

class App(QWidget):
    def __init__(
        self,
        processor: InferenceCore,
        resource_manager: ResourceManager,
        s2m_ctrl: S2MController,
        refer_ctrl: ReferenceController,
        config,
    ):
        super().__init__()

        self.initialized = False
        self.num_objects = config["num_objects"]
        self.s2m_controller = s2m_ctrl
        self.refer_controller = refer_ctrl
        self.config = config
        self.processor = processor
        self.processor.set_all_labels(all_labels=list(range(1, self.num_objects + 1)))
        self.res_man = resource_manager

        self.num_frames = len(self.res_man)
        self.height, self.width = self.res_man.h, self.res_man.w

        # set window
        self.setWindowTitle("Mask Propagation Demo")
        self.setGeometry(100, 100, self.width, self.height + 100)
        self.setWindowIcon(QIcon("docs/icon.png"))

        # some buttons
        self.play_button = QPushButton("Play Video")
        self.play_button.clicked.connect(self.on_play_video)
        self.commit_button = QPushButton("Commit")
        self.commit_button.clicked.connect(self.on_commit)

        self.forward_run_button = QPushButton("Forward Propagate")
        self.forward_run_button.clicked.connect(self.on_forward_propagation)
        self.forward_run_button.setMinimumWidth(200)

        self.backward_run_button = QPushButton("Backward Propagate")
        self.backward_run_button.clicked.connect(self.on_backward_propagation)
        self.backward_run_button.setMinimumWidth(200)

        self.reset_button = QPushButton("Reset Frame")
        self.reset_button.clicked.connect(self.on_reset_mask)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(120)
        self.lcd.setText("{: 4d} / {: 4d}".format(0, self.num_frames - 1))


        # Classnames
        self.classnames_lcd = QTextEdit()
        self.classnames_lcd.setReadOnly(True)
        self.classnames_lcd.setMaximumHeight(28)
        self.classnames_lcd.setMaximumWidth(240)
        self.classnames_lcd.setText("Background")

        # timeline slider
        self.tl_slider = QSlider(Qt.Horizontal)
        self.tl_slider.valueChanged.connect(self.tl_slide)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)

        # brush size slider
        self.brush_label = QLabel()
        self.brush_label.setAlignment(Qt.AlignCenter)
        self.brush_label.setMinimumWidth(100)

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.valueChanged.connect(self.brush_slide)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(3)
        self.brush_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_slider.setTickInterval(2)
        self.brush_slider.setMinimumWidth(300)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        self.combo.addItem("layered")
        self.combo.currentTextChanged.connect(self.set_viz_mode)

        self.save_visualization_checkbox = QCheckBox(self)
        self.save_visualization_checkbox.toggled.connect(
            self.on_save_visualization_toggle
        )
        self.save_visualization_checkbox.setChecked(False)
        self.save_visualization = False

        # Radio buttons for type of interactions
        self.curr_interaction = "Scribble"
        self.interaction_group = QButtonGroup()
        self.radio_refer = QRadioButton("Refer")
        self.radio_s2m = QRadioButton("Scribble")
        self.radio_free = QRadioButton("Free")
        self.interaction_group.addButton(self.radio_refer)
        self.interaction_group.addButton(self.radio_s2m)
        self.interaction_group.addButton(self.radio_free)
        self.radio_refer.toggled.connect(self.interaction_radio_clicked)
        self.radio_s2m.toggled.connect(self.interaction_radio_clicked)
        self.radio_free.toggled.connect(self.interaction_radio_clicked)
        self.radio_s2m.toggle()

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release

        # Legend canvas
        class Legends(QWidget):
            def __init__(self, image_path='docs/legends.png'):
                super().__init__()
                self.setWindowTitle("Color mapping")
                self.legend_canvas = QLabel()
                self.legend_canvas.setAlignment(Qt.AlignCenter)  
                self.legend_pixmap = QPixmap(image_path)

                self.legend_canvas.setPixmap(
                    self.legend_pixmap
                )

                layout = QVBoxLayout()
                layout.addWidget(self.legend_canvas)
                self.setLayout(layout)

            def isVisibleWidget(self):
                if not self.visibleRegion().isEmpty():
                    return True
                return False

            def toggle(self):
                if self.isVisibleWidget():
                    self.close()
                else:
                    self.show()
                

        self.legend_widget = Legends()

        # Multiview canvas
        self.num_views = 3
        self.triple_canvas = QVBoxLayout()
        self.triple_canvas.setAlignment(Qt.AlignRight)  
        self.views_canvas = []
        for i in range(self.num_views):
            self.views_canvas.append(QLabel())
            self.views_canvas[i].setMaximumSize(200,200)
            self.triple_canvas.addWidget(self.views_canvas[i])

        # Minimap -> Also a QLbal
        self.minimap = QLabel()
        self.minimap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.minimap.setAlignment(Qt.AlignTop)
        self.minimap.setMinimumSize(100, 100)

        # Zoom-in buttons
        self.zoom_p_button = QPushButton("Zoom +")
        self.zoom_p_button.clicked.connect(self.on_zoom_plus)
        self.zoom_m_button = QPushButton("Zoom -")
        self.zoom_m_button.clicked.connect(self.on_zoom_minus)

        # Parameters setting
        self.clear_mem_button = QPushButton("Clear memory")
        self.clear_mem_button.clicked.connect(self.on_clear_memory)

        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge(
            "GPU mem. (all processes, w/ caching)"
        )
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge(
            "GPU mem. (used by torch, w/o caching)"
        )

        self.update_gpu_usage()

        self.mem_top_k, self.mem_top_k_layout = create_parameter_box(
            20, 100, "Top k similarity", step=1, callback=self.update_config
        )

        self.mem_max_k, self.mem_max_k_layout = create_parameter_box(
            32, 1280, "Maximum memory stack", step=32, callback=self.update_config
        )

        self.mem_stack_gauge, self.mem_stack_gauge_layout = create_gauge(
            "Memory stack"
        )

        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(
            1, 100, "Memory frame every (r)", callback=self.update_config
        )

        self.include_last_checkbox = QCheckBox(self)
        self.include_last_checkbox.setChecked(True)

        self.mem_top_k.setValue(
            self.processor.get_value("memory.top_k")
        )
        self.mem_max_k.setValue(
            self.processor.get_value("memory.max_k")
        )
        self.mem_stack_gauge.setValue(self.processor.get_value("memory.cur_k"))
        self.mem_every_box.setValue(self.processor.get_value("mem_every"))

        # import mask/layer
        self.import_mask_button = QPushButton("Import mask")
        self.import_mask_button.clicked.connect(self.on_import_mask)
        self.import_layer_button = QPushButton("Import layer")
        self.import_layer_button.clicked.connect(self.on_import_layer)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # navigator
        navi = QHBoxLayout()
        navi.addWidget(self.lcd)
        navi.addWidget(self.play_button)

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignCenter)
        interact_topbox.addWidget(self.radio_refer)
        interact_topbox.addWidget(self.radio_s2m)
        interact_topbox.addWidget(self.radio_free)
        interact_topbox.addWidget(self.brush_label)
        interact_botbox.addWidget(self.brush_slider)
        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        navi.addLayout(interact_subbox)

        navi.addStretch(1)
        navi.addWidget(self.reset_button)

        navi.addStretch(1)
        navi.addWidget(QLabel("Overlay Mode"))
        navi.addWidget(self.combo)
        navi.addWidget(QLabel("Save overlay during propagation"))
        navi.addWidget(self.save_visualization_checkbox)
        navi.addStretch(1)
        navi.addWidget(self.commit_button)
        navi.addWidget(self.forward_run_button)
        navi.addWidget(self.backward_run_button)

        # Drawing area, main canvas and minimap
        draw_area = QHBoxLayout()
        draw_area.addLayout(self.triple_canvas, 1)
        draw_area.addWidget(self.main_canvas, 4)

        # Minimap area
        minimap_area = QVBoxLayout()
        minimap_area.setAlignment(Qt.AlignTop)
        mini_label = QLabel("Minimap")
        mini_label.setAlignment(Qt.AlignTop)
        minimap_area.addWidget(mini_label)
        minimap_area.addWidget(self.classnames_lcd)

        # Minimap zooming
        minimap_ctrl = QHBoxLayout()
        minimap_ctrl.setAlignment(Qt.AlignTop)
        minimap_ctrl.addWidget(self.zoom_p_button)
        minimap_ctrl.addWidget(self.zoom_m_button)
        minimap_area.addLayout(minimap_ctrl)
        minimap_area.addWidget(self.minimap)

        # Parameters
        minimap_area.addLayout(self.gpu_mem_gauge_layout)
        minimap_area.addLayout(self.torch_mem_gauge_layout)
        minimap_area.addWidget(self.clear_mem_button)
        minimap_area.addLayout(self.mem_top_k_layout)
        minimap_area.addLayout(self.mem_max_k_layout)
        minimap_area.addLayout(self.mem_stack_gauge_layout)
        minimap_area.addLayout(self.mem_every_box_layout)

        # include last
        include_last_area = QHBoxLayout()
        include_last_area.addWidget(QLabel("Use last frame as guidance"))
        include_last_area.addWidget(self.include_last_checkbox)
        minimap_area.addLayout(include_last_area)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignTop)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        minimap_area.addLayout(import_area)

        # console
        minimap_area.addWidget(self.console)

        draw_area.addLayout(minimap_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)

        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(self.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # current frame info
        self.curr_frame_dirty = False
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.current_image_torch = None
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_prob = torch.zeros(
            (self.num_objects, self.height, self.width), dtype=torch.float
        ).cuda()

        # initialize visualization
        self.viz_mode = "davis"
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.cursur = 0
        self.on_showing = None

        # Zoom parameters
        self.zoom_pixels = 150

        # initialize action
        self.interaction = None
        self.pressed = False
        self.right_click = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0

        self.propagating = False

        # Objects shortcuts
        for i in range(1, self.num_objects + 1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(
                functools.partial(self.hit_number_key, i)
            )

        # <- and -> shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.on_prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(
            self.on_next_frame
        )

        self.interacted_prob = None
        self.overlay_layer = None
        self.overlay_layer_torch = None

        self.load_current_image_mask()
        self.show_current_frame()
        self.show()

        self.console_push_text("Initialized.")
        self.initialized = True

        # try to load the default overlay
        self._try_load_layer("./docs/ECCV-logo.png")

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Q:
            print("Killing")
            self.parent().close()
        elif event.key() == Qt.Key_L:
            self.legend_widget.toggle()
        event.accept()

    def resizeEvent(self, event):
        self.show_current_frame()

    def console_push_text(self, text):
        self.console.appendPlainText(text)
        self.console.moveCursor(QTextCursor.End)
        print(text)

    def interaction_radio_clicked(self, event):
        self.last_interaction = self.curr_interaction
        if self.radio_s2m.isChecked():
            self.curr_interaction = "Scribble"
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_refer.isChecked():
            self.curr_interaction = "Refer"
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_free.isChecked():
            self.brush_slider.setDisabled(False)
            self.brush_slide()
            self.curr_interaction = "Free"
        if self.curr_interaction == "Scribble":
            self.commit_button.setEnabled(True)
        else:
            self.commit_button.setEnabled(False)

    def load_current_image_mask(self, no_mask=False):
        self.current_image = self.res_man.get_image(self.cursur)
        self.current_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.cursur)
            if loaded_mask is None:
                self.current_mask.fill(0)
            else:
                self.current_mask = loaded_mask.copy()
            self.current_prob = None

    def load_current_torch_image_mask(self, no_mask=False):
        if self.current_image_torch is None:
            self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(
                self.current_image
            )

        if self.current_prob is None and not no_mask:
            self.current_prob = index_numpy_to_one_hot_torch(
                self.current_mask, self.num_objects + 1
            ).cuda()

    def compose_current_im(self):
        self.viz = get_visualization(
            self.viz_mode, self.current_image, self.current_mask, self.overlay_layer
        )

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        vis_map = self.vis_map
        vis_alpha = self.vis_alpha
        brush_vis_map = self.brush_vis_map
        brush_vis_alpha = self.brush_vis_alpha

        self.viz_with_stroke = self.viz * (1 - vis_alpha) + vis_map * vis_alpha
        self.viz_with_stroke = (
            self.viz_with_stroke * (1 - brush_vis_alpha)
            + brush_vis_map * brush_vis_alpha
        )
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(
            self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.main_canvas.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
        )

        # update multiview
        for i in range(self.num_views):
            view_viz = cv2.cvtColor(self.viz[...,i], cv2.COLOR_GRAY2RGB)
            view_viz = get_visualization(
                self.viz_mode, view_viz, self.current_mask, self.overlay_layer
            )
            view_viz = view_viz.astype(np.uint8)

            qViewImg = QImage(
                view_viz.data, width, height, bytesPerLine, QImage.Format_RGB888
            )

            self.views_canvas[i].setPixmap(
                QPixmap(
                    qViewImg.scaled(
                        self.views_canvas[i].size(), Qt.KeepAspectRatio, Qt.FastTransformation
                    )
                )
            )

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_minimap(self):
        ex, ey = self.last_ex, self.last_ey
        r = self.zoom_pixels // 2
        ex = int(round(max(r, min(self.width - r, ex))))
        ey = int(round(max(r, min(self.height - r, ey))))

        patch = self.viz_with_stroke[ey - r : ey + r, ex - r : ex + r, :].astype(
            np.uint8
        )

        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.minimap.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.minimap.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
        )

    def update_current_image_fast(self):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        self.viz = get_visualization_torch(
            self.viz_mode,
            self.current_image_torch_no_norm,
            self.current_prob,
            self.overlay_layer_torch,
        )
        if self.save_visualization:
            self.res_man.save_visualization(self.cursur, self.viz)

        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        qImg = QImage(self.viz.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.main_canvas.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
        )

    def show_current_frame(self, fast=False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast()
        else:
            self.compose_current_im()
            self.update_interact_vis()
            self.update_minimap()

        self.lcd.setText("{: 3d} / {: 3d}".format(self.cursur, self.num_frames - 1))
        self.tl_slider.setValue(self.cursur)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = (
            (x < 0) or (y < 0) or (x > self.width - 1) or (y > self.height - 1)
        )

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        return x, y

    def clear_visualization(self):
        self.vis_map.fill(0)
        self.vis_alpha.fill(0)

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        self.interaction = None

    def set_viz_mode(self):
        self.viz_mode = self.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.cursur, self.current_mask)

    def tl_slide(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        if not self.propagating:
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.cursur = self.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def brush_slide(self):
        self.brush_size = self.brush_slider.value()
        self.brush_label.setText("Brush size: %d" % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            # Initialization, forget about it
            pass

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_next_frame
            self.backward_run_button.setEnabled(False)
            self.forward_run_button.setText("Pause Propagation")
            self.on_propagation()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_prev_frame
            self.forward_run_button.setEnabled(False)
            self.backward_run_button.setText("Pause Propagation")
            self.on_propagation()

    def on_pause(self):
        self.propagating = False
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_mem_button.setEnabled(True)
        self.forward_run_button.setText("Forward Propagate")
        self.backward_run_button.setText("Backward Propagate")
        self.console_push_text("Propagation stopped.")

    def on_propagation(self):
        # start to propagate
        self.load_current_torch_image_mask()
        self.show_current_frame(fast=True)

        self.console_push_text("Propagation started.")
        self.current_prob = self.processor.step(
            image=self.current_image_torch, mask=self.current_prob[1:]
        )

        if isinstance(self.current_prob, str):
            current_prob_np = pickle.loads(self.current_prob.encode("latin-1"))
            self.current_prob = torch.from_numpy(current_prob_np)

        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
        # clear
        self.interacted_prob = None
        self.reset_this_interaction()

        self.propagating = True
        self.clear_mem_button.setEnabled(False)
        # propagate till the end
        while self.propagating:
            self.propagate_fn()

            self.load_current_image_mask(no_mask=True)
            self.load_current_torch_image_mask(no_mask=True)

            self.current_prob = self.processor.step(image=self.current_image_torch)

            if isinstance(self.current_prob, str):
                current_prob_np = pickle.loads(self.current_prob.encode("latin-1"))
                self.current_prob = torch.from_numpy(current_prob_np)

            self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

            self.save_current_mask()
            self.show_current_frame(fast=True)

            self.update_gpu_usage()
            self.update_memory_stack()
            QApplication.processEvents()

            if self.cursur == 0 or self.cursur == self.num_frames - 1:
                break

        self.propagating = False
        self.curr_frame_dirty = False
        self.on_pause()
        self.tl_slide()
        QApplication.processEvents()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        self.complete_interaction()
        self.update_interacted_mask()

    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur - 1)
        self.tl_slider.setValue(self.cursur)

    def on_next_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur + 1, self.num_frames - 1)
        self.tl_slider.setValue(self.cursur)

    def on_play_video_timer(self):
        self.cursur += 1
        if self.cursur > self.num_frames - 1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)

    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play Video")
        else:
            self.timer.start(1000 / 30)
            self.play_button.setText("Stop Video")

    def on_reset_mask(self):
        self.current_mask.fill(0)
        if self.current_prob is not None:
            self.current_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_zoom_plus(self):
        self.zoom_pixels -= 25
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 25
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap()

    def set_navi_enable(self, boolean):
        self.zoom_p_button.setEnabled(boolean)
        self.zoom_m_button.setEnabled(boolean)
        self.run_button.setEnabled(boolean)
        self.tl_slider.setEnabled(boolean)
        self.play_button.setEnabled(boolean)
        self.lcd.setEnabled(boolean)

    def hit_number_key(self, number):
        if number == self.current_object:
            return
        self.current_object = number
        self.console_push_text(f"Current object changed to {number}.")
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.show_current_frame()

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        self.brush_vis_map = cv2.circle(
            self.brush_vis_map,
            (int(round(ex)), int(round(ey))),
            self.brush_size // 2 + 1,
            color_map[self.current_object],
            thickness=-1,
        )
        self.brush_vis_alpha = cv2.circle(
            self.brush_vis_alpha,
            (int(round(ex)), int(round(ey))),
            self.brush_size // 2 + 1,
            0.5,
            thickness=-1,
        )

    def on_mouse_press(self, event):
        if self.is_pos_out_of_bound(event.x(), event.y()):
            return

        self.pressed = True
        self.right_click = event.button() != 1

        h, w = self.height, self.width

        self.load_current_torch_image_mask()
        image = self.current_image_torch

        last_interaction = self.interaction
        new_interaction = None
        if self.curr_interaction == "Scribble":
            if (
                last_interaction is None
                or type(last_interaction) != ScribbleInteraction
            ):
                self.complete_interaction()
                new_interaction = ScribbleInteraction(
                    image,
                    torch.from_numpy(self.current_mask).float().cuda(),
                    (h, w),
                    self.s2m_controller,
                    self.num_objects,
                )
        elif self.curr_interaction == "Free":
            if last_interaction is None or type(last_interaction) != FreeInteraction:
                self.complete_interaction()
                new_interaction = FreeInteraction(
                    image, self.current_mask, (h, w), self.num_objects
                )
                new_interaction.set_size(self.brush_size)
        elif self.curr_interaction == "Refer":
            if (
                last_interaction is None
                or type(last_interaction) != ReferenceInteraction
            ):
                self.complete_interaction()
                new_interaction = ReferenceInteraction(
                    image,
                    self.current_prob,
                    (h, w),
                    self.refer_controller,
                )

        if new_interaction is not None:
            self.interaction = new_interaction

        # Just motion it as the first step
        self.on_mouse_motion(event)

    def on_mouse_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)
        if self.pressed:
            if self.curr_interaction == "Scribble" or self.curr_interaction == "Free":
                obj = 0 if self.right_click else self.current_object
                self.vis_map, self.vis_alpha = self.interaction.push_point(
                    ex, ey, obj, (self.vis_map, self.vis_alpha)
                )

        classname = FLARE22_CLASSNAMES[int(self.current_mask[int(ey), int(ex)])]
        self.classnames_lcd.setText(classname)

        self.update_interact_vis()
        self.update_minimap()

    def update_interacted_mask(self):
        self.current_prob = self.interacted_prob
        self.current_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.show_current_frame()
        self.save_current_mask()
        self.curr_frame_dirty = False

    def complete_interaction(self):
        if self.interaction is not None:
            self.clear_visualization()
            self.interaction = None

    def on_mouse_release(self, event):
        if not self.pressed:
            # this can happen when the initial press is out-of-bound
            return

        ex, ey = self.get_scaled_pos(event.x(), event.y())

        self.console_push_text(
            "%s interaction at frame %d." % (self.curr_interaction, self.cursur)
        )
        interaction = self.interaction

        if self.curr_interaction == "Scribble" or self.curr_interaction == "Free":
            self.on_mouse_motion(event)
            interaction.end_path()
            if self.curr_interaction == "Free":
                self.clear_visualization()
        elif self.curr_interaction == "Refer":
            interaction.push_point(
                self.cursur / self.num_frames
            )
            

        self.interacted_prob = interaction.predict()
        self.update_interacted_mask()
        self.update_gpu_usage()
        self.update_memory_stack()


        self.pressed = self.right_click = False

    def wheelEvent(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        if self.curr_interaction == "Free":
            self.brush_slider.setValue(
                self.brush_slider.value() + event.angleDelta().y() // 30
            )
        self.clear_brush()
        self.vis_brush(ex, ey)
        self.update_interact_vis()
        self.update_minimap()

    def update_gpu_usage(self):
        info = torch.cuda.mem_get_info()
        global_free, global_total = info
        global_free /= 2 ** 30
        global_total /= 2 ** 30
        global_used = global_total - global_free

        self.gpu_mem_gauge.setFormat(f"{global_used:.01f} GB / {global_total:.01f} GB")
        self.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

        used_by_torch = torch.cuda.max_memory_allocated() / (2 ** 20)
        self.torch_mem_gauge.setFormat(
            f"{used_by_torch:.0f} MB / {global_total:.01f} GB"
        )
        self.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100 / 1024))

    def on_gpu_timer(self):
        self.update_gpu_usage()

    # def update_memory_size(self):
    #     try:
    #         max_work_elements = self.processor.get_value("memory.max_work_elements")
    #         max_long_elements = self.processor.get_value("memory.max_long_elements")

    #         curr_work_elements = self.processor.get_value("memory.work_mem.size")
    #         curr_long_elements = self.processor.get_value("memory.long_mem.size")

            # self.work_mem_gauge.setFormat(f"{curr_work_elements} / {max_work_elements}")
            # self.work_mem_gauge.setValue(
            #     round(curr_work_elements / max_work_elements * 100)
            # )

            # self.long_mem_gauge.setFormat(f"{curr_long_elements} / {max_long_elements}")
            # self.long_mem_gauge.setValue(
            #     round(curr_long_elements / max_long_elements * 100)
            # )

        # except AttributeError:
        #     self.work_mem_gauge.setFormat("Unknown")
        #     self.long_mem_gauge.setFormat("Unknown")
        #     self.work_mem_gauge.setValue(0)
        #     self.long_mem_gauge.setValue(0)

    def update_memory_stack(self):
        try:
            current_stack = self.processor.get_value("memory.cur_k")
            max_stack = self.processor.get_value("memory.max_k")
            self.mem_stack_gauge.setFormat(f"{current_stack} / {max_stack}")
            self.mem_stack_gauge.setValue(round(current_stack / max_stack * 100))

        except AttributeError:
            self.mem_stack_gauge.setFormat("Unknown")
            self.mem_stack_gauge.setValue(0)

    def update_config(self):
        if self.initialized:
            self.config["top_k"] = self.mem_top_k.value()
            self.config["max_k"] = self.mem_max_k.value()
            self.config["mem_every"] = self.mem_every_box.value()
            self.config["include_last"] = self.include_last_checkbox.isChecked()

            self.processor.update_config(config=self.config)

    def on_clear_memory(self):
        self.processor.clear_memory()
        torch.cuda.empty_cache()
        self.update_gpu_usage()
        self.update_memory_stack()

    def _open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, prompt, "", "PNG files (*.png)", options=options
        )
        return file_name

    def on_import_mask(self):
        file_name = self._open_file("Mask")
        if len(file_name) == 0:
            return

        mask = self.res_man.read_external_image(file_name)

        condition = (
            (len(mask.shape) == 2)
            and (mask.shape[-1] == self.width)
            and (mask.shape[-2] == self.height)
        )

        if not condition:
            self.console_push_text(
                f"Expected ({self.height}, {self.width}). Got {mask.shape}."
            )
        else:
            self.console_push_text(f"Mask file {file_name} loaded.")
            self.current_image_torch = self.current_prob = None
            self.current_mask = mask
            self.show_current_frame()

    def on_import_layer(self):
        file_name = self._open_file("Layer")
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.read_external_image(
                file_name, size=(self.height, self.width)
            )

            condition = (
                (len(layer.shape) == 3)
                and (layer.shape[-1] == 4)
                and (layer.shape[-2] == self.width)
                and (layer.shape[-3] == self.height)
            )

            if not condition:
                self.console_push_text(
                    f"Expected ({self.height}, {self.width}, 4). Got {layer.shape}."
                )
            else:
                self.console_push_text(f"Layer file {file_name} loaded.")
                self.overlay_layer = layer
                self.overlay_layer_torch = torch.from_numpy(layer).float().cuda() / 255
                self.show_current_frame()
        except FileNotFoundError:
            self.console_push_text(f"{file_name} not found.")

    def on_save_visualization_toggle(self):
        self.save_visualization = self.save_visualization_checkbox.isChecked()

    
