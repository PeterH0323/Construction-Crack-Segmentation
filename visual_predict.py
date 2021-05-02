# -*- coding: utf-8 -*-
# @Time    : 2021/5/1 8:41
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : visual_predict.py
# @Software: PyCharm
# @Brief   : 


# 解决 exe 打包 Can't get source for 的问题 start ======
# https://github.com/pytorch/vision/issues/1899#issuecomment-598200938
import math

import cv2
import numpy as np
import torch.jit
from torch.backends import cudnn

from builders.dataset_builder import build_dataset_predict
from builders.model_builder import build_model
from dataset.custom import img_formats
from utils.utils import save_predict


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


torch.jit.script_method = script_method
torch.jit.script = script
# 解决 exe 打包 Can't get source for 的问题 end ======

import os
import time
import sys
from pathlib import Path
from GPUtil import GPUtil
from PyQt5.QtCore import QThread, pyqtSignal, QUrl, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap, QImage
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart
import torch
from UI.main_window import Ui_MainWindow

# from detect_visual import YOLOPredict
# from utils.datasets import img_formats

CODE_VER = "V1.0.1"
PREDICT_SHOW_TAB_INDEX = 0
REAL_TIME_PREDICT_TAB_INDEX = 1


def get_gpu_info():
    """
    获取 GPU 信息
    :return:
    """

    gpu_list = []
    # GPUtil.showUtilization()

    # 获取多个GPU的信息，存在列表里
    for gpu in GPUtil.getGPUs():
        # print('gpu.id:', gpu.id)
        # print('GPU总量：', gpu.memoryTotal)
        # print('GPU使用量：', gpu.memoryUsed)
        # print('gpu使用占比:', gpu.memoryUtil * 100)  # 内存使用率
        # print('gpu load:', gpu.load * 100)  # 使用率
        # 按GPU逐个添加信息
        gpu_list.append({"gpu_id": gpu.id,
                         "gpu_memoryTotal": gpu.memoryTotal,
                         "gpu_memoryUsed": gpu.memoryUsed,
                         "gpu_memoryUtil": gpu.memoryUtil * 100,
                         "gpu_load": gpu.load * 100})

    return gpu_list


class PredictDataHandlerThread(QThread):
    """
    打印信息的线程
    """
    predict_message_trigger = pyqtSignal(str)

    def __init__(self, predict_model):
        super(PredictDataHandlerThread, self).__init__()
        self.running = False
        self.predict_model = predict_model

    def __del__(self):
        self.running = False
        # self.destroyed()

    def run(self):
        self.running = True
        over_time = 0
        while self.running:
            if self.predict_model.predict_info != "":
                self.predict_message_trigger.emit(self.predict_model.predict_info)
                self.predict_model.predict_info = ""
                over_time = 0
            time.sleep(0.01)
            over_time += 1

            if over_time > 100000:
                self.running = False


class PredictHandlerThread(QThread):
    """
    进行模型推理的线程
    """

    def __init__(self, input_player, output_player, out_file_path, weight_path,
                 predict_info_plain_text_edit, predict_progress_bar, fps_label,
                 button_dict, input_tab, output_tab, input_image_label, output_image_label,
                 output_mask_player, output_mask_tab, output_mask_real_time_label,
                 real_time_show_predict_flag):
        super(PredictHandlerThread, self).__init__()
        self.running = False

        '''加载模型'''
        net_type = Path(weight_path).stem.split("_")[0]
        self.predict_model = SegmentationModel(net_type, weight_path, out_file_path,
                                               dataset_type="custom_dataset", gpu_number="0", class_number=2)
        self.output_predict_file = ""
        self.output_predict_mask_file = ""
        self.parameter_source = ''

        # 传入的QT插件
        self.input_player = input_player
        self.output_player = output_player
        self.output_mask_player = output_mask_player
        self.predict_info_plainTextEdit = predict_info_plain_text_edit
        self.predict_progressBar = predict_progress_bar
        self.fps_label = fps_label
        self.button_dict = button_dict
        self.input_tab = input_tab
        self.output_tab = output_tab
        self.output_mask_tab = output_mask_tab
        self.input_image_label = input_image_label
        self.output_image_label = output_image_label
        self.output_mask_real_time_label = output_mask_real_time_label

        # 是否实时显示推理图片
        self.real_time_show_predict_flag = real_time_show_predict_flag

        # 创建显示进程
        self.predict_data_handler_thread = PredictDataHandlerThread(self.predict_model)
        self.predict_data_handler_thread.predict_message_trigger.connect(self.add_messages)

    def __del__(self):
        self.running = False
        # self.destroyed()

    def run(self):
        self.predict_data_handler_thread.start()

        self.predict_progressBar.setValue(0)  # 进度条清零
        for item, button in self.button_dict.items():
            button.setEnabled(False)

        image_flag = os.path.splitext(self.parameter_source)[-1].lower() in img_formats
        qt_input = None
        qt_output = None
        mask_qt_output = None

        if not image_flag and self.real_time_show_predict_flag:
            qt_input = self.input_image_label
            qt_output = self.output_image_label
            mask_qt_output = self.output_mask_real_time_label
            # tab 设置显示第二栏
            self.input_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)
            self.output_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)
            self.output_mask_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)

        with torch.no_grad():
            self.output_predict_file, self.output_predict_mask_file = self.predict_model.detect(self.parameter_source,
                                                                                                qt_input=qt_input,
                                                                                                qt_output=qt_output,
                                                                                                qt_mask_output=mask_qt_output)

        if self.output_predict_file != "" and self.output_predict_mask_file != "":
            # 将 str 路径转为 QUrl 并显示
            self.input_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.parameter_source)))  # 选取视频文件
            self.input_player.pause()  # 显示媒体

            # 注意 PNG 不能使用 setMedia 显示！！！！
            self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_predict_file)))  # 选取视频文件
            self.output_player.pause()  # 显示媒体

            self.output_mask_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_predict_mask_file)))  # 选取视频文件
            self.output_mask_player.pause()  # 显示媒体

            # tab 设置显示第一栏
            self.input_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
            self.output_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
            self.output_mask_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)

            # video_flag = os.path.splitext(self.parameter_source)[-1].lower() in vid_formats
            for item, button in self.button_dict.items():
                if image_flag and item in ['play_pushButton', 'pause_pushButton']:
                    continue
                button.setEnabled(True)
        # self.predict_data_handler_thread.running = False

    @pyqtSlot(str)
    def add_messages(self, message):
        if message != "":
            self.predict_info_plainTextEdit.appendPlainText(message)

            if ":" not in message:
                # 跳过无用字符
                return

            split_message = message.split(" ")

            # 设置进度条
            if "video" in message:
                percent = split_message[2][1:-1].split("/")  # 提取图片的序号
                value = int((int(percent[0]) / int(percent[1])) * 100)
                value = value if (int(percent[1]) - int(percent[0])) > 2 else 100
                self.predict_progressBar.setValue(value)
            else:
                self.predict_progressBar.setValue(100)

            # 设置 FPS
            second_count = 1 / float(split_message[-1][1:-2])
            self.fps_label.setText(f"--> {second_count:.1f} FPS")


class SegmentationModel(object):

    def __init__(self, net_type, model_path, save_seg_dir, dataset_type="custom_dataset", num_workers=2,
                 batch_size=1, gpu_number="0", class_number=2):
        """
        初始化参数
        :param net_type: 网络类型
        :param model_path: 模型路径
        :param save_seg_dir: 结果保存路径
        :param dataset_type: 数据集类型
        :param num_workers: 使用的线程数
        :param batch_size: 推理的 batch size
        :param gpu_number: 使用的 GPU
        :param class_number: 类别数量
        """
        self.model = None  # 模型实例
        self.net_type = net_type  # 网络类型
        self.dataset_type = dataset_type  # 数据集的类型
        # self.image_input_path = None  # 输入图片的路径
        self.num_workers = num_workers  # 使用的线程数
        self.use_txt_list = False  # 是否使用 txt
        self.batch_size = batch_size  # 推理的 batch size
        self.model_path = model_path  # 模型路径
        self.save_seg_dir = save_seg_dir  # 结果保存路径
        self.cuda = True  # 是否使用 cuda
        self.gpu_number = gpu_number  # 使用的 GPU
        self.classes = class_number  # 类别数量

        self.input_windows_width = 0
        self.input_windows_height = 0
        self.output_windows_height = 0

        self.predict_info = ""  # 推理信息

        self.load_model()

    def load_model(self):
        """
        加载权重
        """
        if self.cuda:
            print("=====> use gpu id: '{}'".format(self.gpu_number))
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_number
            if not torch.cuda.is_available():
                raise Exception("no GPU found or wrong gpu id, please run without --cuda")

        # build the model
        self.model = build_model(self.net_type, num_classes=self.classes)

        if self.cuda:
            self.model = self.model.cuda()  # using GPU for inference
            cudnn.benchmark = True

        if os.path.isfile(self.model_path):
            print(f"=====> loading checkpoint '{self.model_path}'")
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{self.model_path}'")
            raise FileNotFoundError(f"no checkpoint found at '{self.model_path}'")

        if not os.path.exists(self.save_seg_dir):
            os.makedirs(self.save_seg_dir)

    def show_real_time_image(self, scale_type, image_label, img):
        """
        image_label 显示实时推理图片
        :param scale_type: 进行缩放的方向
        :param image_label: 本次需要显示的 label 句柄
        :param img: cv2 图片
        :return:
        """
        if image_label is None:
            return

        input_using_height = False
        if scale_type == "output":
            resize_factor = self.input_windows_height / 2 / img.shape[0]
        else:
            if self.input_windows_height == 0:
                self.input_windows_height = image_label.height()

            if self.input_windows_width == 0:
                self.input_windows_width = image_label.width()
            resize_factor = self.input_windows_width / img.shape[1]
            if img.shape[0] * resize_factor > image_label.height():
                resize_factor = self.input_windows_height / img.shape[0]
                input_using_height = True

        img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)),
                         interpolation=cv2.INTER_CUBIC)

        if scale_type == "output" or input_using_height:
            # 使用黑框填充，确保图片显示在框的正中央
            border_with = (image_label.width() - img.shape[1]) // 2
            img = cv2.copyMakeBorder(img, 0, 0, border_with, border_with, cv2.BORDER_CONSTANT)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 读取的bgr格式图片转换成rgb格式
        image = QImage(img_rgb[:],
                       img_rgb.shape[1],
                       img_rgb.shape[0],
                       img_rgb.shape[1] * 3,
                       QImage.Format_RGB888)

        img_show = QPixmap(image)
        image_label.setPixmap(img_show)

    def detect(self, source, qt_input=None, qt_output=None, qt_mask_output=None):
        """
        args:
          test_loader: loaded for test dataset, for those that do not provide label on the test set
          model: model
        return: class IoU and mean IoU
        """

        # load the test set
        _, dataset_loader = build_dataset_predict(source, self.dataset_type, self.num_workers, none_gt=True)

        show_count = 0

        # evaluation or test mode
        self.model.eval()
        total_batches = len(dataset_loader)
        vid_writer = None
        vid_path = None
        vid_mask_writer = None
        vid_mask_path = None

        self.input_windows_width = 0
        self.input_windows_height = 0
        self.output_windows_height = 0

        for i, (input, size, name, mode, frame_count, img_original, vid_cap, info_str) in enumerate(dataset_loader):
            with torch.no_grad():
                input = input[None, ...]  # 增加多一个维度
                input = torch.tensor(input)  # [1, 3, 224, 224]
                input_var = input.cuda()
            start_time = time.time()
            output = self.model(input_var)
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print(f'[{i + 1}/{total_batches}]  time: {time_taken * 1000:.4f} ms = {1 / time_taken:.1f} FPS')
            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            save_name = Path(name).stem + f'_predict'
            if mode == 'images':
                # 保存图片推理结果
                save_predict(output, None, save_name, self.dataset_type, self.save_seg_dir,
                             output_grey=True, output_color=False, gt_color=False)

            # 将结果和原图画到一起
            img = img_original
            mask = output
            mask[mask == 1] = 255  # 将 mask 的 1 变成 255 --> 用于后面显示充当红色通道
            zeros = np.zeros(mask.shape[:2], dtype="uint8")  # 生成 全为0 的矩阵，用于充当 蓝色 和 绿色通道
            mask_final = cv2.merge([zeros, zeros, mask])  # 合并成 3 通道
            img = cv2.addWeighted(img, 1, mask_final, 1, 0)  # 合并

            # 保存推理信息
            image_shape = f'{img_original.shape[0]}x{img_original.shape[1]} '
            self.predict_info = info_str + '%sDone. (%.3fs)' % (image_shape, time_taken)
            print(self.predict_info)
            # QT 显示
            if qt_input is not None and qt_output is not None and dataset_loader.mode == 'video':
                video_count, vid_total = info_str.split(" ")[2][1:-1].split("/")  # 得出当前总帧数
                fps = (time_taken / 1) * 100
                fps_threshold = 25  # FPS 阈值
                show_flag = True
                if fps > fps_threshold:  # 如果 FPS > 阀值，则跳帧处理
                    fps_interval = 15  # 实时显示的帧率
                    show_unit = math.ceil(fps / fps_interval)  # 取出多少帧显示一帧，向上取整
                    if int(video_count) % show_unit != 0:  # 跳帧显示
                        show_flag = False
                    else:
                        show_count += 1

                if show_flag:
                    # 推理前的图片 origin_image, 推理后的图片 im0
                    self.show_real_time_image("input", qt_input, img_original)  # 原图
                    self.show_real_time_image("output", qt_output, img)  # 最终推理图
                    self.show_real_time_image("output", qt_mask_output, mask_final)  # 分割 mask 图

            if mode == 'images':
                # 保存 推理+原图 结果
                save_path = os.path.join(self.save_seg_dir, save_name + '_img.jpg')
                cv2.imwrite(f"{save_path}", img)

                save_mask_path = os.path.join(self.save_seg_dir, save_name + '_mask_img.jpg')
                cv2.imwrite(f"{save_mask_path}", mask_final)
            else:
                # 保存视频
                save_path = os.path.join(self.save_seg_dir, save_name + '_predict.mp4')
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(img)

                # 保存 mask 视频
                save_mask_path = os.path.join(self.save_seg_dir, save_name + '_mask_predict.mp4')
                if vid_mask_path != save_mask_path:  # new video
                    vid_mask_path = save_mask_path
                    if isinstance(vid_mask_writer, cv2.VideoWriter):
                        vid_mask_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_mask_writer = cv2.VideoWriter(save_mask_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_mask_writer.write(mask_final)

        return save_path, save_mask_path


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, weight_path, out_file_path, real_time_show_predict_flag: bool, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Image Segmentation System for Crack Detection " + CODE_VER)
        self.showMaximized()

        '''按键绑定'''
        # 输入媒体
        self.import_media_pushButton.clicked.connect(self.import_media)  # 导入
        self.start_predict_pushButton.clicked.connect(self.predict_button_click)  # 开始推理
        # 输出媒体
        self.open_predict_file_pushButton.clicked.connect(self.open_file_in_browser)  # 文件中显示推理视频
        # 下方
        self.play_pushButton.clicked.connect(self.play_pause_button_click)  # 播放
        self.pause_pushButton.clicked.connect(self.play_pause_button_click)  # 暂停
        self.button_dict = dict()
        self.button_dict.update({"import_media_pushButton": self.import_media_pushButton,
                                 "start_predict_pushButton": self.start_predict_pushButton,
                                 "open_predict_file_pushButton": self.open_predict_file_pushButton,
                                 "play_pushButton": self.play_pushButton,
                                 "pause_pushButton": self.pause_pushButton,
                                 "real_time_checkBox": self.real_time_checkBox
                                 })

        '''媒体流绑定输出'''
        self.input_player = QMediaPlayer()  # 媒体输入的widget
        self.input_player.setVideoOutput(self.input_video_widget)
        self.input_player.positionChanged.connect(self.change_slide_bar)  # 播放进度条

        self.output_player = QMediaPlayer()  # 媒体输出的widget
        self.output_player.setVideoOutput(self.output_video_widget)

        self.output_mask_player = QMediaPlayer()  # 媒体输入的widget
        self.output_mask_player.setVideoOutput(self.output_mask_video_widget)

        '''初始化GPU chart'''
        self.series = QSplineSeries()
        self.chart_init()

        '''初始化GPU定时查询定时器'''
        # 使用QTimer，0.5秒触发一次，更新数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.draw_gpu_info_chart)
        self.timer.start(1000)

        # 播放时长, 以 input 的时长为准
        self.video_length = 0
        self.out_file_path = out_file_path
        # 推理使用另外一线程
        self.predict_handler_thread = PredictHandlerThread(self.input_player,
                                                           self.output_player,
                                                           self.out_file_path,
                                                           weight_path,
                                                           self.predict_info_plainTextEdit,
                                                           self.predict_progressBar,
                                                           self.fps_label,
                                                           self.button_dict,
                                                           self.input_media_tabWidget,
                                                           self.output_media_tabWidget,
                                                           self.input_real_time_label,
                                                           self.output_real_time_label,
                                                           self.output_mask_player,
                                                           self.output_mask_media_tabWidget,
                                                           self.output_mask_real_time_label,
                                                           real_time_show_predict_flag
                                                           )
        self.weight_label.setText(f" Using weight : ****** {Path(weight_path).name} ******")
        # 界面美化
        self.gen_better_gui()

        self.media_source = ""  # 推理媒体的路径

        self.predict_progressBar.setValue(0)  # 进度条归零

        '''check box 绑定'''
        self.real_time_checkBox.stateChanged.connect(self.real_time_checkbox_state_changed)
        self.real_time_checkBox.setChecked(real_time_show_predict_flag)
        self.real_time_check_state = self.real_time_checkBox.isChecked()

    def gen_better_gui(self):
        """
        美化界面
        :return:
        """
        # Play 按钮
        play_icon = QIcon()
        play_icon.addPixmap(QPixmap("./UI/icon/play.png"), QIcon.Normal, QIcon.Off)
        self.play_pushButton.setIcon(play_icon)

        # Pause 按钮
        play_icon = QIcon()
        play_icon.addPixmap(QPixmap("./UI/icon/pause.png"), QIcon.Normal, QIcon.Off)
        self.pause_pushButton.setIcon(play_icon)

        # 隐藏 tab 标题栏
        self.input_media_tabWidget.tabBar().hide()
        self.output_media_tabWidget.tabBar().hide()
        self.output_mask_media_tabWidget.tabBar().hide()
        # tab 设置显示第一栏
        self.input_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
        self.output_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
        self.output_mask_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)

        # 设置显示图片的 label 为黑色背景
        self.input_real_time_label.setStyleSheet("QLabel{background:black}")
        self.output_real_time_label.setStyleSheet("QLabel{background:black}")
        self.output_mask_real_time_label.setStyleSheet("QLabel{background:black}")

        # self.input_real_time_label.setFixedSize(self.input_media_tabWidget.width(), self.input_media_tabWidget.height())

    def real_time_checkbox_state_changed(self):
        """
        切换是否实时显示推理图片
        :return:
        """
        self.real_time_check_state = self.real_time_checkBox.isChecked()
        self.predict_handler_thread.real_time_show_predict_flag = self.real_time_check_state

    def chart_init(self):
        """
        初始化 GPU 折线图
        :return:
        """
        # self.gpu_info_chart._chart = QChart(title="折线图堆叠")  # 创建折线视图
        self.gpu_info_chart._chart = QChart()  # 创建折线视图
        # chart._chart.setBackgroundVisible(visible=False)      # 背景色透明
        self.gpu_info_chart._chart.setBackgroundBrush(QBrush(QColor("#FFFFFF")))  # 改变图背景色

        # 设置曲线名称
        self.series.setName("GPU Utilization")
        # 把曲线添加到QChart的实例中
        self.gpu_info_chart._chart.addSeries(self.series)
        # 声明并初始化X轴，Y轴
        self.dtaxisX = QDateTimeAxis()
        self.vlaxisY = QValueAxis()
        # 设置坐标轴显示范围
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        self.vlaxisY.setMin(0)
        self.vlaxisY.setMax(100)
        # 设置X轴时间样式
        self.dtaxisX.setFormat("hh:mm:ss")
        # 设置坐标轴上的格点
        self.dtaxisX.setTickCount(5)
        self.vlaxisY.setTickCount(10)
        # 设置坐标轴名称
        self.dtaxisX.setTitleText("Time")
        self.vlaxisY.setTitleText("Percent")
        # 设置网格不显示
        self.vlaxisY.setGridLineVisible(False)
        # 把坐标轴添加到chart中
        self.gpu_info_chart._chart.addAxis(self.dtaxisX, Qt.AlignBottom)
        self.gpu_info_chart._chart.addAxis(self.vlaxisY, Qt.AlignLeft)
        # 把曲线关联到坐标轴
        self.series.attachAxis(self.dtaxisX)
        self.series.attachAxis(self.vlaxisY)
        # 生成 折线图
        self.gpu_info_chart.setChart(self.gpu_info_chart._chart)

    def draw_gpu_info_chart(self):
        """
        绘制 GPU 折线图
        :return:
        """
        # 获取当前时间
        time_current = QDateTime.currentDateTime()
        # 更新X轴坐标
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        # 当曲线上的点超出X轴的范围时，移除最早的点
        remove_count = 600
        if self.series.count() > remove_count:
            self.series.removePoints(0, self.series.count() - remove_count)
        # 对 y 赋值
        # yint = random.randint(0, 100)
        gpu_info = get_gpu_info()
        yint = gpu_info[0].get("gpu_load")
        # 添加数据到曲线末端
        self.series.append(time_current.toMSecsSinceEpoch(), yint)

    def import_media(self):
        """
        导入媒体文件
        :return:
        """
        self.media_source = QFileDialog.getOpenFileUrl()[0]
        self.input_player.setMedia(QMediaContent(self.media_source))  # 选取视频文件
        self.input_player.pause()  # 显示媒体

        # 设置 output 为一张图片，防止资源被占用
        path_current = str(Path.cwd().joinpath(r"./UI/icon/wait.png"))
        self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(path_current)))
        # self.output_player.setMedia(QMediaContent(self.media_source))  # 选取视频文件
        self.output_player.pause()  # 显示媒体

        self.output_mask_player.setMedia(QMediaContent(QUrl.fromLocalFile(path_current)))
        self.output_mask_player.pause()  # 显示媒体

        # 将 QUrl 路径转为 本地路径str
        self.predict_handler_thread.parameter_source = self.media_source.toLocalFile()

        image_flag = os.path.splitext(self.predict_handler_thread.parameter_source)[-1].lower() in img_formats
        for item, button in self.button_dict.items():
            if image_flag and item in ['play_pushButton', 'pause_pushButton']:
                button.setEnabled(False)
            else:
                button.setEnabled(True)
        # self.output_player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件

    def predict_button_click(self):
        """
        推理按钮
        :return:
        """
        # 启动线程去调用
        self.predict_handler_thread.start()

    def change_slide_bar(self, position):
        """
        进度条移动
        :param position:
        :return:
        """
        self.video_length = self.input_player.duration() + 0.1
        self.video_horizontalSlider.setValue(round((position / self.video_length) * 100))
        self.video_percent_label.setText(str(round((position / self.video_length) * 100, 2)) + '%')

    @pyqtSlot()
    def play_pause_button_click(self):
        """
        播放、暂停按钮回调事件
        :return:
        """
        name = self.sender().objectName()

        if self.media_source == "":
            return

        if name == "play_pushButton":
            print("play")
            self.input_player.play()
            self.output_player.play()
            self.output_mask_player.play()

        elif name == "pause_pushButton":
            self.input_player.pause()
            self.output_player.pause()
            self.output_mask_player.pause()

    @pyqtSlot()
    def open_file_in_browser(self):
        os.system(f"start explorer {self.out_file_path}")

    @pyqtSlot()
    def closeEvent(self, *args, **kwargs):
        """
        重写关闭事件
        :param args:
        :param kwargs:
        :return:
        """
        print("Close")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    weight_root = Path.cwd().joinpath("weights")
    if not weight_root.exists():
        raise FileNotFoundError("weights not found !!!")

    weight_file = [item for item in weight_root.iterdir() if item.suffix == ".pth"]
    weight_root = str(weight_file[0])  # 权重文件位置
    out_file_root = Path.cwd().joinpath(r'result/output')
    out_file_root.parent.mkdir(exist_ok=True)
    out_file_root.mkdir(exist_ok=True)

    real_time_show_predict = True  # 是否实时显示推理图片，有可能导致卡顿，软件卡死

    main_window = MainWindow(weight_root, out_file_root, real_time_show_predict)

    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon/icon.ico"), QIcon.Normal, QIcon.Off)
    main_window.setWindowIcon(icon)

    main_window.show()
    sys.exit(app.exec_())
