# -*- coding: utf-8 -*-
import random
import colorsys
import cv2
# import threading
import os
import numpy as np
import tensorflow as tf


class Decode(object):
    def __init__(self, obj_threshold, nms_threshold, input_shape, all_classes, net, iftiny=True):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self.input_shape = input_shape
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self.iftiny = iftiny
        self.net = net

    # 处理一张图片
    def detect_image(self, image, draw_image):
        pimage = self.process_image(np.copy(image))

        boxes, scores, classes = self.predict(pimage, image.shape)
        if boxes is not None and draw_image:
            self.draw(image, boxes, scores, classes)
        return image, boxes, scores, classes

    def draw(self, image, boxes, scores, classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            # show classes and scores
            # bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            # t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            # cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            # cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale_x = float(self.input_shape[1]) / w
        scale_y = float(self.input_shape[0]) / h
        img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        img = img.transpose((2, 0, 1))
        # 转换为字典
        data = {}
        data['inputs'] = np.expand_dims(img, axis=0)
        # pimage = img.astype(np.float32) / 255.
        # pimage = np.expand_dims(pimage, axis=0)
        return data

    def predict(self, image, shape):  #
        if self.iftiny:
            res = self.net.infer(inputs=image)
            outs0 = res['detector/yolo-v4-tiny/Reshape']
            outs1 = res['detector/yolo-v4-tiny/Reshape_4']
            # numpy后处理
            a1 = np.reshape(outs0, (1, self.input_shape[0]//32, self.input_shape[1]//32, 3, 5+self.num_classes))
            a2 = np.reshape(outs1, (1, self.input_shape[0]//16, self.input_shape[1]//16, 3, 5+self.num_classes))
            boxes, scores, classes = self._yolo_out([a1, a2], shape)
        else:
            res = self.net.infer(inputs=image)
            outs0 = res['detector/yolo-v4/Reshape_8']
            outs1 = res['detector/yolo-v4/Reshape_4']
            outs2 = res['detector/yolo-v4/Reshape']
            # numpy后处理
            a1 = np.reshape(outs0, (1, self.input_shape[0]//32, self.input_shape[1]//32, 3, 5+self.num_classes))
            a2 = np.reshape(outs1, (1, self.input_shape[0]//16, self.input_shape[1]//16, 3, 5+self.num_classes))
            a3 = np.reshape(outs2, (1, self.input_shape[0]//8, self.input_shape[1]//8, 3, 5+self.num_classes))
            boxes, scores, classes = self._yolo_out([a1, a2, a3], shape)

        return boxes, scores, classes


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _process_feats(self, out, anchors, mask):
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

        anchors = [anchors[i] for i in mask]
        anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

        # Reshape to batch, height, width, num_anchors, box_params.
        out = out[0]
        box_xy = self._sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4])
        box_wh = box_wh * anchors_tensor

        box_confidence = self._sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self._sigmoid(out[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_shape
        box_xy -= (box_wh / 2.)   # 坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._t1)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep


    def _yolo_out(self, outs, shape):
        if self.iftiny:
            masks = [[3, 4, 5], [0, 1, 2]]
            anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
        else:
            masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146],
                       [142, 110], [192, 243], [459, 401]]

        boxes, classes, scores = [], [], []

        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        # boxes坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
        # Scale boxes back to original image shape.
        w, h = shape[1], shape[0]
        image_dims = [w, h, w, h]
        boxes = boxes * image_dims

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        # 换坐标
        boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

        return boxes, scores, classes


