import os
import cv2
import json
import numpy as np
import pandas as pd
from dnndk import n2cube
from backend.utils import timeit, draw_boxed_text

KERNEL_CONV="tf_yolov3_voc"

"""DPU IN/OUT name for tf_yolov3_voc"""
CONV_INPUT_NODE="conv2d_1_convolution"
CONV_OUTPUT_NODE1="conv2d_59_convolution"
CONV_OUTPUT_NODE2="conv2d_67_convolution"
CONV_OUTPUT_NODE3="conv2d_75_convolution"

LABELS_PATH = './models/yolo-dnndk/labels.json'
ANCHORS_PATH = './models/yolo-dnndk/yolo_anchors.txt'
classes_path = "./models/yolo-dnndk/voc_classes.txt"

with open(LABELS_PATH) as json_data:
    CLASS_NAMES = json.load(json_data)

with open(classes_path) as f:
    class_names = f.readlines()
    class_names = [c.strip() for c in class_names]


class Detector():
    """Class yolo"""

    @timeit
    def __init__(self):
        self.model = n2cube.dpuOpen()
        self.kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
        self.colors = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

    @staticmethod
    def letterbox_image(image, size):
        ih, iw, _ = image.shape
        w, h = size
        scale = min(w/iw, h/ih)

        nw = int(iw*scale)
        nh = int(ih*scale)

        image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.ones((h,w,3), np.uint8) * 128
        h_start = (h-nh)//2
        w_start = (w-nw)//2
        new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
        return new_image


    @staticmethod
    def pre_process(image, model_image_size):
        image = image[...,::-1]
        image_h, image_w, _ = image.shape

        if model_image_size != (None, None):
            assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = Detector.letterbox_image(image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        return image_data

    @timeit
    def prediction(self, image):
        task = n2cube.dpuCreateTask(self.kernel, 0)

        image_data = Detector.pre_process(image, (416, 416))
        image_data = np.array(image_data,dtype=np.float32)
        input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)

        """Get input Tesor"""
        n2cube.dpuSetInputTensorInHWCFP32(task,CONV_INPUT_NODE,image_data,input_len)

        """Model run on DPU"""
        n2cube.dpuRunTask(task)

        """Get the output tensor"""
        conv_sbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1)
        conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, conv_sbbox_size)
        conv_out1 = np.reshape(conv_out1, (1, 13, 13, 75))

        conv_mbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2)
        conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, conv_mbbox_size)
        conv_out2 = np.reshape(conv_out2, (1, 26, 26, 75))

        conv_lbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3)
        conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, conv_lbbox_size)
        conv_out3 = np.reshape(conv_out3, (1, 52, 52, 75))

        return [conv_out1, conv_out2, conv_out3]

    @timeit
    def filter_prediction(self, yolo_outputs, image, conf_th=0.2, conf_class=[]):
        image_shape = image.shape[:2]
        # yolo_outputs, image_shape, max_boxes = 20
        score_thresh = 0.2
        nms_thresh = 0.45
        #class_names = get_class(classes_path)
        anchors     = get_anchors(ANCHORS_PATH)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []

        input_shape = np.shape(yolo_outputs[0])[1 : 3]
        input_shape = np.array(input_shape)*32

        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis = 0)
        box_scores = np.concatenate(box_scores, axis = 0)

        mask = box_scores >= score_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(class_names)):
            class_boxes_np = boxes[mask[:, c]]
            class_box_scores_np = box_scores[:, c]
            class_box_scores_np = class_box_scores_np[mask[:, c]]
            nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np)
            class_boxes_np = class_boxes_np[nms_index_np]
            class_box_scores_np = class_box_scores_np[nms_index_np]
            classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
            boxes_.append(class_boxes_np)
            scores_.append(class_box_scores_np)
            classes_.append(classes_np)
        boxes_ = np.concatenate(boxes_, axis = 0)
        scores_ = np.concatenate(scores_, axis = 0)
        classes_ = np.concatenate(classes_, axis = 0)

        df = pd.DataFrame(np.concatenate((boxes_, np.expand_dims(scores_, axis=1), np.expand_dims(classes_, axis=1)), axis=1),
                columns=['y1', 'x1', 'y2', 'x2', 'confidence', 'class_id'])
        df = df[df['confidence'] > conf_th]
        df = df.assign(
                x1=lambda x: (x['x1'] ).astype(int).clip(0),
                y1=lambda x: (x['y1'] ).astype(int).clip(0),
                x2=lambda x: (x['x2'] ).astype(int),
                y2=lambda x: (x['y2'] ).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)
                    ),
                label=lambda x: (
                    x.class_name + ': ' + (
                        x['confidence'].astype(str).str.slice(stop=4)
                        )
                    )
                )
        if len(conf_class) > 0:
            df = df[df['class_id'].isin(conf_class)]
        return df

    @timeit
    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            x_min, y_min, x_max, y_max = box['x1'], box['y1'], box['x2'], box['y2']
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            txt = box['label']
            image = draw_boxed_text(image, txt, txt_loc, color)
        return image


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)



def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes


def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores


def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.55)[0]  # threshold
        order = order[inds + 1]

    return keep


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)
    cv2.imwrite("./imgs/outputcv.jpg", image)


