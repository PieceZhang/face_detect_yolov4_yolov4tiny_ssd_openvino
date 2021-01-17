import numpy as np
from openvino.inference_engine import IECore
import cv2
import sys
import time
import argparse
from decode_np import Decode


def build_argparser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t", "--tiny", action="store_true",
                        help='store_true: if model is v4 or v4tiny, default: v4')
    parser.add_argument("-d", "--device", type=str, default='VPU',
                        help='str: use CPU or VPU, default=\'VPU\'')
    parser.add_argument("-s", "--source", type=str, default='camera',
                        help='str: detect from camera or file, default=\'camera\'')
    parser.add_argument("-p", "--path", type=str, default='image1.jpg',
                        help='str: path to file if detect from files, default=\'image1.jpg\'')
    args = parser.parse_args()
    assert args.device is 'CPU' or 'VPU', 'parser error! use \'-h\' for help'
    assert args.source is 'camera' or 'file', 'parser error! use \'-h\' for help'
    return args


if __name__ == "__main__":
    args = build_argparser()

    if args.source == 'file':
        image_origin = cv2.imread(args.path)
        assert image_origin is not None, 'Image is not found, No such file or directory'

    print("\nDetect initing...")
    print('=' * 30)

    # load network
    if args.tiny:
        print('model: v4tiny')
        model_xml = './IR_FP16/yolov4-tiny.xml'
        model_bin = './IR_FP16/yolov4-tiny.bin'
    else:
        print('model: v4')
        model_xml = './IR_FP16/yolov4.xml'
        model_bin = './IR_FP16/yolov4.bin'

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)

    print("inputs number: " + str(len(net.input_info.keys())))
    for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape
    print('=' * 30)

    # build net
    print("Loading model to the device...")
    exec_net = ie.load_network(network=net, device_name='MYRIAD' if args.device == 'VPU' else 'CPU')
    print("Creating infer request and starting inference...")

    if args.tiny:
        conf_thresh = 0.25
    else:
        conf_thresh = 0.5
    nms_thresh = 0.60
    input_shape = (416, 416)
    all_classes = ['face']

    # ---------------------------------------------- camera --------------------------------------------------
    _decode = Decode(conf_thresh, nms_thresh, input_shape, all_classes, exec_net, iftiny=args.tiny)

    if args.source == 'camera':
        print('detect from camera...')
        cam = cv2.VideoCapture(0)
        ifsuccess, frame_origin = cam.read()
        assert ifsuccess is True, 'camera error'
        while 1:
            ifsuccess, frame_origin = cam.read()
            time_start = time.time()  # start
            image, boxes, scores, classes = _decode.detect_image(frame_origin, draw_image=True)
            time_stop = time.time()  # stop
            cost_time = time_stop - time_start
            # print(args.device, 'fps: ', 1 / cost_time)
            image = cv2.putText(image, 'Model: {}'.format('YOLOv4-tiny' if args.tiny else 'YOLOv4'), (10, 25),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            image = cv2.putText(image, 'Device: {}'.format(args.device), (10, 50),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            image = cv2.putText(image, 'Cost: {:2.2f} ms'.format(cost_time),
                                (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            image = cv2.putText(image,
                                'FPS: {:2.2f}'.format(1 / cost_time) if cost_time > 0 else 'FPS: --',
                                (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            cv2.imshow("capture", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
    elif args.source == 'file':
        print('detect from file: {}...'.format(args.path))
        time_start = time.time()  # start
        image, boxes, scores, classes = _decode.detect_image(image_origin, draw_image=True)
        time_stop = time.time()  # stop
        cost_time = time_stop - time_start
        # print(args.device, 'fps: ', 1 / cost_time)
        image = cv2.putText(image, 'Model: {}'.format('YOLOv4-tiny' if args.tiny else 'YOLOv4'), (10, 25),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        image = cv2.putText(image, 'Device: {}'.format(args.device), (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        image = cv2.putText(image, 'Cost: {:2.2f} ms'.format(cost_time),
                            (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        image = cv2.putText(image,
                            'FPS: {:2.2f}'.format(1 / cost_time) if cost_time > 0 else 'FPS: --',
                            (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        cv2.imwrite('result.jpg', image)
        cv2.imshow("capture", image)
        cv2.waitKey()
        cv2.destroyAllWindows()