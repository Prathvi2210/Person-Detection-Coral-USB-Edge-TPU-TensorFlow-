import argparse
import cv2
import os
import time
import lgpio

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

CHIP = 0
PIN = 18   # GPIO18

def main():
    default_model_dir = '/home/emhs/Projects/Coral/pycoral/test_data/EfficientDet Models'
    default_model = 'efficientdet_lite0_320_ptq_edgetpu.tflite'
    default_labels = 'efficientdet_lite_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--camera_idx', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()

    print(f'Loading {args.model} with {args.labels}')
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)

    # ---- GPIO SETUP ----
    h = lgpio.gpiochip_open(CHIP)
    lgpio.gpio_claim_output(h, PIN, lgpio.HIGH)   # relay OFF initially
    print("GPIO ready on pin", PIN)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            start_time = time.time()
            if not ret:
                break

            cv2_im = frame
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

            run_inference(interpreter, cv2_im_rgb.tobytes())

            objs = get_objects(interpreter, args.threshold)
            objs = [obj for obj in objs if labels.get(obj.id, "") == "person"]
            objs = sorted(objs, key=lambda x: -x.score)[:args.top_k]

            # ---- RELAY CONTROL ----
            if len(objs) > 0:
                lgpio.gpio_write(h, PIN, lgpio.LOW)   # relay ON
            else:
                lgpio.gpio_write(h, PIN, lgpio.HIGH)  # relay OFF

            cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ---- CLEANUP ----
        print("Cleaning up GPIO...")
        lgpio.gpio_write(h, PIN, lgpio.HIGH)  # make sure relay OFF
        lgpio.gpio_free(h, PIN)
        lgpio.gpiochip_close(h)
        cap.release()
        cv2.destroyAllWindows()


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = f'{percent}% {labels.get(obj.id, obj.id)}'

        cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(cv2_im, label, (x0, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    return cv2_im


if __name__ == '__main__':
    main()
