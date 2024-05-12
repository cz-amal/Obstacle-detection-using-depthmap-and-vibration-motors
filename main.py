import joblib
import time
import click
import json
import cv2

from src.face_detector import FaceDetector
from src import utils
import torch
import cv2
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import time
import cv2
from face_detector import FaceDetector
import utils
 
from torchvision.ops import boxes
import numpy as np
import time
import random
import paho.mqtt.publish as publish
model = joblib.load('linear-depth.pkl')

mqtt_broker = "mqtt.eclipseprojects.io"
mqtt_port = 1883
topic = "vibration_data"

model_type = "MiDaS_small"
# model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
previous_depth = 0.0

@click.command()
@click.option('-v','--video_source', default=0)
@click.option('-c','--confidence', type=float, default=0.5)



def main(video_source, confidence):

    detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=confidence,
                            overlap_thr=0.7)
    video = cv2.VideoCapture(video_source)
    alpha = 0.2

    depth_scale = 1.0

    # Applying exponential moving average filter
    def evaluate_spline(coord):
        try:
            x, y = coord
        except ValueError:
            return 1

        return spline(x, y)

    # def apply_ema_filter(current_depth):
    #     global previous_depth
    #     filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    #     previous_depth = filtered_depth  # Update the previous depth value
    #     return filtered_depth

    # Define depth to distance
    def depth_to_distance(depth_value, depth_scale=1):
         return 1.0 / (depth_value * depth_scale)



    while True:
        vibration_mat = [0,0,0]

        ret, frame = video.read()
        bboxes, scores = detector.inference(frame)
        loc = ""


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_batch = transform(image).to('cuda')

        with torch.no_grad():
            prediction = midas(image_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

            output = prediction.cpu().numpy()

            h, w = output.shape
            x_grid = np.arange(w)
            y_grid = np.arange(h)
            # Create a spline object using the output_norm array
            spline = RectBivariateSpline(y_grid, x_grid, output)



        frame = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        right = []
        center = []
        left = []

        if len(bboxes)!= 0:
            for det in bboxes:

                xmin,ymin,xmax,ymax = det
                mx,my = (xmin+xmax)/2,(ymin+ymax)/2
                print(mx,my)
                if mx < w/3:
                    loc = "left"
                    left.append((mx,my))
                elif mx > w/3 and mx < 2*w/3:
                    loc = "center"
                    center.append((mx,my))

                else:
                    loc = "right"
                    right.append((mx,my))


                depth = spline(mx, my)
                depth_midas = depth_to_distance(depth, depth_scale)

                dpt = depth_midas.tolist()[0][0] * 10000

                result = model.predict([[dpt]])
                rdistance = int(result.tolist()[0])
                frame = utils.draw_boxes_with_scores(frame, bboxes, scores)
                cv2.putText(frame,
                            loc,
                            (int(mx), int(my)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 255, 0),
                            )
                cv2.putText(frame,
                            str(rdistance)+" cm",
                            (int(mx), int(my+20)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.2,
                            (0, 255, 0),
                            )

            if right:
                print("in right")
                right = list(map(evaluate_spline,right))
                right = list(map(depth_to_distance, right))


                smallest = min(right[0][0])

                result = model.predict([[smallest*10000]])

                vibration_mat[2] = int(result.tolist()[0])
            if center:
                print("in center")
                center = list(map(evaluate_spline,center))
                center = list(map(depth_to_distance, center))
                smallest = min(center[0][0])
                result = model.predict([[smallest*10000]])
                vibration_mat[1] = int(result.tolist()[0])
            if left:
                print("in left")
                left = list(map(evaluate_spline,left))
                left = list(map(depth_to_distance,left))
                smallest = min(left[0][0])
                result = model.predict([[smallest*10000]])
                vibration_mat[0] = int(result.tolist()[0])


            print(vibration_mat)
            data_json = json.dumps(vibration_mat)
            publish.single(topic, payload=data_json, hostname=mqtt_broker, port=mqtt_port)




        cv2.imshow('CV2Frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

if __name__ == '__main__':
    main()
