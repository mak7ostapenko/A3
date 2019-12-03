from __future__ import division, print_function, absolute_import

import os
import argparse
import warnings

import cv2
import numpy as np
from PIL import Image
from yolo3.yolo import YOLO

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools import generate_detections as gdet
warnings.filterwarnings('ignore')


def file_system_work(videofile, out_root_dir):
    """Create output directories and files.
    """
    videofile_name = videofile.split('/')[-1].split('.')[0]
    out_dir = os.path.join(out_root_dir, videofile_name)

    # create directory for output
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    out_video_file_name = os.path.join(out_dir, 'RESULT_' + videofile_name)
    out_list_file_name = os.path.join(out_dir, 'DETECTION_LIST_RESULT_' + videofile_name)

    return out_video_file_name, out_list_file_name


def main(detector, videofile='input/real.MOV', out_root_dir='output',
         process_stream=False, writeVideo_flag = True, show_detections=False):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    tracks_ids = []

    if process_stream:
        print("SOURCE: Stream is processing.")
        video_capture = cv2.VideoCapture(0)
    else:
        print("SOURCE: File {} is processing.".format(videofile))
        video_capture = cv2.VideoCapture(videofile)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_video_file_name, out_list_file_name = file_system_work(videofile, out_root_dir)

        out = cv2.VideoWriter(out_video_file_name, fourcc, 15, (w, h))
        list_file = open(out_list_file_name, 'w')
        frame_index = -1

    print('EXECUTION: Processing...')
    print('EXECUTION: Press Q to stop execution.')
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # frame shape 640*480*3

        if not ret:
            break

        image = Image.fromarray(frame[...,::-1]) # bgr to rgb

        boxs = detector.detect_image(image)
        features = encoder(frame,boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maximum suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            if track.track_id not in tracks_ids:
                tracks_ids.append(track.track_id)

            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        if show_detections:
            cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')

            list_file.write('\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('############ RESULT ###################')
    print('RESULT: Number of tracks = ', len(tracks_ids))
    print('############ RESULT ###################')
    # end processing and write
    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()

    cv2.destroyAllWindows()


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="People counter")

    parser.add_argument("--videofile", default="input/real.MOV",
                        help="Path to file which you want to process.", required=True)
    parser.add_argument("--out_root_dir",  default="output",
                        help="Directory for output.", required=True)
    parser.add_argument("--process_stream", default=False,
                        help="If True then read video from camera else process file", required=True)
    parser.add_argument("--writeVideo_flag", default=True,
                        help="If True then write detections on output video else don't", required=True)
    parser.add_argument("--show_detections", default=False,
                        help="If True display detections on each frame of video, else don't. "
                             "NOTE: if you run on server it has to be FALSE", required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    detector = YOLO()
    main(detector,
         args.videofile,
         args.out_root_dir,
         args.process_stream,
         args.writeVideo_flag,
         args.show_detections)
