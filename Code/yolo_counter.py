# yolo_counter.py
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# โหลด YOLO แค่ครั้งเดียว
model = YOLO("yolov8n.pt")

# Global state
initialized = False
counting_line_x = 0
frame_width = 0
frame_height = 0

track_history = defaultdict(lambda: [])
total_in = 0
total_out = 0


def process_frame(frame):
    """
    frame: numpy array (H,W,3) uint8 จาก Simulink
    return: current_people, total_in, total_out
    """
    global initialized, counting_line_x, frame_width, frame_height
    global total_in, total_out, track_history

    # init จากเฟรมแรก
    if not initialized:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        counting_line_x = int(frame_width * 0.1)
        initialized = True

    # YOLO + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],   # person
        verbose=False
    )

    current_people = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        # จำนวนคนที่ตรวจพบในภาพ
        current_people = len(ids)

        # update movement + count IN/OUT
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)

            track_history[tid].append(cx)

            if len(track_history[tid]) > 2:
                prev = track_history[tid][-2]
                curr = track_history[tid][-1]

                # OUT
                if prev < counting_line_x and curr >= counting_line_x:
                    total_out += 1
                    track_history[tid] = []

                # IN
                elif prev > counting_line_x and curr <= counting_line_x:
                    total_in += 1
                    track_history[tid] = []

    return int(current_people), int(total_in), int(total_out)
