# yolo_counter.py
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# โหลดโมเดลแค่ครั้งเดียว
model = YOLO("yolov8n.pt")

# ตัวแปร global (คงค่าในทุกเฟรมจาก Simulink)
track_history = defaultdict(lambda: [])
total_in = 0
total_out = 0

# กำหนดเส้นนับ (กำหนดตอนรับเฟรมแรก)
initialized = False
counting_line_x = 0
frame_height = 0
frame_width = 0

def process_frame(frame):
    """
    frame = numpy array (H,W,3) uint8 ส่งมาจาก Simulink
    return: (total_in, total_out)
    """

    global initialized, counting_line_x, frame_height, frame_width
    global total_in, total_out, track_history

    # ----------------------------------------------------
    # 1) กำหนดค่าเริ่มต้นจากเฟรมแรก
    # ----------------------------------------------------
    if not initialized:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        counting_line_x = int(frame_width * 0.1)   # 10% ของความกว้าง
        initialized = True

    # ----------------------------------------------------
    # 2) รัน YOLO + tracking
    # ----------------------------------------------------
    results = model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        classes=[0],           # 0 = person
        verbose=False
    )

    # ----------------------------------------------------
    # 3) นับ IN/OUT แบบ logic เดิมเป๊ะๆ
    # ----------------------------------------------------
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)

            # เก็บประวัติ
            track_history[track_id].append(center_x)

            if len(track_history[track_id]) > 2:
                prev = track_history[track_id][-2]
                curr = track_history[track_id][-1]

                # OUT
                if prev < counting_line_x and curr >= counting_line_x:
                    total_out += 1
                    track_history[track_id] = []

                # IN
                elif prev > counting_line_x and curr <= counting_line_x:
                    total_in += 1
                    track_history[track_id] = []

    # ----------------------------------------------------
    # 4) ส่งผลลัพธ์กลับ
    # ----------------------------------------------------
    return int(total_in), int(total_out)
