from ultralytics import YOLO
import cv2
from collections import defaultdict

# ----------------------------------------------------------------------
# 1. การตั้งค่าเริ่มต้น
# ----------------------------------------------------------------------

# โหลดโมเดล YOLOv8 (nano รุ่นเล็ก/เร็ว)
model = YOLO('yolov8n.pt')

# กำหนดเส้นทางไฟล์วิดีโอ (ใช้ absolute path เพื่อความปลอดภัย)
video_path = r"D:\Senior Project\Video\people.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: ไม่สามารถเปิดไฟล์วิดีโอได้")
    exit()

# ตัวแปรสำหรับเก็บข้อมูลการนับและการติดตาม
track_history = defaultdict(lambda: [])
total_in = 0
total_out = 0

# กำหนดคลาสที่สนใจ: [0] คือ 'person'
target_classes = [0]

# ----------------------------------------------------------------------
# 2. กำหนดเส้นนับ (Counting Line)
# ----------------------------------------------------------------------
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
counting_line_x = int(frame_width * 0.1)  # 10% จากซ้ายของ frame

# ----------------------------------------------------------------------
# 3. ลูปประมวลผลวิดีโอ (Frame-by-Frame)
# ----------------------------------------------------------------------

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        print("⚠️ Warning: ไม่สามารถอ่าน frame ได้ / วิดีโอจบ")
        break

    # Resize frame ให้ตรงกับ input ของ YOLO (640x640)
    frame_resized = cv2.resize(frame, (640, 640))

    # รันโมเดลเพื่อตรวจจับและติดตามวัตถุ (ByteTrack)
    results = model.track(frame_resized, persist=True, tracker='bytetrack.yaml',
                          classes=target_classes, verbose=False)

    # 3.1 การนับจำนวนคน
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N_boxes x 4]
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)

            # เก็บประวัติการเคลื่อนที่ของวัตถุแต่ละ ID
            track_history[track_id].append(center_x)
            if len(track_history[track_id]) > 2:
                current_x = track_history[track_id][-1]
                previous_x = track_history[track_id][-2]

                # ตรวจสอบการข้ามเส้น
                if previous_x < counting_line_x and current_x >= counting_line_x:
                    total_out += 1
                    track_history[track_id] = []  # ป้องกันการนับซ้ำ
                elif previous_x > counting_line_x and current_x <= counting_line_x:
                    total_in += 1
                    track_history[track_id] = []

            # วาดกรอบและ ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ----------------------------------------------------------------------
    # 4. การแสดงผลลัพธ์บนจอ
    # ----------------------------------------------------------------------
    cv2.line(frame, (counting_line_x, 0), (counting_line_x, frame_height), (255, 0, 0), 3)
    cv2.putText(frame, f"IN: {total_in}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"OUT: {total_out}", (frame_width - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Object Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------
# 5. สิ้นสุดโปรแกรม
# ----------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print(f"\n✅ สรุปการนับ: เข้า (IN) = {total_in}, ออก (OUT) = {total_out}")

# ----------------------------------------------------------------------
# 6. Export โมเดลเป็น ONNX (fixed input size)
# ----------------------------------------------------------------------
model.export(format='onnx', imgsz=640, opset=11)
print('✅ Export ONNX สำเร็จ')
