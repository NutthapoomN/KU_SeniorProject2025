from ultralytics import YOLO
import cv2
from collections import defaultdict

# ----------------------------------------------------------------------
# 1. การตั้งค่าเริ่มต้น
# ----------------------------------------------------------------------

# โหลดโมเดล YOLOv8
# 'yolov8n.pt' คือโมเดลขนาดเล็ก/เร็ว (nano) สำหรับการใช้งานเริ่มต้น
model = YOLO('yolov8n.pt') 

# กำหนดเส้นทางไปยังไฟล์วิดีโอของคุณ (ต้องเปลี่ยนชื่อไฟล์นี้ให้ถูกต้อง)
video_path = 'people.mp4' 
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
# กำหนดเส้นนับให้อยู่ที่ 50% ของความสูงของภาพ
counting_line_x = int(frame_width * 0.1) 

# ----------------------------------------------------------------------
# 3. ลูปประมวลผลวิดีโอ (Frame-by-Frame Processing)
# ----------------------------------------------------------------------

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # รันโมเดลเพื่อตรวจจับและติดตามวัตถุ (Tracking)
    # ByteTrack เป็นหนึ่งในอัลกอริทึมติดตามที่มีประสิทธิภาพ
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=target_classes, verbose=False)

    # 3.1 การนับจำนวนคน/สิ่งของ
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2) # จุดศูนย์กลางแกน Y

            # เก็บประวัติการเคลื่อนที่ของวัตถุแต่ละ ID
            track_history[track_id].append(center_x)
            if len(track_history[track_id]) > 2: # ต้องการตำแหน่งปัจจุบันและก่อนหน้า
                 
                current_x = track_history[track_id][-1]
                previous_x = track_history[track_id][-2]
                
                # ตรวจสอบการข้ามเส้น (การนับ)
                # ข้ามจากบน (น้อยกว่า) ลงล่าง (มากกว่า) = OUT
                if previous_x < counting_line_x and current_x >= counting_line_x:
                    total_out += 1
                    track_history[track_id] = [] # ป้องกันการนับซ้ำ
                
                # ข้ามจากล่าง (มากกว่า) ขึ้นบน (น้อยกว่า) = IN
                elif previous_x > counting_line_x and current_x <= counting_line_x:
                    total_in += 1
                    track_history[track_id] = [] # ป้องกันการนับซ้ำ
            
            # วาดกรอบและ ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 3.2 การตรวจจับประตู: ต้องใช้โมเดลที่ฝึกมาเฉพาะ (หรือเทคนิคอื่น)
    # ณ จุดนี้คือการแสดงผลการตรวจจับคน/สิ่งของหลักเท่านั้น

    # ----------------------------------------------------------------------
    # 4. การแสดงผลลัพธ์บนจอ
    # ----------------------------------------------------------------------
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # วาดเส้นนับสีน้ำเงิน
    cv2.line(frame, (counting_line_x, 0), (counting_line_x, frame_height), (255, 0, 0), 3)

    # แสดงผลการนับ
    cv2.putText(frame, f"IN: {total_in}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"OUT: {total_out}", (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Object Counting", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# สิ้นสุดโปรแกรม
cap.release()
cv2.destroyAllWindows()
print(f"\n✅ สรุปการนับ: เข้า (IN) = {total_in}, ออก (OUT) = {total_out}")
