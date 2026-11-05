import collections
import os
import sys
import cv2

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from services.pose_predict import PoseEstimation

OUT_DIR = "dataset_classroom"
STEP = 15            # 0,5 giây (15 frame) / 30 fps lấy 1 ảnh
MIN_TRACK = 15       # bỏ mặt xuất hiện dưới 15 frame
MIN_FACE = 40        # bỏ kích thước mắt dưới 40 px

def face_width(person_data):
    points = person_data[:5]
    points = points[points[:, 2] >= 0.35][:, :2]
    if len(points) < 2:
        return 0.0
    xs, ys = points[:, 0], points[:, 1]
    return max(xs.max() - xs.min(), ys.max() - ys.min())


def crop_video(pose, path):
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("video không mở được:", path)
        return 0

    picked = []
    frames_seen = collections.Counter()

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, persons = pose.detect(frame)
        for p in persons:
            frames_seen[p["id"]] += 1
            if frame_no % STEP:
                continue
            if face_width(p["person_data"]) < MIN_FACE:
                continue
            crop, face_ok = pose.get_face_crop(
                frame, p["person_data"],
                margin=1.8, min_conf=0.35, out_size=(128, 128))
            if face_ok and crop is not None:
                picked.append((frame_no, p["id"], crop))
        frame_no += 1
    cap.release()

    total_frames = frame_no
    written = 0
    for fno, track_id, crop in picked:
        if frames_seen[track_id] < MIN_TRACK:
            continue
        file_name = "%s_f%06d_id%02d.jpg" % (name, fno, track_id)
        cv2.imwrite(os.path.join(OUT_DIR, file_name), crop)
        written += 1

    print("%-22s %5d frame  %2d nguoi  ->  %4d anh"
          % (name, total_frames, len(frames_seen), written))
    return written


def main():
    videos = sys.argv[1:]
    if not videos:
        print("python crop_dataset.py video1.mp4 [video2.mp4 ...]")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    pose = PoseEstimation(
        model_path=os.path.join(PROJECT, "yolomodel", "yolov8n-pose.pt"))

    total = 0
    for path in videos:
        total += crop_video(pose, path)
    print("TONG:", total, "anh ->", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
