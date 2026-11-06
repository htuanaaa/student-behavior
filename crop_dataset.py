import os
import sys
import zlib

import cv2
import numpy as np

P = os.environ.get("PROJECT_DIR", "/home/tuanexodus/codeaiss/project")
sys.path.insert(0, os.path.join(P, "src"))
os.chdir(P)

from services.pose_predict import PoseEstimation

OUT = os.path.join(P, "dataset_raw")
DUP_HAMMING = 3
DUP_WINDOW = 40

SOURCES = [
    ("video1.mp4", "sleep", "sau_lung", 0, -1, 60, 0.10),
    ("video2.mp4", "sleep", "ngang_hong", 0, -1, 60, 0.10),
    ("video3.mp4", "sleep", "cao_truoc_trai", 0, -1, 60, 0.10),
    ("video4.mp4", "sleep", "cao_truoc_rong", 0, -1, 60, 0.10),
    ("IMG_1017.MOV", "sleep", "sau_lung_1080", 0, -1, 60, 0.10),
    ("IMG_1016.MOV", "sleep", "truoc_cheo_xuong", 3840, 6150, 60, 0.10),
    ("IMG_1016.MOV", "awake", "truoc_cheo_xuong", 0, 3600, 15, 0.10),
    ("sample/classroom_cctv.mp4", "awake", "cao_dau_lop", 0, -1, 15, 0.25),
]


def dhash(img):
    g = cv2.cvtColor(cv2.resize(img, (9, 8)), cv2.COLOR_BGR2GRAY)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def head_up(person_data, margin):
    nose, ls, rs = person_data[0], person_data[5], person_data[6]
    if ls[2] < 0.4 or rs[2] < 0.4 or nose[2] < 0.4:
        return False
    sw = np.hypot(rs[0] - ls[0], rs[1] - ls[1])
    if sw <= 1e-3:
        return False
    return nose[1] < (ls[1] + rs[1]) / 2 - margin * sw


def collect(pose):
    rows = []
    for name, label, angle, start, stop, step, margin in SOURCES:
        cap = cv2.VideoCapture(os.path.join(P, name))
        if not cap.isOpened():
            print("không mở được:", name)
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stop = total if stop < 0 else min(stop, total)
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tag = os.path.splitext(os.path.basename(name))[0].replace(".", "")
        recent, made = {}, 0
        k = start
        while k < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if (k - start) % step == 0:
                _, persons = pose.detect(frame)
                for p in persons:
                    data = p["person_data"]
                    if label == "awake" and (min(data[1][2], data[2][2]) < 0.4 or not head_up(data, margin)):
                        continue
                    crop, ok_crop = pose.get_face_crop(frame, data)
                    if not ok_crop:
                        continue
                    key = (tag, int(p["id"]), label)
                    h = dhash(crop)
                    if any(bin(h ^ old).count("1") <= DUP_HAMMING for old in recent.get(key, [])):
                        continue
                    recent.setdefault(key, []).append(h)
                    recent[key] = recent[key][-DUP_WINDOW:]
                    rows.append({"img": crop, "label": label, "angle": angle, "source": tag,
                                 "track": int(p["id"]), "frame": k,
                                 "group": "%s_%s" % (tag, int(p["id"]))})
                    made += 1
            k += 1
        cap.release()
        print("%-26s %-5s %-16s tao=%4d" % (name, label, angle, made))
    return rows


def assign_splits(rows, ratios=(0.75, 0.10, 0.15)):
    groups = {}
    for r in rows:
        groups.setdefault((r["label"], r["group"]), []).append(r)
    for label in ("sleep", "awake"):
        items = sorted(((k, v) for k, v in groups.items() if k[0] == label),
                       key=lambda kv: len(kv[1]), reverse=True)
        total = sum(len(v) for _, v in items)
        target = [total * x for x in ratios]
        got = [0, 0, 0]
        for _, members in items:
            s = max(range(3), key=lambda i: target[i] - got[i])
            for r in members:
                r["split"] = ("train", "valid", "test")[s]
            got[s] += len(members)
        print("%-5s chia tập: train=%d valid=%d test=%d" % (label, got[0], got[1], got[2]))
    return rows


def balance(rows, ratio=4.0):
    dropped = 0
    for split in ("train", "valid", "test"):
        sl = [r for r in rows if r["split"] == split and r["label"] == "sleep"]
        aw = sum(1 for r in rows if r["split"] == split and r["label"] == "awake") or 1
        if len(sl) <= ratio * aw:
            continue
        keep = max(1, int(round(len(sl) / (ratio * aw))))
        seen, out = {}, []
        for r in rows:
            if r["split"] == split and r["label"] == "sleep":
                c = seen[r["group"]] = seen.get(r["group"], 0) + 1
                if c % keep:
                    dropped += 1
                    continue
            out.append(r)
        rows = out
    return rows, dropped


def main():
    pose = PoseEstimation()
    rows = assign_splits(collect(pose))
    rows, dropped = balance(rows)
    if os.path.isdir(OUT):
        for root, _, files in os.walk(OUT, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
    for r in rows:
        d = os.path.join(OUT, r["split"], r["label"])
        os.makedirs(d, exist_ok=True)
        fn = "%s_t%02d_%06d.jpg" % (r["source"], r["track"], r["frame"])
        cv2.imwrite(os.path.join(d, fn), r["img"])
        r["file"] = os.path.relpath(os.path.join(d, fn), OUT)
    with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
        f.write("file\tsplit\tlabel\tangle\tsource\ttrack\tframe\n")
        for r in rows:
            f.write("%s\t%s\t%s\t%s\t%s\t%d\t%d\n" % (r["file"], r["split"], r["label"],
                                                      r["angle"], r["source"], r["track"], r["frame"]))
    print("tổng= %d | bỏ bớt khi cân bằng= %d" % (len(rows), dropped))
    for split in ("train", "valid", "test"):
        s = sum(1 for r in rows if r["split"] == split and r["label"] == "sleep")
        a = sum(1 for r in rows if r["split"] == split and r["label"] == "awake")
        print("%-6s sleep=%4d awake=%4d" % (split, s, a))
    counts = {}
    for r in rows:
        counts[(r["angle"], r["label"])] = counts.get((r["angle"], r["label"]), 0) + 1
    print("theo góc:", sorted("%s/%s=%d" % (k[0], k[1], v) for k, v in counts.items()))


if __name__ == "__main__":
    main()
