import csv
import os
import shutil

P = os.environ.get("PROJECT_DIR", "/home/tuanexodus/codeaiss/project")
SRC = os.path.join(P, "dataset_raw")
OUT = os.path.join(P, os.environ.get("OUT_NAME", "dataset"))

TEST_ONLY = {s.strip() for s in os.environ.get("TEST_ONLY", "").split(",") if s.strip()}

DUPLICATE_HAMMING = 3
SLEEP_PER_AWAKE = 3
RATIOS = (0.70, 0.10, 0.20)
SPLITS = ("train", "valid", "test")


def dhash(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def read_manifest():
    rows = []
    with open(os.path.join(SRC, "manifest.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            path = os.path.join(SRC, r["file"])
            if os.path.exists(path):
                r["path"] = path
                r["frame"] = int(r["frame"])
                rows.append(r)
    return rows


def drop_duplicates(rows):
    rows = sorted(rows, key=lambda r: (r["source"], r["label"], r["frame"]))
    kept, hashes = [], []
    for r in rows:
        h = dhash(r["path"])
        if h is None:
            continue
        if any(bin(h ^ old).count("1") <= DUPLICATE_HAMMING for old in hashes):
            continue
        hashes.append(h)
        kept.append(r)
    return kept, len(rows) - len(kept)


def split_by_time(rows):
    for r in rows:
        if r["source"] in TEST_ONLY:
            r["split"] = "test"
    for (source, label), group in group_by(rows, ("source", "label")).items():
        if source in TEST_ONLY:
            continue
        frames = sorted({r["frame"] for r in group})
        n = len(frames)
        cut1 = int(round(n * RATIOS[0]))
        cut2 = int(round(n * (RATIOS[0] + RATIOS[1])))
        where = {}
        for i, frame in enumerate(frames):
            where[frame] = 0 if i < cut1 else (1 if i < cut2 else 2)
        for r in group:
            r["split"] = SPLITS[where[r["frame"]]]
    return rows


def group_by(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return groups


def limit_sleep(rows):
    # giữ tỷ lệ 3 sleep : 1 awake
    kept = []
    for split in SPLITS:
        awake = [r for r in rows if r["split"] == split and r["label"] == "awake"]
        sleep = sorted((r for r in rows if r["split"] == split and r["label"] == "sleep"),
                       key=lambda r: (r["source"], r["frame"]))
        allowed = min(len(sleep), SLEEP_PER_AWAKE * max(1, len(awake)))
        if allowed < len(sleep):
            step = len(sleep) / float(allowed)
            sleep = [sleep[int(i * step)] for i in range(allowed)]
        kept.extend(awake)
        kept.extend(sleep)
    return kept, len(rows) - len(kept)


def write_out(rows):
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for r in rows:
        d = os.path.join(OUT, r["split"], r["label"])
        os.makedirs(d, exist_ok=True)
        name = "%s_t%02d_%06d.jpg" % (r["source"], int(r["track"]), r["frame"])
        shutil.copyfile(r["path"], os.path.join(d, name))
        r["file"] = "%s/%s/%s" % (r["split"], r["label"], name)
    path = os.path.join(OUT, "manifest.tsv")
    with open(path, "w") as f:
        f.write("file\tsplit\tlabel\tangle\tsource\ttrack\tframe\n")
        for r in rows:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%d\n" % (
                r["file"], r["split"], r["label"], r["angle"], r["source"], r["track"], r["frame"]))


def report(rows):
    print("tổng=%d" % len(rows))
    for split in SPLITS:
        s = sum(1 for r in rows if r["split"] == split and r["label"] == "sleep")
        a = sum(1 for r in rows if r["split"] == split and r["label"] == "awake")
        print("%-6s sleep=%4d awake=%4d" % (split, s, a))
    print("theo góc quay:")
    for (angle, label), group in sorted(group_by(rows, ("angle", "label")).items()):
        print("  %-16s %-6s %4d" % (angle, label, len(group)))


def main():
    rows = read_manifest()
    print("đọc được %d ảnh" % len(rows))
    rows, removed_dup = drop_duplicates(rows)
    rows = split_by_time(rows)
    rows, removed_balance = limit_sleep(rows)
    write_out(rows)
    print("bỏ ảnh trùng= %d | bỏ bớt ảnh để cân bằng= %d" % (removed_dup, removed_balance))
    report(rows)
    print("đã ghi:", OUT)


if __name__ == "__main__":
    main()
