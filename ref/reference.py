"""기준 구현 — 같은 파이프라인을 numpy/scipy로 짜서 C++ 결과와 대조한다.

라벨링은 직접 짜지 않고 scipy.ndimage.label을 쓴다. 같은 알고리즘을 두 번 옮겨 적으면
같은 실수를 두 번 하게 되므로, 라벨링만은 독립적인 구현을 기준으로 삼는다.

사용: python ref/reference.py <input.pgm> <min_area> <out.csv> [otsu|sauvola] [window] [k]
"""

import sys
import time

import numpy as np
from scipy import ndimage


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()

    # 헤더 토큰 4개(P5, w, h, maxval)를 읽는다. 주석(#)은 줄 끝까지 버린다.
    tokens, i = [], 0
    while len(tokens) < 4:
        if data[i : i + 1] == b"#":
            while data[i : i + 1] not in (b"\n", b""):
                i += 1
            continue
        if data[i : i + 1].isspace():
            i += 1
            continue
        j = i
        while not data[j : j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    i += 1  # maxval 뒤의 공백 한 칸

    assert tokens[0] == b"P5", tokens[0]
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    assert maxval == 255
    px = np.frombuffer(data[i : i + w * h], dtype=np.uint8).reshape(h, w)
    return px


def median3(img):
    # 테두리는 clamp 확장 — C++ 구현과 같은 규칙
    pad = np.pad(img, 1, mode="edge")
    win = np.stack(
        [pad[dy : dy + img.shape[0], dx : dx + img.shape[1]] for dy in range(3) for dx in range(3)],
        axis=0,
    )
    return np.median(win, axis=0).astype(np.uint8)


def otsu(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)

    w_b = np.cumsum(hist)
    sum_b = np.cumsum(np.arange(256) * hist)
    w_f = total - w_b

    valid = (w_b > 0) & (w_f > 0)
    m_b = np.divide(sum_b, w_b, out=np.zeros(256), where=w_b > 0)
    m_f = np.divide(sum_all - sum_b, w_f, out=np.zeros(256), where=w_f > 0)
    between = np.where(valid, w_b * w_f * (m_b - m_f) ** 2, -1.0)
    return int(np.argmax(between))  # 동점이면 가장 작은 t (argmax 기본 동작)


def sauvola(img, window, k):
    """T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1)), R = 128.

    C++ 구현과 연산 순서를 그대로 맞춘다 — 적분영상은 int64로 누적하고,
    분산은 E[x^2] - E[x]^2 로 구한 뒤 음수를 0으로 자른다. 창은 경계에서 잘라 쓰고
    잘린 만큼 픽셀 수를 다시 센다.
    """
    h, w = img.shape
    half = (window - 1) // 2
    R = 128.0

    v = img.astype(np.int64)
    sat = np.zeros((h + 1, w + 1), dtype=np.int64)
    sat2 = np.zeros((h + 1, w + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(v, axis=0), axis=1)
    sat2[1:, 1:] = np.cumsum(np.cumsum(v * v, axis=0), axis=1)

    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.maximum(0, ys - half)
    y1 = np.minimum(h - 1, ys + half)
    x0 = np.maximum(0, xs - half)
    x1 = np.minimum(w - 1, xs + half)

    def box(s):
        return (
            s[np.ix_(y1 + 1, x1 + 1)]
            - s[np.ix_(y0, x1 + 1)]
            - s[np.ix_(y1 + 1, x0)]
            + s[np.ix_(y0, x0)]
        )

    cnt = ((y1 - y0 + 1)[:, None] * (x1 - x0 + 1)[None, :]).astype(np.float64)
    mean = box(sat).astype(np.float64) / cnt
    mean_sq = box(sat2).astype(np.float64) / cnt
    var = np.maximum(mean_sq - mean * mean, 0.0)
    sd = np.sqrt(var)

    thr = mean * (1.0 + k * (sd / R - 1.0))
    return img.astype(np.float64) <= thr


def main():
    in_path, min_area, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    method = sys.argv[4] if len(sys.argv) > 4 else "otsu"
    window = int(sys.argv[5]) if len(sys.argv) > 5 else 31
    k = float(sys.argv[6]) if len(sys.argv) > 6 else 0.20
    denoise = sys.argv[7] if len(sys.argv) > 7 else "median3"
    assert method in ("otsu", "sauvola"), method
    assert denoise in ("median3", "none"), denoise

    img = read_pgm(in_path)

    t0 = time.perf_counter()

    med = median3(img) if denoise == "median3" else img
    # 결함은 배경보다 어둡다
    if method == "otsu":
        t = otsu(med)
        binary = med <= t
    else:
        t = -1
        binary = sauvola(med, window, k)

    structure = np.ones((3, 3), dtype=bool)  # 8-연결
    lab, n = ndimage.label(binary, structure=structure)

    rows = []
    for idx in range(1, n + 1):
        ys, xs = np.nonzero(lab == idx)
        area = int(ys.size)
        if area < min_area:
            continue
        rows.append(
            (area, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), xs.mean(), ys.mean())
        )

    # C++ 구현은 좌상단 우선(행 우선 첫 등장) 순으로 blob을 낸다. 같은 순서로 맞춘다.
    rows.sort(key=lambda r: (r[2], r[1]))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    with open(out_path, "w") as f:
        f.write("area,x0,y0,x1,y1,cx,cy\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")

    if method == "otsu":
        print(
            f"method=otsu denoise={denoise} otsu_threshold={t} blobs_total={n} "
            f"blobs_kept={len(rows)} elapsed_ms={elapsed_ms:.3f}"
        )
    else:
        print(
            f"method=sauvola denoise={denoise} window={window} k={k:.2f} blobs_total={n} "
            f"blobs_kept={len(rows)} elapsed_ms={elapsed_ms:.3f}"
        )


if __name__ == "__main__":
    main()
