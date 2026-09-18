"""대조용 합성 검사 영상 생성.

밝은 금속 배경에 어두운 결함(기공·스크래치)을 심고 가우시안 노이즈를 얹는다.
seed 고정이라 몇 번 돌려도 같은 영상이 나온다.

심은 결함의 위치를 정답(GT) CSV로 같이 남긴다. 검출 결과를 눈으로 세지 않고
정밀도·재현율로 재기 위한 것이다. RNG 호출 순서는 GT 추가 전과 동일하게 유지했으므로
같은 seed 에서 영상 바이트는 바뀌지 않는다.
"""

import sys

import numpy as np

def make(w, h, n_defects, seed, noise_sigma=3.5):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 190, dtype=np.float64)

    # 완만한 조명 불균일
    yy, xx = np.mgrid[0:h, 0:w]
    img += 22.0 * np.sin(xx / max(w, 1) * np.pi) * np.cos(yy / max(h, 1) * np.pi)

    truth = []
    for _ in range(n_defects):
        kind = rng.integers(0, 2)
        cy, cx = rng.integers(6, h - 6), rng.integers(6, w - 6)
        if kind == 0:  # 기공: 원형
            r = rng.integers(2, 6)
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        else:  # 스크래치: 가는 선분
            length = rng.integers(8, 26)
            ang = rng.uniform(0, np.pi)
            t = np.arange(length)
            ys = np.clip((cy + t * np.sin(ang)).astype(int), 0, h - 1)
            xs = np.clip((cx + t * np.cos(ang)).astype(int), 0, w - 1)
            m = np.zeros((h, w), dtype=bool)
            m[ys, xs] = True
            m |= np.roll(m, 1, axis=0)
        img[m] -= rng.integers(70, 115)

        gy, gx = np.nonzero(m)
        truth.append(
            {
                "kind": "pore" if kind == 0 else "scratch",
                "area": int(gy.size),
                "x0": int(gx.min()),
                "y0": int(gy.min()),
                "x1": int(gx.max()),
                "y1": int(gy.max()),
                "cx": float(gx.mean()),
                "cy": float(gy.mean()),
            }
        )

    img += rng.normal(0, noise_sigma, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), truth


def write_pgm(path, arr):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"P5\n# synthetic inspection frame\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())


def write_truth(path, truth):
    with open(path, "w") as f:
        f.write("kind,area,x0,y0,x1,y1,cx,cy\n")
        for t in sorted(truth, key=lambda r: (r["y0"], r["x0"])):
            f.write(
                f"{t['kind']},{t['area']},{t['x0']},{t['y0']},{t['x1']},{t['y1']},"
                f"{t['cx']:.4f},{t['cy']:.4f}\n"
            )


if __name__ == "__main__":
    out, w, h, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    sigma = float(sys.argv[6]) if len(sys.argv) > 6 else 3.5  # 생략하면 기존 영상과 동일
    img, truth = make(w, h, n, seed, sigma)
    write_pgm(out, img)
    truth_path = out.rsplit(".", 1)[0] + "_truth.csv"
    write_truth(truth_path, truth)
    print(f"wrote {out} {w}x{h} defects={n} seed={seed}")
    print(f"wrote {truth_path} truth_rows={len(truth)}")
