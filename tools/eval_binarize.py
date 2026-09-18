"""이진화 방식별 결함 검출 성능 측정 — Otsu(전역) vs Sauvola(국소 적응).

`inspect_core.exe`가 낸 후보 CSV를 `make_data.py`가 남긴 정답 CSV와 맞춰
정밀도·재현율·F1을 낸다. 후보 개수를 눈으로 세는 대신 정답 기준으로 재기 위한 것이다.

매칭 규칙: 후보 bbox와 정답 bbox가 겹치면 그 정답은 검출된 것으로 본다.
어느 정답과도 겹치지 않는 후보는 오검출로 센다. 결함이 서로 붙어 한 덩어리로
잡히는 경우가 있어 IoU 대신 겹침 여부를 쓴다 — 이 데이터에서는 위치를 맞혔는지가
판단 기준이고, 경계를 픽셀 단위로 맞히는 것이 목적이 아니다.

사용:
  python tools/eval_binarize.py data/frame_512_truth.csv out/cpp.csv out/cpp_sauvola.csv
  python tools/eval_binarize.py --sweep data/frame_512.pgm
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "inspect_core.exe"


def read_boxes(path, has_kind):
    boxes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            boxes.append(
                (int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"]), int(row["area"]))
            )
    return boxes


def overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def score(truth, cand):
    hit = [False] * len(truth)
    tp_cand = 0
    for c in cand:
        matched = False
        for i, t in enumerate(truth):
            if overlaps(c, t):
                hit[i] = True
                matched = True
        if matched:
            tp_cand += 1

    recall = sum(hit) / len(truth) if truth else 0.0
    precision = tp_cand / len(cand) if cand else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "candidates": len(cand),
        "detected": sum(hit),
        "truth": len(truth),
        "false_pos": len(cand) - tp_cand,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fmt(name, s):
    return (
        f"{name:<22} 후보 {s['candidates']:>4}  검출 {s['detected']:>2}/{s['truth']:<2}  "
        f"오검출 {s['false_pos']:>4}  정밀도 {s['precision']:.3f}  "
        f"재현율 {s['recall']:.3f}  F1 {s['f1']:.3f}"
    )


def run(pgm, min_area, out, method, window=31, k=0.20, denoise="median3"):
    cmd = [str(EXE), str(pgm), str(min_area), str(out), method, str(window), str(k), denoise]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"inspect_core failed (rc={r.returncode}): {r.stderr}")
    return r.stdout.strip()


def sweep(pgm):
    """파라미터를 눈대중으로 고르지 않기 위한 탐색.

    같은 정답 위에서 이진화 방식·면적 하한·Sauvola 계수·노이즈 제거 여부를
    격자로 돌려 F1 을 본다.
    """
    truth_path = Path(str(pgm).rsplit(".", 1)[0] + "_truth.csv")
    truth = read_boxes(truth_path, has_kind=True)
    tmp = ROOT / "out" / "_sweep.csv"

    print(f"정답 결함 {len(truth)}개 · {pgm}\n")
    print("=== Otsu (전역 임계) ===")
    for min_area in (4, 6, 10, 20):
        run(pgm, min_area, tmp, "otsu")
        print(fmt(f"otsu min_area={min_area}", score(truth, read_boxes(tmp, has_kind=False))))

    print("\n=== Sauvola (국소 적응, window=31, median3) ===")
    best = None
    for min_area in (4, 6, 10, 20):
        for k in (0.10, 0.20, 0.30, 0.40):
            run(pgm, min_area, tmp, "sauvola", 31, k)
            s = score(truth, read_boxes(tmp, has_kind=False))
            print(fmt(f"sauvola m={min_area} k={k:.2f}", s))
            if best is None or s["f1"] > best[1]["f1"]:
                best = (f"min_area={min_area} k={k:.2f} denoise=median3", s)

    print("\n=== 창 크기 (min_area=6, k=0.20) ===")
    for window in (15, 31, 51, 101):
        run(pgm, 6, tmp, "sauvola", window, 0.20)
        print(fmt(f"sauvola window={window}", score(truth, read_boxes(tmp, has_kind=False))))

    # 3x3 median 은 폭 1~2px 스크래치를 지운다. 노이즈 제거를 끄면 그 결함이 살아나지만
    # 노이즈가 그대로 남으므로, 어느 쪽이 이득인지는 정답 대비로만 판단할 수 있다.
    print("\n=== 노이즈 제거 유/무 (window=31) ===")
    for denoise in ("median3", "none"):
        for min_area in (4, 6, 10):
            for k in (0.10, 0.20, 0.30):
                run(pgm, min_area, tmp, "sauvola", 31, k, denoise)
                s = score(truth, read_boxes(tmp, has_kind=False))
                print(fmt(f"{denoise:<8} m={min_area} k={k:.2f}", s))
                if s["f1"] > best[1]["f1"]:
                    best = (f"min_area={min_area} k={k:.2f} denoise={denoise}", s)

    print(f"\n최고 F1: {best[0]} → F1 {best[1]['f1']:.3f} "
          f"(정밀도 {best[1]['precision']:.3f} 재현율 {best[1]['recall']:.3f})")
    tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", metavar="PGM", help="파라미터 격자 탐색")
    ap.add_argument("args", nargs="*", help="<truth.csv> <otsu.csv> <sauvola.csv>")
    a = ap.parse_args()

    if a.sweep:
        sweep(Path(a.sweep))
        return

    if len(a.args) != 3:
        ap.error("truth.csv otsu.csv sauvola.csv 세 개가 필요하다")
    truth = read_boxes(a.args[0], has_kind=True)
    print(f"정답 결함 {len(truth)}개\n")
    print(fmt("Otsu(전역)", score(truth, read_boxes(a.args[1], has_kind=False))))
    print(fmt("Sauvola(국소)", score(truth, read_boxes(a.args[2], has_kind=False))))


if __name__ == "__main__":
    main()
