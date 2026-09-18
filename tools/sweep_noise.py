"""노이즈 세기별로 3x3 median 이 이득인지 손해인지 재는 실험.

기본 영상(sigma=3.5)에서는 median 을 끄는 쪽이 나았다. 그렇다고 "median 은 나쁘다"로
결론내면 조건을 빼먹은 주장이 된다. 노이즈를 키워 가며 median 유/무를 같은 정답 위에서
비교해, 뒤집히는 지점을 찾는다.

사용: python tools/sweep_noise.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_binarize import EXE, ROOT, read_boxes, score  # noqa: E402

TMP = ROOT / "out"
# 넓게 훑은 뒤 뒤집히는 구간(16~22)을 1 단위로 좁힌 결과를 그대로 남긴다.
SIGMAS = (3.5, 9.0, 16.0, 17.0, 18.0, 19.0, 22.0)
MIN_AREA, WINDOW, K = 6, 31, 0.20


def gen(sigma):
    pgm = TMP / f"_noise_{sigma:g}.pgm"
    subprocess.run(
        [sys.executable, str(ROOT / "ref" / "make_data.py"), str(pgm), "512", "512", "14", "42",
         str(sigma)],
        check=True, capture_output=True,
    )
    return pgm, Path(str(pgm).rsplit(".", 1)[0] + "_truth.csv")


def run(pgm, denoise):
    out = TMP / "_noise_out.csv"
    r = subprocess.run(
        [str(EXE), str(pgm), str(MIN_AREA), str(out), "sauvola", str(WINDOW), str(K), denoise],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"inspect_core failed (rc={r.returncode}): {r.stderr}")
    return read_boxes(out, has_kind=False)


def main():
    print(f"Sauvola window={WINDOW} k={K} min_area={MIN_AREA} · 결함 14개 · seed 42\n")
    print(f"{'노이즈 sigma':>12} | {'median3 F1':>10} {'검출':>7} {'오검출':>6} | "
          f"{'none F1':>8} {'검출':>7} {'오검출':>6} | 유리한 쪽")
    print("-" * 88)

    for sigma in SIGMAS:
        pgm, truth_path = gen(sigma)
        truth = read_boxes(truth_path, has_kind=True)

        a = score(truth, run(pgm, "median3"))
        b = score(truth, run(pgm, "none"))
        if abs(a["f1"] - b["f1"]) < 1e-9:
            winner = "동률"
        else:
            winner = "median3" if a["f1"] > b["f1"] else "none"

        print(
            f"{sigma:>12.1f} | {a['f1']:>10.3f} {a['detected']:>4}/{a['truth']:<2} "
            f"{a['false_pos']:>6} | {b['f1']:>8.3f} {b['detected']:>4}/{b['truth']:<2} "
            f"{b['false_pos']:>6} | {winner}"
        )

    for f in TMP.glob("_noise_*"):
        f.unlink()
    (TMP / "_noise_out.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
