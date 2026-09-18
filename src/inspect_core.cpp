// inspect_core.cpp — 검사 전처리 파이프라인 (C++17, 의존성 없음)
//
// 파이프라인: PGM(P5) 읽기 -> 3x3 median -> 이진화(Otsu 전역 | Sauvola 국소)
//             -> 8-연결 연결요소 라벨링 -> blob 통계(면적/bbox/무게중심)
//             -> 면적 하한으로 결함 후보 선별 -> CSV
//
// 목적: 같은 입력에 대해 Python(numpy/scipy) 기준 구현과 결과가 일치하는지 대조한다.
//       처리시간도 같이 찍어 두 구현의 처리량을 비교한다.
//
// 빌드: g++ -std=c++17 -O2 -Wall -Wextra -static -o inspect_core src/inspect_core.cpp
//       (-static 는 MinGW 런타임 DLL 의존을 없애기 위한 것. 없으면 PATH 에 mingw64/bin 이
//        빠진 셸에서 실행이 죽는다.)
// 실행: ./inspect_core <input.pgm> <min_area> <out.csv> [otsu|sauvola] [window] [k]
//                      [median3|none]

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
    inline uint8_t at(int y, int x) const { return px[static_cast<size_t>(y) * w + x]; }
    inline uint8_t& at(int y, int x) { return px[static_cast<size_t>(y) * w + x]; }
};

// --- PGM(P5) 읽기. 주석(#) 줄을 건너뛴다. -----------------------------------
bool readPGM(const std::string& path, Image& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    auto nextToken = [&f]() -> std::string {
        std::string tok;
        int c;
        while ((c = f.get()) != EOF) {
            if (c == '#') {                       // 주석은 줄 끝까지 버린다
                while ((c = f.get()) != EOF && c != '\n') {}
                continue;
            }
            if (std::isspace(c)) {
                if (!tok.empty()) break;
                continue;
            }
            tok.push_back(static_cast<char>(c));
        }
        return tok;
    };

    if (nextToken() != "P5") return false;
    img.w = std::stoi(nextToken());
    img.h = std::stoi(nextToken());
    const int maxval = std::stoi(nextToken());
    if (img.w <= 0 || img.h <= 0 || maxval != 255) return false;

    img.px.resize(static_cast<size_t>(img.w) * img.h);
    f.read(reinterpret_cast<char*>(img.px.data()), static_cast<std::streamsize>(img.px.size()));
    return f.gcount() == static_cast<std::streamsize>(img.px.size());
}

// --- 3x3 median. 테두리는 clamp 로 확장한다. --------------------------------
Image median3(const Image& src) {
    Image dst;
    dst.w = src.w;
    dst.h = src.h;
    dst.px.resize(src.px.size());

    std::array<uint8_t, 9> win{};
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = std::clamp(y + dy, 0, src.h - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::clamp(x + dx, 0, src.w - 1);
                    win[n++] = src.at(yy, xx);
                }
            }
            std::nth_element(win.begin(), win.begin() + 4, win.end());
            dst.at(y, x) = win[4];
        }
    }
    return dst;
}

// --- Otsu 임계값. 클래스 간 분산이 최대가 되는 t 를 고른다. ------------------
// 동점이면 가장 작은 t 를 쓴다(파이썬 기준 구현과 동일 규칙).
int otsuThreshold(const Image& img) {
    std::array<double, 256> hist{};
    for (uint8_t v : img.px) hist[v] += 1.0;

    const double total = static_cast<double>(img.px.size());
    double sumAll = 0.0;
    for (int i = 0; i < 256; ++i) sumAll += i * hist[i];

    double sumB = 0.0, wB = 0.0, best = -1.0;
    int bestT = 0;
    for (int t = 0; t < 256; ++t) {
        wB += hist[t];
        if (wB == 0.0) continue;
        const double wF = total - wB;
        if (wF == 0.0) break;

        sumB += static_cast<double>(t) * hist[t];
        const double mB = sumB / wB;
        const double mF = (sumAll - sumB) / wF;
        const double between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) {
            best = between;
            bestT = t;
        }
    }
    return bestT;
}

// --- Sauvola 국소 적응 임계값 ----------------------------------------------
// T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1)), R = 128.
// 창 안의 평균 m 과 표준편차 s 를 적분영상으로 O(1)에 구한다. 창은 이미지 경계에서
// 잘라 쓰고(clamp), 잘린 만큼 픽셀 수를 다시 세어 평균을 낸다.
// 전역 Otsu 와 달리 조명 기울기를 배경으로 흡수하므로 그림자 쪽 배경이 전경으로
// 통째로 넘어가지 않는다.
std::vector<uint8_t> sauvolaBinarize(const Image& img, int window, double k) {
    const int w = img.w, h = img.h;
    const int half = (window - 1) / 2;
    const double R = 128.0;

    // 적분영상 (h+1) x (w+1). 합은 int64 로 누적해 정밀도 손실을 없앤다.
    const size_t iw = static_cast<size_t>(w) + 1;
    std::vector<int64_t> sat(iw * (static_cast<size_t>(h) + 1), 0);
    std::vector<int64_t> sat2(iw * (static_cast<size_t>(h) + 1), 0);
    for (int y = 0; y < h; ++y) {
        int64_t rowSum = 0, rowSum2 = 0;
        for (int x = 0; x < w; ++x) {
            const int64_t v = img.at(y, x);
            rowSum += v;
            rowSum2 += v * v;
            sat[static_cast<size_t>(y + 1) * iw + (x + 1)] =
                sat[static_cast<size_t>(y) * iw + (x + 1)] + rowSum;
            sat2[static_cast<size_t>(y + 1) * iw + (x + 1)] =
                sat2[static_cast<size_t>(y) * iw + (x + 1)] + rowSum2;
        }
    }
    auto boxSum = [&](const std::vector<int64_t>& s, int y0, int x0, int y1, int x1) -> int64_t {
        return s[static_cast<size_t>(y1 + 1) * iw + (x1 + 1)] -
               s[static_cast<size_t>(y0) * iw + (x1 + 1)] -
               s[static_cast<size_t>(y1 + 1) * iw + x0] + s[static_cast<size_t>(y0) * iw + x0];
    };

    std::vector<uint8_t> bin(static_cast<size_t>(w) * h, 0);
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - half), y1 = std::min(h - 1, y + half);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - half), x1 = std::min(w - 1, x + half);
            const double cnt = static_cast<double>((y1 - y0 + 1)) * (x1 - x0 + 1);

            const double mean = static_cast<double>(boxSum(sat, y0, x0, y1, x1)) / cnt;
            const double meanSq = static_cast<double>(boxSum(sat2, y0, x0, y1, x1)) / cnt;
            double var = meanSq - mean * mean;
            if (var < 0.0) var = 0.0;  // 부동소수 오차로 음수가 나오는 경우
            const double sd = std::sqrt(var);

            const double T = mean * (1.0 + k * (sd / R - 1.0));
            bin[static_cast<size_t>(y) * w + x] = (static_cast<double>(img.at(y, x)) <= T) ? 1 : 0;
        }
    }
    return bin;
}

// --- union-find ------------------------------------------------------------
struct DSU {
    std::vector<int> parent;
    int add() {
        parent.push_back(static_cast<int>(parent.size()));
        return static_cast<int>(parent.size()) - 1;
    }
    int find(int a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);  // 작은 쪽을 대표로
    }
};

struct Blob {
    long long area = 0;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    double cx = 0.0, cy = 0.0;
};

// 8-연결 연결요소 라벨링(2-pass). 전경 = 값이 1 인 픽셀.
std::vector<Blob> labelBlobs(const std::vector<uint8_t>& bin, int w, int h) {
    std::vector<int> lab(static_cast<size_t>(w) * h, -1);
    DSU dsu;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (!bin[idx]) continue;

            int best = -1;
            // 이미 방문한 이웃 4개: (y-1,x-1) (y-1,x) (y-1,x+1) (y,x-1)
            const int nb[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};
            for (auto& d : nb) {
                const int yy = y + d[0], xx = x + d[1];
                if (yy < 0 || xx < 0 || xx >= w) continue;
                const int l = lab[static_cast<size_t>(yy) * w + xx];
                if (l < 0) continue;
                best = (best < 0) ? l : std::min(best, l);
            }
            if (best < 0) {
                lab[idx] = dsu.add();
            } else {
                lab[idx] = best;
                for (auto& d : nb) {
                    const int yy = y + d[0], xx = x + d[1];
                    if (yy < 0 || xx < 0 || xx >= w) continue;
                    const int l = lab[static_cast<size_t>(yy) * w + xx];
                    if (l >= 0) dsu.unite(best, l);
                }
            }
        }
    }

    // 대표 라벨을 좌상단 우선 순서로 다시 매긴다(구현 간 라벨 번호를 맞추기 위함).
    std::vector<int> remap(dsu.parent.size(), -1);
    std::vector<Blob> blobs;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (lab[idx] < 0) continue;
            const int root = dsu.find(lab[idx]);
            int id = remap[root];
            if (id < 0) {
                id = static_cast<int>(blobs.size());
                remap[root] = id;
                Blob b;
                b.x0 = b.x1 = x;
                b.y0 = b.y1 = y;
                blobs.push_back(b);
            }
            Blob& b = blobs[id];
            b.area += 1;
            b.x0 = std::min(b.x0, x);
            b.x1 = std::max(b.x1, x);
            b.y0 = std::min(b.y0, y);
            b.y1 = std::max(b.y1, y);
            b.cx += x;
            b.cy += y;
        }
    }
    for (Blob& b : blobs) {
        b.cx /= static_cast<double>(b.area);
        b.cy /= static_cast<double>(b.area);
    }
    return blobs;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: inspect_core <input.pgm> <min_area> <out.csv> "
                     "[otsu|sauvola] [window] [k]\n";
        return 2;
    }
    const std::string inPath = argv[1];
    const long long minArea = std::stoll(argv[2]);
    const std::string outPath = argv[3];
    const std::string method = (argc > 4) ? argv[4] : "otsu";
    const int window = (argc > 5) ? std::stoi(argv[5]) : 31;
    const double kParam = (argc > 6) ? std::stod(argv[6]) : 0.20;
    const std::string denoise = (argc > 7) ? argv[7] : "median3";

    if (method != "otsu" && method != "sauvola") {
        std::cerr << "unknown method: " << method << " (otsu|sauvola)\n";
        return 2;
    }
    if (denoise != "median3" && denoise != "none") {
        std::cerr << "unknown denoise: " << denoise << " (median3|none)\n";
        return 2;
    }
    if (method == "sauvola" && (window < 3 || window % 2 == 0)) {
        std::cerr << "window must be an odd number >= 3\n";
        return 2;
    }

    Image img;
    if (!readPGM(inPath, img)) {
        std::cerr << "failed to read PGM: " << inPath << "\n";
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // 3x3 median 은 가는 결함을 지워 버린다(폭 1~2px 스크래치의 대비가 무너진다).
    // 끌 수 있게 열어 두고 어느 쪽이 나은지는 정답 대비로 측정한다.
    const Image med = (denoise == "median3") ? median3(img) : img;

    // 결함은 배경보다 어둡다고 본다: 임계값 이하를 전경으로 잡는다.
    int t = -1;
    std::vector<uint8_t> bin;
    if (method == "otsu") {
        t = otsuThreshold(med);
        bin.resize(med.px.size());
        for (size_t i = 0; i < med.px.size(); ++i) bin[i] = (med.px[i] <= t) ? 1 : 0;
    } else {
        bin = sauvolaBinarize(med, window, kParam);
    }

    std::vector<Blob> blobs = labelBlobs(bin, med.w, med.h);

    std::vector<Blob> kept;
    for (const Blob& b : blobs)
        if (b.area >= minArea) kept.push_back(b);

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::ofstream out(outPath);
    out << "area,x0,y0,x1,y1,cx,cy\n";
    out.setf(std::ios::fixed);
    out.precision(4);
    for (const Blob& b : kept)
        out << b.area << ',' << b.x0 << ',' << b.y0 << ',' << b.x1 << ',' << b.y1 << ','
            << b.cx << ',' << b.cy << '\n';

    if (method == "otsu")
        std::printf("method=otsu denoise=%s otsu_threshold=%d blobs_total=%zu blobs_kept=%zu "
                    "elapsed_ms=%.3f\n",
                    denoise.c_str(), t, blobs.size(), kept.size(), ms);
    else
        std::printf("method=sauvola denoise=%s window=%d k=%.2f blobs_total=%zu blobs_kept=%zu "
                    "elapsed_ms=%.3f\n",
                    denoise.c_str(), window, kParam, blobs.size(), kept.size(), ms);
    return 0;
}
