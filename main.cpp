// Driver: reads points.bin -> runs voxelVelocityCluster -> writes labels.bin
// points.bin : int32 N, then N * (float x, y, z, v)   (little-endian)
// labels.bin : int32 N, then N * int32 label

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "cluster.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " points.bin labels.bin [res_x res_y res_z vel_thresh]\n";
        return 1;
    }
    ClusterParams p;
    if (argc >= 6) { p.res_x = std::atof(argv[3]); p.res_y = std::atof(argv[4]); p.res_z = std::atof(argv[5]); }
    if (argc >= 7) p.velocity_threshold = std::atof(argv[6]);

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
    int32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || n < 0) { std::cerr << "bad header\n"; return 3; }
    std::vector<Point4D> pts(n);
    in.read(reinterpret_cast<char*>(pts.data()), std::streamsize(n) * sizeof(Point4D));
    if (!in) { std::cerr << "truncated points file\n"; return 3; }

    std::cout << "points: " << n << "  voxel=(" << p.res_x << "," << p.res_y << "," << p.res_z
              << ")  vel_thresh=" << p.velocity_threshold << "\n";

    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> labels = voxelVelocityCluster(pts, p);
    auto t1 = std::chrono::steady_clock::now();

    std::unordered_map<int,int> counts;
    for (int l : labels) counts[l]++;
    int singletons = 0, big = 0;
    for (auto& kv : counts) { if (kv.second == 1) ++singletons; if (kv.second >= 10) ++big; }
    std::cout << "clusters: " << counts.size() << "  (>=10 pts: " << big
              << ", singletons: " << singletons << ")\n";
    std::cout << "time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::ofstream out(argv[2], std::ios::binary);
    int32_t m = (int32_t)labels.size();
    out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    out.write(reinterpret_cast<const char*>(labels.data()), std::streamsize(m) * sizeof(int));
    if (!out) { std::cerr << "write failed\n"; return 4; }
    return 0;
}
