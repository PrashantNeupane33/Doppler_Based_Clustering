#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <array>

struct Point4D {
    float x, y, z, v;
};

struct ClusterParams {
    float res_x = 0.5f;
    float res_y = 0.5f;
    float res_z = 0.5f;
    float velocity_threshold = 0.8f;
};

struct VoxelKey {
    int32_t ix, iy, iz;
    bool operator==(const VoxelKey& o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        // simple spatial hash combine
        size_t h1 = std::hash<int32_t>()(k.ix);
        size_t h2 = std::hash<int32_t>()(k.iy);
        size_t h3 = std::hash<int32_t>()(k.iz);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL) ^ (h3 * 0xc2b2ae3d27d4eb4fULL);
    }
};

struct VoxelData {
    std::vector<int> point_indices; // indices into original point array
    float v_sum = 0.f;
    int v_count = 0;
    float mean_v() const { return v_count ? v_sum / v_count : 0.f; }
};

// Union-Find
class UnionFind {
public:
    explicit UnionFind(size_t n) : parent(n), rank_(n, 0) {
        for (size_t i = 0; i < n; ++i) parent[i] = (int)i;
    }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        return x;
    }
    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank_[ra] == rank_[rb]) rank_[ra]++;
    }
private:
    std::vector<int> parent;
    std::vector<int> rank_;
};

inline std::vector<int> voxelVelocityCluster(const std::vector<Point4D>& points,
                                              const ClusterParams& params)
{
    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> grid;
    grid.reserve(points.size());

    auto keyOf = [&](const Point4D& p) -> VoxelKey {
        return VoxelKey{
            (int32_t)std::floor(p.x / params.res_x),
            (int32_t)std::floor(p.y / params.res_y),
            (int32_t)std::floor(p.z / params.res_z)
        };
    };

    // 1. Bin points into voxels
    for (int i = 0; i < (int)points.size(); ++i) {
        VoxelKey k = keyOf(points[i]);
        VoxelData& vd = grid[k];
        vd.point_indices.push_back(i);
        vd.v_sum += points[i].v;
        vd.v_count++;
    }

    // 2. Union-Find over POINTS directly.
    //    IMPORTANT: do NOT blindly merge every point in a voxel. If two
    //    different-velocity objects share a voxel (possible when voxel
    //    resolution is coarse relative to object spacing), merge only
    //    points within velocity_threshold of each other, even within the
    //    same voxel. Otherwise mean_v gets blurred and leaks across objects.
    UnionFind uf(points.size());

    for (auto& [key, vd] : grid) {
        const auto& idxs = vd.point_indices;
        for (size_t i = 0; i < idxs.size(); ++i) {
            for (size_t j = i + 1; j < idxs.size(); ++j) {
                if (std::fabs(points[idxs[i]].v - points[idxs[j]].v) <= params.velocity_threshold)
                    uf.unite(idxs[i], idxs[j]);
            }
        }
    }

    // 26-neighborhood offsets
    static const std::array<std::array<int,3>,26> offsets = {{
        {-1,-1,-1},{-1,-1,0},{-1,-1,1},
        {-1,0,-1}, {-1,0,0}, {-1,0,1},
        {-1,1,-1}, {-1,1,0}, {-1,1,1},
        {0,-1,-1}, {0,-1,0}, {0,-1,1},
        {0,0,-1},             {0,0,1},
        {0,1,-1},  {0,1,0},  {0,1,1},
        {1,-1,-1}, {1,-1,0}, {1,-1,1},
        {1,0,-1},  {1,0,0},  {1,0,1},
        {1,1,-1},  {1,1,0},  {1,1,1}
    }};

    // For neighbor merging, a voxel may now contain multiple distinct
    // velocity-subgroups (different UF roots). Get one representative point
    // per distinct root present in the voxel, then compare representatives
    // across neighboring voxels pairwise (not blended means).
    auto representatives = [&](const VoxelData& vd) {
        std::vector<int> reps;
        std::unordered_map<int,int> seen_root; // root -> rep point index
        for (int idx : vd.point_indices) {
            int r = uf.find(idx);
            if (seen_root.find(r) == seen_root.end()) {
                seen_root[r] = idx;
                reps.push_back(idx);
            }
        }
        return reps;
    };

    for (auto& [key, vd] : grid) {
        std::vector<int> repsA = representatives(vd);
        for (const auto& off : offsets) {
            VoxelKey nk{key.ix + off[0], key.iy + off[1], key.iz + off[2]};
            auto it = grid.find(nk);
            if (it == grid.end()) continue;
            // avoid double work: only process each pair once via key ordering
            if (!(nk.ix > key.ix || (nk.ix == key.ix && nk.iy > key.iy) ||
                  (nk.ix == key.ix && nk.iy == key.iy && nk.iz > key.iz)))
                continue;
            std::vector<int> repsB = representatives(it->second);
            for (int ra : repsA) {
                for (int rb : repsB) {
                    if (std::fabs(points[ra].v - points[rb].v) <= params.velocity_threshold)
                        uf.unite(ra, rb);
                }
            }
        }
    }

    // 3. Compact labels
    std::vector<int> raw_label(points.size());
    for (int i = 0; i < (int)points.size(); ++i) raw_label[i] = uf.find(i);

    std::unordered_map<int,int> remap;
    std::vector<int> labels(points.size());
    int next_id = 0;
    for (int i = 0; i < (int)points.size(); ++i) {
        int r = raw_label[i];
        auto it = remap.find(r);
        if (it == remap.end()) {
            remap[r] = next_id;
            labels[i] = next_id;
            next_id++;
        } else {
            labels[i] = it->second;
        }
    }
    return labels;
}
