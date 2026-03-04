#include "anchors.h"

torch::Tensor generate_anchors(int img_size, int num_anchors_per_cell) {
    struct Level {
        int stride;
        std::vector<float> sizes;
    };

    std::vector<Level> levels = {
        {8,  {16.0f, 32.0f}},
        {16, {64.0f, 128.0f}},
        {32, {256.0f, 512.0f}}
    };

    std::vector<torch::Tensor> all_anchors;

    for (const auto& level : levels) {
        int grid_size = img_size / level.stride;

        auto grid_y = torch::arange(grid_size, torch::kFloat32);
        auto grid_x = torch::arange(grid_size, torch::kFloat32);
        auto grids = torch::meshgrid({grid_y, grid_x}, "ij");
        auto cy = (grids[0] + 0.5f) * level.stride / static_cast<float>(img_size);
        auto cx = (grids[1] + 0.5f) * level.stride / static_cast<float>(img_size);

        std::vector<torch::Tensor> per_pos;
        for (float size : level.sizes) {
            float w = size / static_cast<float>(img_size);
            float h = size / static_cast<float>(img_size);
            per_pos.push_back(torch::stack({cx, cy, torch::full_like(cx, w), torch::full_like(cy, h)}, -1));
        }

        auto stacked = torch::stack(per_pos, 2);
        all_anchors.push_back(stacked.reshape({-1, 4}));
    }

    return torch::cat(all_anchors, 0);
}

torch::Tensor decode_boxes(const torch::Tensor& offsets, const torch::Tensor& anchors) {
    auto tx = offsets.select(-1, 0);
    auto ty = offsets.select(-1, 1);
    auto tw = offsets.select(-1, 2);
    auto th = offsets.select(-1, 3);

    auto ax = anchors.select(-1, 0);
    auto ay = anchors.select(-1, 1);
    auto aw = anchors.select(-1, 2);
    auto ah = anchors.select(-1, 3);

    auto gx = torch::clamp(tx * 0.1f * aw + ax, 0.0f, 1.0f);
    auto gy = torch::clamp(ty * 0.1f * ah + ay, 0.0f, 1.0f);
    auto gw = torch::exp(torch::clamp(tw * 0.2f, -10.0f, 10.0f)) * aw;
    auto gh = torch::exp(torch::clamp(th * 0.2f, -10.0f, 10.0f)) * ah;

    return torch::stack({gx, gy, gw, gh}, -1);
}
