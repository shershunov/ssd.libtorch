#include "metrics.h"
#include "anchors.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <unordered_map>

namespace {

struct Det {
    int img_id;
    int cls;
    float conf;
    std::array<float, 4> box;
};

struct GT {
    int img_id;
    int cls;
    std::array<float, 4> box;
};

float scalar_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6f);
}

float compute_ap(const std::vector<float>& precs, const std::vector<float>& recs) {
    if (precs.empty()) return 0.0f;

    std::vector<float> env(precs);
    for (int i = (int)env.size() - 2; i >= 0; --i) {
        env[i] = std::max(env[i], env[i + 1]);
    }

    float ap = 0.0f;
    for (int t = 0; t <= 100; ++t) {
        float r = t / 100.0f;
        auto it = std::lower_bound(recs.begin(), recs.end(), r);
        if (it != recs.end()) {
            ap += env[it - recs.begin()];
        }
    }
    return ap / 101.0f;
}

MapResult compute_map(const std::vector<Det>& dets, const std::vector<GT>& gts, int num_classes) {
    constexpr int NT = 10;
    float thresholds[NT];
    for (int i = 0; i < NT; ++i) thresholds[i] = 0.5f + i * 0.05f;

    float sum_ap_50 = 0.0f;
    float sum_ap_all = 0.0f;
    int valid_classes = 0;

    for (int c = 1; c < num_classes; ++c) {
        std::vector<const Det*> cd;
        cd.reserve(dets.size() / std::max(num_classes - 1, 1));
        for (const auto& d : dets) {
            if (d.cls == c) cd.push_back(&d);
        }
        std::sort(cd.begin(), cd.end(),
                  [](const Det* a, const Det* b) { return a->conf > b->conf; });

        std::unordered_map<int, std::vector<const GT*>> cg;
        int total_gt = 0;
        for (const auto& g : gts) {
            if (g.cls == c) {
                cg[g.img_id].push_back(&g);
                ++total_gt;
            }
        }

        if (total_gt == 0) continue;
        ++valid_classes;

        for (int t = 0; t < NT; ++t) {
            std::unordered_map<int, std::vector<bool>> matched;
            for (auto& [id, gs] : cg) matched[id].assign(gs.size(), false);

            std::vector<float> precs, recs;
            precs.reserve(cd.size());
            recs.reserve(cd.size());
            int tp = 0, fp = 0;

            for (const auto* det : cd) {
                bool found = false;
                auto it = cg.find(det->img_id);
                if (it != cg.end()) {
                    float best_iou = 0;
                    int best_idx = -1;
                    auto& img_gts = it->second;
                    auto& img_m = matched[det->img_id];
                    for (size_t g = 0; g < img_gts.size(); ++g) {
                        if (img_m[g]) continue;
                        float iv = scalar_iou(det->box, img_gts[g]->box);
                        if (iv > best_iou) { best_iou = iv; best_idx = (int)g; }
                    }
                    if (best_iou >= thresholds[t] && best_idx >= 0) {
                        img_m[best_idx] = true;
                        ++tp;
                        found = true;
                    }
                }
                if (!found) ++fp;
                precs.push_back((float)tp / (tp + fp));
                recs.push_back((float)tp / total_gt);
            }

            float ap = compute_ap(precs, recs);
            if (t == 0) sum_ap_50 += ap;
            sum_ap_all += ap;
        }
    }

    if (valid_classes == 0) return {0.0f, 0.0f};
    return {sum_ap_50 / valid_classes, sum_ap_all / (valid_classes * NT)};
}

} // namespace

MapResult validate(Net& model, torch::Device& device,
                   std::vector<torch::Tensor>& images,
                   std::vector<torch::Tensor>& targets,
                   int num_classes, int batch_size,
                   const torch::Tensor& anchors) {
    if (images.empty()) return {0.0f, 0.0f};

    torch::NoGradGuard no_grad;
    model.eval();

    auto anchors_dev = anchors.to(device);

    std::vector<Det> all_dets;
    std::vector<GT> all_gts;

    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].size(0) == 0) continue;
        auto t = targets[i];
        auto bxy = xywh_to_xyxy(t.slice(1, 0, 4));
        auto cls = t.select(1, 4);
        auto ba = bxy.accessor<float, 2>();
        auto ca = cls.accessor<float, 1>();
        for (int j = 0; j < (int)t.size(0); ++j) {
            all_gts.push_back({(int)i, (int)ca[j], {ba[j][0], ba[j][1], ba[j][2], ba[j][3]}});
        }
    }

    constexpr float CONF_THRESH = 0.001f;
    constexpr float NMS_THRESH = 0.65f;
    constexpr int MAX_DET = 300;

    for (size_t start = 0; start < images.size(); start += batch_size) {
        size_t end = std::min(start + (size_t)batch_size, images.size());
        std::vector<torch::Tensor> batch_imgs;
        batch_imgs.reserve(end - start);
        for (size_t i = start; i < end; ++i) batch_imgs.push_back(images[i]);
        auto input = torch::stack(batch_imgs).to(device);

        auto [offsets, scores] = model.forward(input);
        auto decoded_boxes = decode_boxes(offsets, anchors_dev);
        auto probs = torch::softmax(scores, 2);

        for (size_t b = 0; b < end - start; ++b) {
            int img_id = (int)(start + b);
            auto pb = decoded_boxes[(int64_t)b];
            auto pp = probs[(int64_t)b];

            auto fg = pp.slice(1, 1);
            auto [ms, mc] = fg.max(1);

            auto mask = ms > CONF_THRESH;
            int cnt = mask.sum().item<int>();
            if (cnt == 0) continue;

            auto fb = pb.index({mask});
            auto fs = ms.index({mask});
            auto fc = mc.index({mask});

            if (cnt > MAX_DET) {
                auto [topk_vals, topk_idx] = fs.topk(MAX_DET);
                fb = fb.index_select(0, topk_idx);
                fs = topk_vals;
                fc = fc.index_select(0, topk_idx);
                cnt = MAX_DET;
            }

            auto fb_xyxy = xywh_to_xyxy(fb);
            auto offsets_nms = fc.to(torch::kFloat32) * 4096.0f;
            auto ob = fb_xyxy + offsets_nms.unsqueeze(1);

            auto [ss, si] = fs.sort(0, true);
            auto cpu_ob = ob.index_select(0, si).cpu();
            auto cpu_s = ss.cpu();
            auto cpu_c = fc.index_select(0, si).cpu();
            auto cpu_b = fb_xyxy.index_select(0, si).cpu();

            int n = (int)cpu_b.size(0);
            std::vector<bool> sup(n, false);
            auto oa = cpu_ob.accessor<float, 2>();

            for (int i = 0; i < n; ++i) {
                if (sup[i]) continue;
                for (int j = i + 1; j < n; ++j) {
                    if (sup[j]) continue;
                    float x1 = std::max(oa[i][0], oa[j][0]);
                    float y1 = std::max(oa[i][1], oa[j][1]);
                    float x2 = std::min(oa[i][2], oa[j][2]);
                    float y2 = std::min(oa[i][3], oa[j][3]);
                    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
                    float ai = (oa[i][2] - oa[i][0]) * (oa[i][3] - oa[i][1]);
                    float aj = (oa[j][2] - oa[j][0]) * (oa[j][3] - oa[j][1]);
                    if (inter / (ai + aj - inter + 1e-6f) > NMS_THRESH) sup[j] = true;
                }
            }

            auto sa = cpu_s.accessor<float, 1>();
            auto ca = cpu_c.accessor<int64_t, 1>();
            auto ba = cpu_b.accessor<float, 2>();

            for (int i = 0; i < n; ++i) {
                if (sup[i]) continue;
                all_dets.push_back({img_id, (int)ca[i] + 1, sa[i],
                                    {ba[i][0], ba[i][1], ba[i][2], ba[i][3]}});
            }
        }
    }

    model.train();
    return compute_map(all_dets, all_gts, num_classes);
}
