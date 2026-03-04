#include "loss.h"

namespace {

torch::Tensor batch_xywh_to_xyxy(const torch::Tensor& boxes) {
    auto x = boxes.select(-1, 0);
    auto y = boxes.select(-1, 1);
    auto hw = boxes.select(-1, 2) * 0.5f;
    auto hh = boxes.select(-1, 3) * 0.5f;
    return torch::stack({ x - hw, y - hh, x + hw, y + hh }, -1);
}

torch::Tensor batch_iou(const torch::Tensor& a, const torch::Tensor& b) {
    auto a_exp = a.unsqueeze(2);
    auto b_exp = b.unsqueeze(1);

    auto inter_w = torch::clamp_min(
        torch::min(a_exp.select(-1, 2), b_exp.select(-1, 2)) -
        torch::max(a_exp.select(-1, 0), b_exp.select(-1, 0)), 0);
    auto inter_h = torch::clamp_min(
        torch::min(a_exp.select(-1, 3), b_exp.select(-1, 3)) -
        torch::max(a_exp.select(-1, 1), b_exp.select(-1, 1)), 0);
    auto inter = inter_w * inter_h;

    auto area_a = (a.select(-1, 2) - a.select(-1, 0)) * (a.select(-1, 3) - a.select(-1, 1));
    auto area_b = (b.select(-1, 2) - b.select(-1, 0)) * (b.select(-1, 3) - b.select(-1, 1));
    auto uni = area_a.unsqueeze(2) + area_b.unsqueeze(1) - inter;

    return inter / torch::clamp_min(uni, 1e-6f);
}

}

torch::Tensor ssd_loss(
    const torch::Tensor& pred_offsets,
    const torch::Tensor& pred_labels,
    const std::vector<torch::Tensor>& targets,
    const torch::Tensor& anchors) {

    auto device = pred_offsets.device();
    int64_t B = pred_offsets.size(0);
    int64_t N = pred_offsets.size(1);

    constexpr float iou_threshold = 0.5f;
    constexpr float neg_pos_ratio = 3.0f;

    int64_t M = 0;
    for (int64_t i = 0; i < B; ++i)
        M = std::max(M, targets[i].size(0));

    if (M == 0) {
        auto conf_targets = torch::zeros({ B * N }, torch::TensorOptions().dtype(torch::kLong).device(device));
        return torch::nn::functional::cross_entropy(
            pred_labels.reshape({ B * N, -1 }),
            conf_targets,
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    }

    auto gt_padded = torch::zeros({ B, M, 5 }, torch::TensorOptions().device(device));
    auto gt_valid = torch::zeros({ B, M }, torch::TensorOptions().dtype(torch::kBool).device(device));

    for (int64_t i = 0; i < B; ++i) {
        int64_t n = targets[i].size(0);
        if (n > 0) {
            gt_padded[i].slice(0, 0, n).copy_(targets[i].to(device));
            gt_valid[i].slice(0, 0, n).fill_(true);
        }
    }

    auto gt_boxes = gt_padded.slice(2, 0, 4);
    auto gt_labels = gt_padded.select(2, 4).to(torch::kLong);

    auto anchors_dev = anchors.to(device);
    auto anchors_xyxy = batch_xywh_to_xyxy(anchors_dev).unsqueeze(0).expand({ B, N, 4 });
    auto gt_xyxy = batch_xywh_to_xyxy(gt_boxes);

    auto ious = batch_iou(anchors_xyxy, gt_xyxy);
    ious.masked_fill_(~gt_valid.unsqueeze(1), -1.0f);

    auto [max_iou, max_gt_idx] = ious.max(2);
    auto pos_mask = max_iou >= iou_threshold;

    auto [best_anchor_iou, best_anchor_idx] = ious.max(1);
    auto gt_indices = torch::arange(M, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0).expand({ B, M });
    pos_mask.scatter_(1, best_anchor_idx, gt_valid);
    max_gt_idx.scatter_(1, best_anchor_idx, gt_indices);

    auto pos_f = pos_mask.to(torch::kFloat);

    auto matched_gt_boxes = torch::gather(gt_boxes, 1, max_gt_idx.unsqueeze(2).expand({ B, N, 4 }));
    auto anchors_expanded = anchors_dev.unsqueeze(0).expand({ B, N, 4 });

    auto gx = matched_gt_boxes.select(2, 0);
    auto gy = matched_gt_boxes.select(2, 1);
    auto gw = matched_gt_boxes.select(2, 2);
    auto gh = matched_gt_boxes.select(2, 3);

    auto ax = anchors_expanded.select(2, 0);
    auto ay = anchors_expanded.select(2, 1);
    auto aw = anchors_expanded.select(2, 2);
    auto ah = anchors_expanded.select(2, 3);

    auto tx = (gx - ax) / aw / 0.1f;
    auto ty = (gy - ay) / ah / 0.1f;
    auto tw = torch::log(torch::clamp_min(gw / aw, 1e-6f)) / 0.2f;
    auto th = torch::log(torch::clamp_min(gh / ah, 1e-6f)) / 0.2f;

    auto target_offsets = torch::stack({ tx, ty, tw, th }, 2);

    auto loc_loss_elem = torch::nn::functional::smooth_l1_loss(
        pred_offsets, target_offsets,
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
    auto loc_loss = (loc_loss_elem.sum(2) * pos_f).sum();

    auto matched_labels = torch::gather(gt_labels, 1, max_gt_idx);
    auto conf_targets = torch::where(pos_mask, matched_labels, torch::zeros_like(matched_labels));

    auto conf_loss_all = torch::nn::functional::cross_entropy(
        pred_labels.reshape({ B * N, -1 }),
        conf_targets.reshape({ B * N }),
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)
    ).reshape({ B, N });

    auto pos_conf_loss = (conf_loss_all * pos_f).sum();

    auto neg_mask = ~pos_mask;
    auto num_pos = pos_mask.sum(1).to(torch::kFloat);
    auto num_neg_limit = torch::clamp_min(num_pos * neg_pos_ratio, neg_pos_ratio).to(torch::kLong);

    auto neg_scores = conf_loss_all * neg_mask.to(torch::kFloat);
    auto sorted_neg = std::get<0>(neg_scores.sort(1, true));

    auto ranks = torch::arange(N, torch::TensorOptions().device(device)).unsqueeze(0);
    auto neg_mining_mask = ranks < num_neg_limit.unsqueeze(1);

    auto neg_conf_loss = (sorted_neg * neg_mining_mask.to(torch::kFloat)).sum();

    auto total_pos = torch::clamp_min(pos_mask.sum().to(torch::kFloat), 1.0f);
    return (loc_loss + pos_conf_loss + neg_conf_loss) / total_pos;
}
