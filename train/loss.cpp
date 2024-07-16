#include "loss.h"
#include "../utils/utils.h"


torch::Tensor iou(const torch::Tensor& boxesA, const torch::Tensor& boxesB) {
    torch::Tensor boxesA_exp = boxesA.unsqueeze(1);
    torch::Tensor boxesB_exp = boxesB.unsqueeze(0);

    torch::Tensor xA = torch::max(boxesA_exp.select(2, 0), boxesB_exp.select(2, 0));
    torch::Tensor yA = torch::max(boxesA_exp.select(2, 1), boxesB_exp.select(2, 1));
    torch::Tensor xB = torch::min(boxesA_exp.select(2, 2), boxesB_exp.select(2, 2));
    torch::Tensor yB = torch::min(boxesA_exp.select(2, 3), boxesB_exp.select(2, 3));

    torch::Tensor interWidth = torch::clamp(xB - xA + 1, 0, std::numeric_limits<float>::infinity());
    torch::Tensor interHeight = torch::clamp(yB - yA + 1, 0, std::numeric_limits<float>::infinity());
    torch::Tensor interArea = interWidth * interHeight;

    torch::Tensor boxesAArea = (boxesA.select(1, 2) - boxesA.select(1, 0) + 1) * (boxesA.select(1, 3) - boxesA.select(1, 1) + 1);
    torch::Tensor boxesBArea = (boxesB.select(1, 2) - boxesB.select(1, 0) + 1) * (boxesB.select(1, 3) - boxesB.select(1, 1) + 1);

    torch::Tensor boxesAArea_exp = boxesAArea.unsqueeze(1);
    torch::Tensor boxesBArea_exp = boxesBArea.unsqueeze(0);

    torch::Tensor unionArea = boxesAArea_exp + boxesBArea_exp - interArea;

    unionArea = torch::max(unionArea, torch::ones_like(unionArea) * 1e-6);

    torch::Tensor iou = interArea / unionArea;

    return iou;
}

torch::Tensor ssd_loss(
    const torch::Tensor& pred_boxes,
    const torch::Tensor& pred_labels,
    const std::vector<torch::Tensor>& targets) {

    torch::Device device = pred_boxes.device();

    std::vector<torch::Tensor> true_boxes, true_labels;

    for (const auto& target : targets) {
        true_boxes.push_back(target.slice(-1, 0, 4).to(device));
        true_labels.push_back(target.narrow(-1, -1, 1).squeeze(1).to(device));
    }

    int num_batches = pred_boxes.size(0);

    torch::Tensor total_loc_loss = torch::zeros({ 1 }, torch::TensorOptions().device(device));
    torch::Tensor total_conf_loss = torch::zeros({ 1 }, torch::TensorOptions().device(device));

    const float overlap_threshold = 0.5F;
    const float neg_pos_ratio = 3.0F;
    const float loc_loss_weight = 1.0F;
    const float conf_loss_weight = 1.0F;

    for (int i = 0; i < num_batches; ++i) {
        torch::Tensor current_true_boxes = xywh_to_xyxy(true_boxes[i]); // [num_real_boxes, 4] (xyxy)
        torch::Tensor current_true_labels = true_labels[i]; // [num_real_boxes] class indices
        torch::Tensor current_pred_boxes = xywh_to_xyxy(pred_boxes[i]); // [num_pred_boxes, 4] (xyxy)
        torch::Tensor current_pred_labels = pred_labels[i]; // [num_pred_boxes, num_classes] class probabilities

        torch::Tensor ious = iou(current_pred_boxes, current_true_boxes); // [num_anchors, num_real_boxes]
        std::tuple< torch::Tensor, torch::Tensor > max_iou_results = ious.max(1);
        torch::Tensor max_iou = std::get<0>(max_iou_results);
        torch::Tensor max_iou_idx = std::get<1>(max_iou_results);

        torch::Tensor positive_mask = max_iou >= overlap_threshold;
        torch::Tensor negative_mask = max_iou < overlap_threshold;

        int num_positives = positive_mask.sum().item<int>();
        int num_negatives = negative_mask.sum().item<int>();

        if (num_positives > 0) {
            torch::Tensor positive_pred_boxes = current_pred_boxes.index({ positive_mask });
            torch::Tensor positive_true_boxes = current_true_boxes.index({ max_iou_idx.index({ positive_mask }) });

            torch::Tensor loc_loss = torch::nn::functional::smooth_l1_loss(
                positive_pred_boxes,
                positive_true_boxes,
                torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kSum)
            );

            total_loc_loss += loc_loss_weight * loc_loss;

            torch::Tensor positive_pred_labels = current_pred_labels.index({ positive_mask });
            torch::Tensor positive_true_labels = current_true_labels.index({ max_iou_idx.index({ positive_mask }) }).to(torch::kLong);

            torch::Tensor conf_loss_pos = torch::nn::functional::cross_entropy(
                positive_pred_labels,
                positive_true_labels,
                torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)
            );

            total_conf_loss += conf_loss_weight * conf_loss_pos;
        }

        if (num_negatives > 0) {
            torch::Tensor negative_pred_labels = current_pred_labels.index({ negative_mask });

            torch::Tensor conf_loss_neg_all = torch::nn::functional::cross_entropy(
                negative_pred_labels,
                torch::zeros({ negative_pred_labels.size(0) }, torch::TensorOptions().dtype(torch::kLong).device(device)),
                torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)
            );

            auto [sorted_neg_conf_loss, sorted_neg_indices] = conf_loss_neg_all.sort(0, true);

            int num_neg = std::min(static_cast<int>(num_positives * neg_pos_ratio), static_cast<int>(sorted_neg_indices.size(0)));
            if (num_neg == 0) {
                num_neg = std::min(static_cast<int>(neg_pos_ratio), static_cast<int>(sorted_neg_indices.size(0)));
            }

            torch::Tensor hard_neg_indices = sorted_neg_indices.slice(0, 0, num_neg);
            torch::Tensor hard_neg_conf_loss = conf_loss_neg_all.index({ hard_neg_indices });

            torch::Tensor conf_loss_neg = hard_neg_conf_loss.sum();

            total_conf_loss += conf_loss_weight * conf_loss_neg;
        }
    }

    torch::Tensor total_loss = total_loc_loss + total_conf_loss;
    return total_loss;
}
