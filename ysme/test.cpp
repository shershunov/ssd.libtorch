#include "test.h"
#include <filesystem>

cv::Mat plot_box(const int x[4], cv::Mat& img, const std::string& label, int line_thickness, cv::Scalar color) {
    int tl = line_thickness == 0 ? static_cast<uint8_t>(.002 * (img.cols + img.rows) / 2 + 1) : line_thickness;
    const cv::Scalar box_color = color;
    const cv::Scalar label_bg_color = color;
    const cv::Scalar label_text_color(225, 255, 255);

    const cv::Point c1(x[0], x[1]);
    const cv::Point c2(x[2], x[3]);
    cv::rectangle(img, c1, c2, box_color, tl, cv::LINE_AA);

    const int label_width = static_cast<int>(label.size() * 7 * line_thickness * 0.6);
    const int label_height = static_cast<int>(10 * tl * 1.25);
    const cv::Point label_bottom_right(c1.x + label_width, c1.y - label_height);

    cv::rectangle(img, cv::Point(c1.x - tl + 1, c1.y), label_bottom_right, label_bg_color, -1, cv::LINE_AA);
    cv::putText(img, label, cv::Point(c1.x + 2, c1.y - 4), cv::FONT_HERSHEY_DUPLEX, 0.3 * tl, label_text_color, 1, cv::LINE_AA);
    return img;
}

std::vector<int> tensor_to_int_bbox(torch::Tensor& input, int width, int height) {
    int boxes = input.size(0);

    std::vector<int> result;
    result.reserve(boxes * 4);

    for (int i = 0; i < boxes; ++i) {
        int x1 = static_cast<int>((input[i][0].item<float>() * width));
        int y1 = static_cast<int>((input[i][1].item<float>() * height));
        int x2 = static_cast<int>((input[i][2].item<float>() * width));
        int y2 = static_cast<int>((input[i][3].item<float>() * height));

        result.push_back(x1);
        result.push_back(y1);
        result.push_back(x2);
        result.push_back(y2);
    }

    return result;
}

void test_model(Net& model, torch::Device& device, int width, int height) {
    model.eval();
    std::string abc = "-";
    std::string num_photo;
    while (true) {
        std::cout << std::endl;
        std::cin >> num_photo;
        cv::Mat image = cv::imread("C:/Users/dev/source/repos/ysme/dataset/images/train/" + num_photo + ".png", cv::IMREAD_COLOR);
        torch::Tensor test_img = normalize_image(image, width, height, device).unsqueeze(0);
        auto [boxes, scores] = model.forward(test_img);
        auto boxes_xyxy = xywh_to_xyxy(boxes.squeeze(0));

        auto boxes_int = tensor_to_int_bbox(boxes_xyxy, width, height);
        for (int i = 0; i < boxes_int.size(); i += 4) {
            int box[4];
            box[0] = boxes_int[i];
            box[1] = boxes_int[i + 1];
            box[2] = boxes_int[i + 2];
            box[3] = boxes_int[i + 3];

            image = plot_box(box, image, abc, 1, cv::Scalar(0, 165, 255));
        }
        cv::imshow("window", image);
        cv::waitKey(1);
    }
}
