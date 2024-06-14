#include "test.h"
#include "dataloader.h"
#include "loss.h"
#include "model.h"
#include "train.h"
#include <iostream>

int main() {
    try {
        const int EPOCHS = 200;
        const int batch_size = 1;
        const int width = 640;
        const int height = 640;
        const int num_classes = 4;
        const int num_anchors = 3;
        const float learning_rate = 0.0001F;
        const bool shuffle_dataset = true;
        std::string dataset_path = "C:/datasets/data";

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }

        Net model(num_classes, num_anchors);
        model.to(device);

        auto [images, targets] = get_train_data(dataset_path, width, height, device, num_classes);

        std::vector<float> losses_train = train_model(model, device, images, targets, EPOCHS, batch_size, learning_rate, shuffle_dataset);

        //load_model(model, "best.pt");
        test_model(model, device, width, height);
    }
    catch (const torch::Error& error) {
        std::cerr << "LibTorch error: " << error.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
};
