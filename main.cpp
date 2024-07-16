#include "utils/test.h"
#include "utils/dataloader.h"
#include "model/model.h"
#include "train/train.h"

int main() {
    try {
        const int EPOCHS = 4000;
        const int batch_size = 56;
        const int img_size = 640;
        const int num_classes = 4;
        const int num_anchors = 2;
        const float learning_rate = 0.0001F;

        float depth_multiple = 0.33F;
        float width_multiple = 0.25F;

        std::string dataset_path = "C:/datasets/data/";
        const int num_workers = 6;

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
        torch::manual_seed(1337);

        Net model(num_classes, num_anchors, depth_multiple, width_multiple);
        model.to(device);

        auto [images, targets] = get_train_data(dataset_path, img_size, device, num_classes);

        std::vector<float> losses_train = train_model(model, device, images, targets, EPOCHS, batch_size, learning_rate, num_workers);

        load_model(model, "best.pt");
        test_model(model, device, img_size, dataset_path);
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
