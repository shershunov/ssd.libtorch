# Single Shot Multibox Detector (SSD) with LibTorch in C++

Implementation of the Single Shot Multibox Detector (SSD) using LibTorch for real-time object detection.

## Features

- **Multi-scale detection** with 3 feature pyramid levels (stride 8, 16, 32)
- **CSPDarknet backbone** with ConvBNSiLU, BottleNeck, and C3 modules
- **Anchor-based detection** with multi-scale anchor generation
- **Mixed precision training** (FP16, BF16, FP32) with GradScaler
- **Cosine annealing LR** with linear warmup
- **mAP validation** (mAP@0.5 and mAP@0.5:0.95) during training
- **Data augmentation** (horizontal flip)
- **Hard negative mining** for class-balanced loss

## Requirements
libtorch and opencv are located in ```C:/Program Files``` by default.
- CMake 3.8 or higher
- C++17
- [OpenCV](https://opencv.org/releases/) 4.12 or higher
- [LibTorch](https://pytorch.org/get-started/locally/) 2.3.0 or higher

## Usage

### Init model
```cpp
float depth_multiple = 0.33F;
float width_multiple = 0.25F;

torch::Device device(torch::kCPU);
if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
}
torch::manual_seed(1337);

Net model(num_classes, num_anchors, depth_multiple, width_multiple);
model.to(device);
```

### Generate anchors
```cpp
auto anchors = generate_anchors(img_size, num_anchors);
```

### Load data
```cpp
auto [images, targets] = get_data(dataset_path, img_size, "train");
auto [val_images, val_targets] = get_data(dataset_path, img_size, "val");
```

### Train
```cpp
const int EPOCHS = 70;
const int batch_size = 64;
const int img_size = 640;
const int num_classes = 5;
const int num_anchors = 2;
const float learning_rate = 0.0001F;
std::string dataset_path = "C:/datasets/data/";
const int num_workers = 6;
PrecisionMode precision = PrecisionMode::FP16;

std::vector<float> losses_train = train_model(model, device, images, targets,
    val_images, val_targets, EPOCHS, batch_size, learning_rate, num_workers,
    precision, num_classes, anchors);
```
Training saves `best.pt` (best mAP@0.5:0.95) and `last.pt` checkpoints automatically.

### Test
```cpp
load_model(model, "best.pt");
test_model(model, device, img_size, dataset_path, anchors);
```
Results are saved to the `test/` directory with visualized bounding boxes.

### Detect on photo
```cpp
cv::Mat image = cv::imread("photo.png", cv::IMREAD_COLOR);
torch::Tensor test_img = normalize_image(image, img_size).to(device).unsqueeze(0);

auto [loc_preds, conf_preds] = model.forward(test_img);
auto decoded_boxes = decode_boxes(loc_preds, anchors);
```

## Dataset
Dataset structure and markup similar to YOLO.
```
data/
  labels.txt
  images/
    train/
      1.png
    val/
      1.png
  labels/
    train/
      1.txt
    val/
      1.txt
```
#### labels.txt example
```
0: pig
1: sheep
2: horse
3: cow
```
