# Single Shot Multibox Detector (SSD) with LibTorch in C++

This repository contains an implementation of the Single Shot Multibox Detector (SSD) using LibTorch. SSD is a popular model for real-time object detection in images.

## Features

- **LibTorch**: Utilizes LibTorch for deep learning operations.
- **Easy to Use**: Clear and convenient interface for working with the model.

## Requirements
libtorch and opencv are located in ```C:/Program Files``` by default.
- CMake 3.8 or higher
- [OpenCV](https://opencv.org/releases/) 4.9 or higher (for image processing)
- [LibTorch](https://pytorch.org/get-started/locally/) 2.3.0 or higher

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

### Train
```cpp
const int EPOCHS = 4000;
const int batch_size = 56;
const int img_size = 640;
const int num_classes = 4;
const int num_anchors = 2;
const float learning_rate = 0.0001F;
std::string dataset_path = "C:/datasets/data/";
const int num_workers = 6;

auto [images, targets] = get_train_data(dataset_path, img_size, device, num_classes);

std::vector<float> losses_train = train_model(model, device, images, targets, EPOCHS, batch_size, learning_rate, num_workers);
```

### Detect on photo
```cpp
cv::Mat image = cv::imread("photo.png", cv::IMREAD_COLOR);
torch::Tensor test_img = normalize_image(image, img_size).to(device).unsqueeze(0);

auto [boxes, scores] = model.forward(test_img); // xywh
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
