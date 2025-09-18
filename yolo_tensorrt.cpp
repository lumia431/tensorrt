/*
 * 完整的 TensorRT + YOLO 目标检测系统实现
 * 涵盖：ONNX模型加载、TensorRT引擎构建、GPU内存管理、推理优化、后处理等
 * 面试重点：深度学习模型部署、CUDA编程、内存管理、性能优化
 */

// ========== 标准库头文件 ==========
#include <iostream>          // 标准输入输出流，用于日志打印
#include <fstream>           // 文件流操作（本例中未直接使用，但常用于模型文件读取）
#include <memory>            // 智能指针，用于RAII资源管理，面试常问unique_ptr vs shared_ptr
#include <vector>            // 动态数组，存储检测结果、像素数据等
#include <algorithm>         // STL算法库，用于max_element查找最大类别置信度
#include <cassert>           // 断言，用于调试时检查条件
#include <chrono>            // 高精度时间测量，用于性能分析

// ========== 第三方库头文件 ==========
#include <opencv2/opencv.hpp> // OpenCV计算机视觉库，图像预处理和可视化
#include <cuda_runtime.h>     // CUDA运行时API，GPU内存管理和核函数调用
#include <cuda_fp16.h>        // CUDA半精度浮点支持，用于FP16推理优化
#include <NvInfer.h>          // TensorRT推理引擎核心API
#include <NvOnnxParser.h>     // ONNX模型解析器，将ONNX转换为TensorRT网络

// ========== 命名空间声明 ==========
using namespace nvinfer1;     // TensorRT命名空间，包含推理引擎相关类
using namespace nvonnxparser;  // ONNX解析器命名空间
using namespace cv;           // OpenCV命名空间，图像处理相关函数

/* 面试知识点：命名空间的作用
 * 1. 避免命名冲突，特别是在大型项目中
 * 2. 提高代码可读性，明确标识符来源
 * 3. 但过度使用using namespace可能导致命名污染
 */

// ========== TensorRT日志记录器 ==========
/* 面试重点：为什么需要自定义Logger？
 * 1. TensorRT内部会产生大量日志信息（构建过程、优化细节、警告等）
 * 2. 自定义Logger可以控制日志级别，避免输出过多信息
 * 3. 生产环境中通常需要将日志重定向到文件或日志系统
 */
class Logger : public ILogger {
public:
    // 重写TensorRT的日志接口，noexcept表示此函数不会抛出异常
    void log(Severity severity, const char* msg) noexcept override {
        // 只输出警告级别及以上的日志（ERROR, INTERNAL_ERROR, WARNING）
        // 过滤掉INFO和VERBOSE级别，减少控制台输出
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger; // 全局Logger实例，整个程序共享使用

// ========== 检测结果数据结构 ==========
/* 面试考点：数据结构设计原则
 * 1. 简洁性：只包含必要字段，避免冗余
 * 2. 内存对齐：合理排列字段顺序，优化内存布局
 * 3. 可扩展性：后续可加入track_id、depth等字段
 */
struct Detection {
    float x, y, w, h;    // 边界框坐标：左上角(x,y) + 宽高(w,h)
                         // 注意：这是绝对像素坐标，已从归一化坐标转换
    float confidence;    // 检测置信度，范围[0,1]，越高表示越确信
    int class_id;        // 类别ID，对应COCO数据集的80个类别（0=person, 1=bicycle...）
};

/* 拓展知识：其他常见的边界框表示方法
 * 1. (x1,y1,x2,y2): 左上角和右下角坐标
 * 2. (cx,cy,w,h): 中心点坐标和宽高
 * 3. 归一化坐标: 相对于图像尺寸的比例[0,1]
 */

// ========== YOLO TensorRT 推理引擎类 ==========
/* 面试重点：面向对象设计原则
 * 1. 封装：隐藏内部实现细节，提供简洁接口
 * 2. RAII：Resource Acquisition Is Initialization，资源管理的黄金法则
 * 3. 单一职责：只负责YOLO模型的TensorRT推理
 */
class YOLOv5TensorRT {
private:
    // ========== TensorRT核心组件 ==========
    std::unique_ptr<ICudaEngine> engine;        // TensorRT推理引擎，已优化的神经网络
    std::unique_ptr<IExecutionContext> context; // 执行上下文，存储运行时状态
    void* buffers[2];                           // GPU内存缓冲区：[0]=输入，[1]=输出
    cudaStream_t stream;                        // CUDA流，用于异步执行和内存拷贝

    // ========== 张量名称管理 ==========
    std::string input_tensor_name;              // 输入张量名称（通常为"images"）
    std::string output_tensor_name;             // 输出张量名称（通常为"output0"）

    // ========== YOLO模型参数 ==========
    int input_size = 640;           // 输入图像尺寸（YOLOv5s标准尺寸）
    int num_classes = 80;           // COCO数据集类别数，面试常问：为什么是80？
    int num_anchors = 25200;        // 预测框数量 = 80x80/4 + 40x40/4 + 20x20/4 = 8400每层x3层
    float conf_threshold = 0.01f;   // 置信度阈值，过滤低置信度检测
    float nms_threshold = 0.45f;    // NMS阈值，去除重复检测框

    /* 面试知识点：YOLO Anchor机制
     * YOLOv5使用三个尺度的特征图：
     * - P3: 80x80, 检测小目标
     * - P4: 40x40, 检测中等目标
     * - P5: 20x20, 检测大目标
     * 每个特征点预测3个边界框，总计：(80x80+40x40+20x20)x3 = 25200
     */

    // ========== COCO数据集类别名称 ==========
    /* 面试考点：为什么将类别名写死在代码里？
     * 1. 优点：简单直接，无需额外文件
     * 2. 缺点：不灵活，更换模型需重新编译
     * 3. 生产级实现：通常从配置文件或数据库加载
     * 4. COCO数据集标准：80个常见物体类别，计算机视觉领域的基准
     */
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

public:
    // ========== 构造函数：初始化CUDA流 ==========
    /* 面试重点：RAII资源管理模式
     * 1. 构造函数中获取资源（CUDA流）
     * 2. 析构函数中释放资源，保证无内存泄漏
     * 3. 初始化列表语法：buffers{nullptr, nullptr}
     */
    YOLOv5TensorRT() : buffers{nullptr, nullptr} {
        cudaStreamCreate(&stream);  // 创建CUDA流，用于异步执行
    }

    // ========== 析构函数：清理GPU资源 ==========
    ~YOLOv5TensorRT() {
        // 释放GPU内存缓冲区，防止内存泄漏
        if (buffers[0]) cudaFree(buffers[0]);
        if (buffers[1]) cudaFree(buffers[1]);
        cudaStreamDestroy(stream);  // 销毁CUDA流
    }

    // ========== 核心方法：从 ONNX 构建 TensorRT 引擎 ==========
    /* 面试重点：TensorRT 引擎构建流程
     * 1. ONNX -> TensorRT Network -> Optimized Engine
     * 2. 构建时间较长（几分钟），但推理速度极快
     * 3. 引擎可序列化保存，下次直接加载
     * 4. 针对特定GPU架构优化，不同显卡不通用
     */
    bool buildEngine(const std::string& onnx_path) {
        // 步骤1：创建 TensorRT 构建器
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
        if (!builder) {
            std::cerr << "Failed to create builder" << std::endl;
            return false;
        }

        // 步骤2：创建网络定义（传参0表示默认设置）
        auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
        if (!network) {
            std::cerr << "Failed to create network" << std::endl;
            return false;
        }

        // 步骤3：创建 ONNX 解析器
        auto parser = std::unique_ptr<IParser>(createParser(*network, gLogger));
        if (!parser) {
            std::cerr << "Failed to create parser" << std::endl;
            return false;
        }

        // 步骤4：解析 ONNX 文件到 TensorRT 网络
        /* 面试知识点：ONNX 模型格式
         * 1. ONNX = Open Neural Network Exchange，开放神经网络交换格式
         * 2. 跨框架模型表示，支持 PyTorch/TensorFlow 等转换
         * 3. 基于 protobuf 序列化，包含网络结构和权重
         */
        auto parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
        if (!parsed) {
            std::cerr << "Failed to parse ONNX file" << std::endl;
            return false;
        }

        // 步骤5：创建构建配置
        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create builder config" << std::endl;
            return false;
        }

        // 设置工作空间内存限制：1GB（1U << 30 = 1024^3 字节）
        /* 面试考点：为什么需要工作空间？
         * 1. TensorRT 在优化过程中需要临时内存存储中间结果
         * 2. 更大的工作空间允许更复杂的优化策略
         * 3. 但会占用更多 GPU 内存，需要平衡
         */
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

        // 步骤6：构建优化后的引擎（这一步耗时最长）
        /* 面试重点：TensorRT 优化原理
         * 1. 算子融合：将多个小算子合并为一个大算子
         * 2. 精度优化：FP32->FP16->INT8，减少计算量
         * 3. 内存优化：去除不必要的内存拷贝
         * 4. 核函数选择：为每个操作选择最优实现
         */
        auto serialized_engine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!serialized_engine) {
            std::cerr << "Failed to build engine" << std::endl;
            return false;
        }

        // 步骤7：创建运行时环境
        auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        if (!runtime) {
            std::cerr << "Failed to create runtime" << std::endl;
            return false;
        }

        // 步骤8：反序列化引擎（将二进制数据转换为可执行引擎）
        engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(
            serialized_engine->data(), serialized_engine->size()));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }

        // 步骤9：创建执行上下文（用于实际推理）
        context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }

        // ========== 调试信息：打印张量信息 ==========
        /* 面试技巧：如何调试 TensorRT 模型？
         * 1. 打印张量名称和形状，确保与预期一致
         * 2. 检查输入输出数据类型（float32/float16）
         * 3. 验证内存分配是否正确
         */
        std::cout << "Engine tensor count: " << engine->getNbIOTensors() << std::endl;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            auto name = engine->getIOTensorName(i);
            auto dims = engine->getTensorShape(name);
            std::cout << "Tensor " << i << ": " << name << " shape: ";
            for (int j = 0; j < dims.nbDims; ++j) {
                std::cout << dims.d[j] << " ";
            }
            std::cout << " mode: " << (engine->getTensorIOMode(name) == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;
        }

        // 动态获取张量名称（避免硬编码）
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            auto name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                input_tensor_name = name;
            } else {
                output_tensor_name = name;
            }
        }

        std::cout << "Using input tensor: " << input_tensor_name << std::endl;
        std::cout << "Using output tensor: " << output_tensor_name << std::endl;

        // ========== GPU 内存分配 ==========
        /* 面试重点：GPU 内存管理
         * 1. 计算张量大小：各维度乘积
         * 2. 分配连续内存：cudaMalloc 分配连续的 GPU 内存
         * 3. 内存对齐：现代 GPU 需要内存地址对齐以获得最佳性能
         */
        auto input_dims = engine->getTensorShape(input_tensor_name.c_str());
        auto output_dims = engine->getTensorShape(output_tensor_name.c_str());

        // 计算输入张量大小：1 * 3 * 640 * 640 = 1,228,800 个元素
        size_t input_size = 1;
        for (int i = 0; i < input_dims.nbDims; ++i) {
            input_size *= input_dims.d[i];
        }

        // 计算输出张量大小：1 * 25200 * 85 = 2,142,000 个元素
        size_t output_size = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size *= output_dims.d[i];
        }

        // 分配 GPU 内存：使用 FP16 半精度以节约内存和提高性能
        /* 面试知识点：为什么使用 FP16？
         * 1. 内存占用减半：FP16 只需 FP32 一半的存储空间
         * 2. 计算速度更快：现代 GPU 对 FP16 有专门优化
         * 3. 精度损失小：对于绝大多数深度学习任务可接受
         */
        cudaMalloc(&buffers[0], input_size * sizeof(__half));
        cudaMalloc(&buffers[1], output_size * sizeof(__half));

        // 初始化输出缓冲区为零，避免随机值干扰
        cudaMemset(buffers[1], 0, output_size * sizeof(__half));

        // 打印构建成功信息和张量形状
        std::cout << "Engine built successfully!" << std::endl;
        std::cout << "Input shape: ";
        for (int i = 0; i < input_dims.nbDims; ++i) {
            std::cout << input_dims.d[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Output shape: ";
        for (int i = 0; i < output_dims.nbDims; ++i) {
            std::cout << output_dims.d[i] << " ";
        }
        std::cout << std::endl;

        return true;  // 构建成功
    }

    // ========== 图像预处理方法 ==========
    /* 面试重点：计算机视觉预处理步骤
     * 1. 尺寸缩放：统一输入尺寸，便于批处理
     * 2. 数值归一化：[0,255] -> [0,1]，稳定训练和推理
     * 3. 数据类型转换：uint8 -> float32
     * 4. 通道转换在后面单独处理：HWC -> CHW
     */
    Mat preprocess(const Mat& image) {
        Mat resized, blob;
        // OpenCV resize：双线性插值缩放到 640x640
        resize(image, resized, Size(input_size, input_size));

        // 数值归一化并转换为浮点型：[0,255] -> [0,1]
        resized.convertTo(blob, CV_32F, 1.0 / 255.0);

        return blob;
    }

    // ========== 后处理方法：解析模型输出 ==========
    /* 面试重点：YOLO 后处理流程
     * 1. 置信度过滤：去除低置信度检测
     * 2. 类别选择：找到最高置信度的类别
     * 3. 坐标转换：中心点 -> 左上角，归一化 -> 绝对像素
     * 4. NMS：Non-Maximum Suppression，去除重复检测框
     */
    std::vector<Detection> postprocess(const std::vector<float>& output, float scale_x, float scale_y) {
        std::vector<Detection> detections;
        std::cout << "Processing " << num_anchors << " predictions..." << std::endl;

        // 打印前几个预测结果用于调试
        for (int i = 0; i < 5; ++i) {
            const float* data = &output[i * (num_classes + 5)];
            std::cout << "Prediction " << i << ": "
                      << "cx=" << data[0] << " cy=" << data[1]
                      << " w=" << data[2] << " h=" << data[3]
                      << " conf=" << data[4] << std::endl;
        }

        int valid_count = 0;
        // 遍历所有 25200 个预测框
        for (int i = 0; i < num_anchors; ++i) {
            // 每个预测框格式：[cx, cy, w, h, confidence, class0_score, class1_score, ..., class79_score]
            const float* data = &output[i * (num_classes + 5)];
            float confidence = data[4];  // 目标存在置信度

            // 第一次过滤：目标存在置信度过低的直接跳过
            if (confidence >= conf_threshold) {
                valid_count++;

                // 获取类别分数数组（第 5 个元素之后的 80 个元素）
                float* class_scores = (float*)&data[5];

                // 找到最高分数的类别索引
                int class_id = std::max_element(class_scores, class_scores + num_classes) - class_scores;

                // 计算最终置信度：目标存在置信度 × 类别置信度
                float class_confidence = class_scores[class_id] * confidence;

                // 第二次过滤：最终置信度过低的跳过
                if (class_confidence >= conf_threshold) {
                    // 获取边界框坐标（归一化坐标，中心点格式）
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];

                    Detection det;
                    // 坐标转换：中心点 -> 左上角，归一化 -> 绝对像素
                    det.x = (cx - w / 2) * scale_x;
                    det.y = (cy - h / 2) * scale_y;
                    det.w = w * scale_x;
                    det.h = h * scale_y;
                    det.confidence = class_confidence;
                    det.class_id = class_id;

                    detections.push_back(det);
                }
            }
        }

        std::cout << "Found " << valid_count << " predictions above confidence threshold" << std::endl;
        std::cout << "Found " << detections.size() << " detections before NMS" << std::endl;

        // ========== NMS（非最大值抑制）算法 ==========
        /* 面试重点：NMS 算法原理
         * 1. 问题：同一个目标可能被多个边界框检测到
         * 2. 解决：保留置信度最高的框，删除重复的框
         * 3. 判断标准：IoU (Intersection over Union) > threshold
         * 4. 复杂度：O(n^2)，但实际上由于预过滤，n 不大
         */
        std::vector<int> indices;        // NMS 后保留的检测索引
        std::vector<Rect> boxes;         // OpenCV 边界框格式
        std::vector<float> scores;       // 对应的置信度分数

        // 转换为 OpenCV 格式
        for (const auto& det : detections) {
            boxes.push_back(Rect(det.x, det.y, det.w, det.h));
            scores.push_back(det.confidence);
        }

        // 调用 OpenCV 的 NMS 实现
        dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

        // 根据 NMS 结果构建最终检测结果
        std::vector<Detection> final_detections;
        for (int idx : indices) {
            final_detections.push_back(detections[idx]);
        }

        return final_detections;
    }

    // ========== 主推理函数：完整的检测流程 ==========
    /* 面试重点：深度学习推理流程
     * 1. 预处理：图像 -> 模型输入格式
     * 2. 数据转换：HWC -> CHW，float32 -> float16
     * 3. GPU 推理：TensorRT 高性能推理
     * 4. 后处理：解析输出 -> 检测结果
     */
    std::vector<Detection> detect(const Mat& image) {
        // 检查引擎是否初始化
        if (!engine || !context) {
            std::cerr << "Engine not initialized" << std::endl;
            return {};
        }

        // 步骤1：图像预处理
        Mat processed = preprocess(image);

        // 步骤2：通道转换 HWC -> CHW
        /* 面试知识点：为什么需要通道转换？
         * 1. OpenCV 格式：HWC (Height-Width-Channel)
         * 2. 深度学习格式：CHW (Channel-Height-Width)
         * 3. GPU 对 CHW 格式的内存访问更高效
         */
        std::vector<Mat> channels(3);
        split(processed, channels);  // 分离 BGR 三个通道

        // 步骤3：数据类型转换 float32 -> float16
        std::vector<__half> input_data_half;
        for (int c = 0; c < 3; ++c) {
            // 将每个通道的 2D 矩阵拉平为 1D 数组
            Mat flat = channels[c].reshape(1, input_size * input_size);
            float* data = (float*)flat.data;
            for (int i = 0; i < input_size * input_size; ++i) {
                // CUDA 函数：float32 -> float16 转换
                input_data_half.push_back(__float2half(data[i]));
            }
        }

        // 调试信息：打印输入数据统计
        std::cout << "Input data size: " << input_data_half.size() << std::endl;
        std::cout << "First few input values (as float): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << __half2float(input_data_half[i]) << " ";
        }
        std::cout << std::endl;

        // 步骤4：GPU 内存拷贝（CPU -> GPU）
        /* 面试知识点：CUDA 内存管理
         * 1. cudaMemcpy: 同步拷贝，阻塞直到拷贝完成
         * 2. cudaMemcpyAsync: 异步拷贝，需要 CUDA 流同步
         * 3. Host -> Device: CPU 内存到 GPU 内存
         */
        cudaMemcpy(buffers[0], input_data_half.data(), input_data_half.size() * sizeof(__half), cudaMemcpyHostToDevice);

        // 步骤5：设置张量地址（TensorRT 10.x 新 API）
        if (!context->setTensorAddress(input_tensor_name.c_str(), buffers[0])) {
            std::cerr << "Failed to set input tensor address" << std::endl;
            return {};
        }
        if (!context->setTensorAddress(output_tensor_name.c_str(), buffers[1])) {
            std::cerr << "Failed to set output tensor address" << std::endl;
            return {};
        }

        // 步骤6：设置输入形状（对于动态形状必需）
        auto input_dims = engine->getTensorShape(input_tensor_name.c_str());
        if (!context->setInputShape(input_tensor_name.c_str(), input_dims)) {
            std::cerr << "Failed to set input shape" << std::endl;
        }

        // 步骤7：执行推理（这是最关键的一步）
        /* 面试重点：TensorRT 推理执行
         * 1. enqueueV3: 异步执行，高性能
         * 2. executeV2: 同步执行，简单但慢
         * 3. 流式处理：允许 CPU-GPU 并行计算
         */
        bool success = context->enqueueV3(stream);
        if (!success) {
            std::cerr << "Inference failed!" << std::endl;
            return {};
        }

        // 等待推理完成
        cudaStreamSynchronize(stream);

        // 步骤8：读取推理结果（GPU -> CPU）
        auto output_dims = engine->getTensorShape(output_tensor_name.c_str());
        size_t output_size = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size *= output_dims.d[i];
        }

        // 从 GPU 读取 FP16 结果
        std::vector<__half> output_half(output_size);
        cudaMemcpy(output_half.data(), buffers[1], output_size * sizeof(__half), cudaMemcpyDeviceToHost);

        // 转换为 FP32 以便后处理
        std::vector<float> output(output_size);
        for (size_t i = 0; i < output_size; ++i) {
            output[i] = __half2float(output_half[i]);
        }

        // 调试信息：打印输出统计
        std::cout << "Output size: " << output_size << std::endl;
        std::cout << "First few output values: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;

        // 步骤9：计算缩放比例（将归一化坐标转换为绝对像素）
        float scale_x = static_cast<float>(image.cols) / input_size;
        float scale_y = static_cast<float>(image.rows) / input_size;

        // 步骤10：后处理得到最终检测结果
        return postprocess(output, scale_x, scale_y);
    }

    // ========== 可视化方法：绘制检测结果 ==========
    /* 面试考点：计算机视觉可视化技巧
     * 1. 边界框绘制：rectangle 函数绘制矩形
     * 2. 文字标签：显示类别名称和置信度
     * 3. 颜色选择：绿色边框，黑色文字，绿色背景
     * 4. 扩展：可为不同类别设置不同颜色
     */
    void drawDetections(Mat& image, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            // 创建 OpenCV 矩形对象
            Rect box(det.x, det.y, det.w, det.h);
            // 绘制绿色边界框，线宽 2 像素
            rectangle(image, box, Scalar(0, 255, 0), 2);

            // 构建标签文字：类别名 + 置信度百分比
            std::string label = class_names[det.class_id] + " " +
                               std::to_string(static_cast<int>(det.confidence * 100)) + "%";

            // 计算文字尺寸，用于背景矩形
            int baseline;
            Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            // 绘制文字背景矩形（绿色填充）
            rectangle(image, Point(det.x, det.y - text_size.height - 5),
                     Point(det.x + text_size.width, det.y), Scalar(0, 255, 0), FILLED);

            // 绘制文字标签（黑色文字）
            putText(image, label, Point(det.x, det.y - 5),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }
    }
}; // 类定义结束

// ========== 主函数：程序入口 ==========
/* 面试重点：完整的深度学习应用流程
 * 1. 参数解析和验证
 * 2. 模型初始化和加载
 * 3. 数据预处理和推理
 * 4. 结果后处理和可视化
 * 5. 性能测量和统计
 */
int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // 解析命令行参数
    std::string image_path = argv[1];   // 输入图像路径
    std::string onnx_path = "yolov5s.onnx";  // ONNX 模型文件路径

    // 创建检测器实例
    YOLOv5TensorRT detector;

    // 步骤1：构建 TensorRT 引擎（这一步耗时较长）
    std::cout << "Building TensorRT engine from ONNX..." << std::endl;
    if (!detector.buildEngine(onnx_path)) {
        std::cerr << "Failed to build engine" << std::endl;
        return -1;
    }

    // 步骤2：加载输入图像
    Mat image = imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // 步骤3：执行推理并测量性能
    /* 面试知识点：性能测量最佳实践
     * 1. 使用高精度时钟：std::chrono::high_resolution_clock
     * 2. 排除冷启动影响：第一次推理通常较慢
     * 3. 多次测量取平均值：获得更稳定的性能数据
     */
    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto detections = detector.detect(image);
    auto end = std::chrono::high_resolution_clock::now();

    // 计算推理耗时（毫秒）
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    std::cout << "Detected " << detections.size() << " objects" << std::endl;

    // 步骤4：绘制检测结果
    detector.drawDetections(image, detections);

    // 步骤5：保存结果图像
    std::string output_path = "tensorrt_detection_result.jpg";
    imwrite(output_path, image);
    std::cout << "Result saved to: " << output_path << std::endl;

    return 0;  // 程序正常退出
}

/* ========== 面试总结和拓展 ==========
 *
 * 核心技术栈：
 * 1. TensorRT: NVIDIA GPU 推理优化库
 * 2. CUDA: 并行计算平台和编程模型
 * 3. OpenCV: 计算机视觉库
 * 4. YOLO: 实时目标检测算法
 *
 * 性能优化点：
 * 1. FP16 半精度：内存和计算速度优化
 * 2. GPU 内存管理：减少 CPU-GPU 数据传输
 * 3. 异步执行：CUDA 流并行处理
 * 4. 算子融合：TensorRT 自动优化
 *
 * 生产环境考虑：
 * 1. 引擎序列化：保存并复用 .trt 文件
 * 2. 批处理：多张图像同时推理
 * 3. 多线程：CPU 预处理 + GPU 推理并行
 * 4. 内存池：避免频繁内存分配释放
 *
 * 常见面试问题：
 * 1. TensorRT vs ONNX Runtime vs PyTorch 性能对比？
 * 2. 如何调试 TensorRT 模型推理结果？
 * 3. GPU 内存不足时如何优化？
 * 4. 如何实现实时视频流检测？
 * 5. 多模型集成和负载平衡策略？
 */