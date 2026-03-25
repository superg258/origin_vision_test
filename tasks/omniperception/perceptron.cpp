#include "perceptron.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{
Perceptron::Perceptron(
  io::USBCamera * usbcam1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
  io::USBCamera * usbcam4, const std::string & config_path)
: Perceptron(
    std::vector<WorkerConfig>{
      WorkerConfig{usbcam1, "left", std::nullopt, "", {}},
      WorkerConfig{usbcam2, "right", std::nullopt, "", {}},
      WorkerConfig{usbcam3, "back", std::nullopt, "", {}},
      WorkerConfig{usbcam4, "back", std::nullopt, "", {}}},
    config_path)
{}

Perceptron::Perceptron(const std::vector<WorkerConfig> & workers, const std::string & config_path)
: detection_queue_(10), worker_configs_(workers), decider_(config_path), stop_flag_(false)
{
  debug_snapshots_.resize(worker_configs_.size());
  yolo_workers_.reserve(worker_configs_.size());

  for (size_t i = 0; i < worker_configs_.size(); ++i) {
    const auto & worker = worker_configs_[i];
    if (worker.infer_device_key.empty()) {
      yolo_workers_.push_back(std::make_shared<auto_aim::YOLO>(config_path, false));
    } else {
      yolo_workers_.push_back(
        std::make_shared<auto_aim::YOLO>(config_path, false, worker.infer_device_key));
    }

    if (worker.camera_spec.has_value()) {
      debug_snapshots_[i].spec = worker.camera_spec.value();
    } else {
      debug_snapshots_[i].spec.label = worker.camera_label;
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));
  for (size_t i = 0; i < worker_configs_.size(); ++i) {
    threads_.emplace_back([this, i] { parallel_infer(worker_configs_[i], yolo_workers_[i]); });
  }

  tools::logger()->info("Perceptron initialized with {} workers.", worker_configs_.size());
}

Perceptron::~Perceptron()
{
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_flag_ = true;  // 设置退出标志
  }
  condition_.notify_all();  // 唤醒所有等待的线程

  // 等待线程结束
  for (auto & t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  tools::logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::vector<DetectionResult> result;
  DetectionResult temp;

  // 注意：这里的 pop 不阻塞（假设队列为空时会报错或忽略）
  while (!detection_queue_.empty()) {
    detection_queue_.pop(temp);
    result.push_back(std::move(temp));
  }

  return result;
}

std::vector<Perceptron::DebugSnapshot> Perceptron::debug_snapshots() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return debug_snapshots_;
}

// 将并行推理逻辑移动到类成员函数
void Perceptron::parallel_infer(
  const WorkerConfig & worker, std::shared_ptr<auto_aim::YOLO> & yolov8_parallel)
{
  auto * cam = worker.camera;
  if (!cam) {
    tools::logger()->error("Camera pointer is null!");
    return;
  }
  try {
    while (true) {
      cv::Mat usb_img;
      std::chrono::steady_clock::time_point ts;

      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_flag_) break;  // 检查是否需要退出
      }

      cam->read(usb_img, ts);
      const double base_yaw_rad = worker.base_yaw_provider ? worker.base_yaw_provider() : 0.0;
      if (usb_img.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        continue;
      }

      const auto infer_begin = std::chrono::steady_clock::now();
      auto armors = yolov8_parallel->detect(usb_img);
      const auto infer_end = std::chrono::steady_clock::now();
      const double infer_ms = tools::delta_time(infer_end, infer_begin) * 1e3;
      std::optional<auto_aim::Armor> top_armor;
      if (!armors.empty()) top_armor = armors.front();

      double delta_yaw_deg = 0.0;
      double delta_pitch_deg = 0.0;
      if (!armors.empty()) {
        Eigen::Vector2d delta_angle;
        if (worker.camera_spec.has_value()) {
          delta_angle = decider_.delta_angle(armors, worker.camera_spec.value());
        } else {
          const std::string camera_label =
            worker.camera_label.empty() ? cam->device_name : worker.camera_label;
          delta_angle = decider_.delta_angle(armors, camera_label);
        }
        delta_yaw_deg = delta_angle[0];
        delta_pitch_deg = delta_angle[1];

        DetectionResult dr;
        dr.armors = std::move(armors);
        dr.timestamp = ts;
        dr.delta_yaw = delta_yaw_deg / 57.3;
        dr.delta_pitch = delta_pitch_deg / 57.3;
        dr.slot = worker.camera_spec.has_value() ? worker.camera_spec->slot : OmniCameraSlot::unknown;
        dr.camera_label = worker.camera_spec.has_value() ? worker.camera_spec->label : worker.camera_label;
        dr.base_yaw_rad = base_yaw_rad;
        dr.has_base_yaw = static_cast<bool>(worker.base_yaw_provider);
        dr.infer_ms = infer_ms;
        detection_queue_.push(dr);  // 推入线程安全队列
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(
          worker_configs_.begin(), worker_configs_.end(),
          [&](const auto & candidate) { return candidate.camera == worker.camera; });
        if (it != worker_configs_.end()) {
          const size_t index = static_cast<size_t>(std::distance(worker_configs_.begin(), it));
          auto & snapshot = debug_snapshots_[index];
          if (worker.camera_spec.has_value()) {
            snapshot.spec = worker.camera_spec.value();
          } else {
            snapshot.spec.label = worker.camera_label;
          }
          snapshot.image = usb_img.clone();
          snapshot.timestamp = ts;
          snapshot.top_armor = top_armor;
          snapshot.delta_yaw_deg = delta_yaw_deg;
          snapshot.delta_pitch_deg = delta_pitch_deg;
          snapshot.infer_ms = infer_ms;
          snapshot.base_yaw_rad = base_yaw_rad;
          snapshot.has_base_yaw = static_cast<bool>(worker.base_yaw_provider);
          snapshot.has_detection = top_armor.has_value();
        }
      }
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Exception in parallel_infer: {}", e.what());
  }
}

}  // namespace omniperception
