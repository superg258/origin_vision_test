#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace
{
std::filesystem::path find_repo_root()
{
  auto path = std::filesystem::current_path();
  while (!path.empty()) {
    if (std::filesystem::exists(path / "configs") && std::filesystem::exists(path / "assets")) {
      return path;
    }
    if (path == path.parent_path()) break;
    path = path.parent_path();
  }
  return std::filesystem::current_path();
}
}  // namespace

int main()
{
  const auto root = find_repo_root();
  const std::vector<std::string> buff_configs = {
    "configs/ascento.yaml", "configs/hero.yaml", "configs/sentry.yaml",
    "configs/standard3.yaml", "configs/standard4.yaml", "configs/uav.yaml"};

  bool ok = true;
  for (const auto & config : buff_configs) {
    const auto path = root / config;
    const auto yaml = YAML::LoadFile(path.string());
    const auto model = yaml["model"].as<std::string>("");
    if (model != "assets/buff_repvgg.xml") {
      std::cerr << config << " model is '" << model << "', expected assets/buff_repvgg.xml\n";
      ok = false;
    }
    if (!yaml["ConfidenceThreshold"]) {
      std::cerr << config << " missing ConfidenceThreshold\n";
      ok = false;
    }
    if (!yaml["IouThreshold"]) {
      std::cerr << config << " missing IouThreshold\n";
      ok = false;
    }
  }

  for (const auto & asset : {"assets/buff_repvgg.xml", "assets/buff_repvgg.bin"}) {
    if (!std::filesystem::exists(root / asset)) {
      std::cerr << asset << " is missing\n";
      ok = false;
    }
  }

  return ok ? 0 : 1;
}
