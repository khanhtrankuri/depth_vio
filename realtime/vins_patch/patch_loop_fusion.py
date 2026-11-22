#!/usr/bin/env python3
"""Small fixes so loop_fusion (VINS-Fusion-ROS2-Humble) can run with ROS remappings.

1. main() counted ROS arguments (--ros-args -r ...) as positional and exited; use the non-ROS args only.
2. ~PoseGraph() detached a thread that was never started (std::system_error at exit).
3. Optional loop-closure tuning read from the config yaml (loop_skip_cnt, loop_skip_dis).
Idempotent; fails loudly if the upstream source changed.
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "loop_fusion" / "src"


def patch(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        sys.exit(f"patch_loop_fusion: anchor not found in {path}: {old[:60]!r}")
    path.write_text(text.replace(old, new, 1))


node = root / "pose_graph_node.cpp"
patch(node,
      "    if(argc != 2)\n",
      "    std::vector<std::string> plain_args = rclcpp::remove_ros_arguments(argc, argv);\n"
      "    if(plain_args.size() != 2)\n")
patch(node,
      "    string config_file = argv[1];\n    printf(\"config_file: %s\\n\", argv[1]);\n",
      "    string config_file = plain_args[1];\n    printf(\"config_file: %s\\n\", config_file.c_str());\n")
patch(node,
      "    int USE_IMU = fsSettings[\"imu\"];\n",
      "    if (!fsSettings[\"loop_skip_cnt\"].empty()) SKIP_CNT = (int)fsSettings[\"loop_skip_cnt\"];\n"
      "    if (!fsSettings[\"loop_skip_dis\"].empty()) SKIP_DIS = (double)fsSettings[\"loop_skip_dis\"];\n"
      "    int USE_IMU = fsSettings[\"imu\"];\n")
patch(root / "pose_graph.cpp",
      "    t_optimization.detach();\n",
      "    if (t_optimization.joinable())\n        t_optimization.detach();\n")
print("patched loop_fusion")
