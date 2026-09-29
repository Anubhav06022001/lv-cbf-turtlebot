# LV-CBF TurtleBot

This repository contains the implementation, training, evaluation, and experimental framework for **LV-CBF-based safe control of TurtleBot**.

## 📁 Project Structure

```text
lv-cbf-turtle_bot/
│
├── README.md                         # Project overview & installation
│
├── robotis_tb3/                      # Contains all XML files
│
├── config/
│   └── eval_grid.yaml                # Evaluation configuration
│
├── data/
│   └── intial_pos/
│       ├── start_positions_500.csv   # 500 initial positions
│       └── start_positions_100.csv   # 100 initial positions
│
├── experiments/
│   ├── baselines/
│   │   ├── collision_cone_500.py     # Collision-cone baseline
│   │   └── test_collision_cone.py    # Baseline evaluation
│   │
│   ├── data_collection/
│   │   ├── collect_lv_cbf_data.py   # LV-CBF data collection
│   │   └── generate_eval_dataset.py  # Evaluation dataset generation
│   │
│   └── lv_cbf_test/
│       ├── single_case_dynamic_turtlebot.py
│       ├── single_case_static_turtlebot.py
│       ├── test_dynamic_turtlebot.py
│       └── test_static_turtlebot.py
│
├── models/
│   └── drone_vocbf/                  # Saved model weights (.pt)
│
├── results/
│   ├── plots/                         # Evaluation plots and figures
│   └── kinematic_grid_search.csv     # Safety & reachability results
│
├── src/
│   ├── collision_cone/
│   │   └── collision_cone_cbf.py     # Collision-cone CBF implementation
│   │
│   ├── control/
│   │   └── cbf_qp_multi.py           # CBF-QP safety filter
│   │
│   ├── dynamics/
│   │   └── drone_dynamics.py         # System dynamics
│   │
│   ├── utils/
│   │   ├── env_setup.py
│   │   ├── env_utils.py
│   │   ├── eval_metrics.py
│   │   ├── plot_metrics.py
│   │   └── plot_utils.py
│   │
│   └── lvcbf/
│       ├── cbf_network.py
│       ├── expectile.py
│       ├── train_dynamic_vocbf.py
│       ├── train_vision_vocbf_turtlebot
│       ├── train_vocbf.py
│       └── vision_cbf_network.py
│
└── .gitignore
```
