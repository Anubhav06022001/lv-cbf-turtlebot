lv-cbf-turtle_bot/
|
├── README.md                   # Project overview & installation
|
├── robotis_tb3              # contains all xml                      
|
├── config/                       
│   └── eval_grid.yaml
│       
├── data/                       
│   └── intial_pos/
│        ├── start_positions_500.csv
|        └── start_positions_100.csv
│
├── experiments/
|        |  
|        ├── baselines
|        |       ├── collision_cone_500.py
|        |       └── test_collision_cone.py 
|        |  
│        ├── data_collection
|        |       ├── collect_lv_cbf_data.py 
|        |       └── generate_eval_dataset.py    
|        |
|        └── lv_cbf_test
|              ├── single_case_dynamic_turtlebot.py 
|              ├── single_case_static_turtlebot.py 
|              ├── test_dynamic_turtlebot.py                            
|              └── test_static_turtlebot.py
|
├── models/                     # Saved .pt weights
│   └── drone_vocbf
|
├── results/   
│   ├── plots/         # contains relevant plots and figures related to project
│   └── kinematic_grid_search.csv     # contains safety and rechability rate of kinematic_yaml.py for different hyperparameters          
│   
├── src/
│   ├── collision_cone/          # collision_cone_cbf.py contains implementation of collision cone baseline
│   │
│   ├── control/
│   │       └──cbf_qp_multi.py    # QP Filter: min ||u - u_vla|| subject to CBF
|   |
│   ├── dynamics/
|   |       └── drone_dynamics.py 
│   │
│   ├── utils/
│   │   ├── env_setup.py
│   │   ├── env_utils.py
│   │   ├── eval_metrics.py
│   │   ├── plot_metrics.py   
│   │   └── plot_utils.py        
|   | 
│   └──lvcbf/
│       ├── cbf_network.py
│       ├── expectile.py
│       ├── train_dynamic_vocbf.py 
│       ├── train_vision_vocbf_turtlebot
│       ├── train_vocbf.py
|       └── vision_cbf_network.py
│   
└── .gitignore