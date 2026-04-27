# navigation_model

RL navigation model library for `goal_seeker_rl`.

Default model:

```bash
navigation_model/td3_latest.pth
```

Docker uses:

```bash
RL_BASE_MODEL_DIR=/workspace/navigation_model
```

Local scripts use the repo path by default. You can override it with:

```bash
RL_BASE_MODEL_DIR=/path/to/navigation_model scripts/launch_rl_base_clean_venv.sh
```

Select another model:

```bash
scripts/docker_run_stretch3.sh model_name:=td3_stable_lidar_auto_20260423.pth
scripts/launch_rl_base_clean_venv.sh model_name:=td3_step_3600.pth
```

Training checkpoints from `start_goal_seeker.launch.py` are saved here by default.

Continued Realsense training:

```bash
scripts/train_realsense_td3_clean_venv.sh
```

Default continued-training output:

```bash
navigation_model/realsense_td3_current/td3_latest.pth
```

Use that model:

```bash
scripts/docker_run_stretch3.sh model_name:=realsense_td3_current/td3_latest.pth
scripts/launch_rl_base_clean_venv.sh model_name:=realsense_td3_current/td3_latest.pth
```

Verify model files:

```bash
sha256sum -c navigation_model/MODEL_MANIFEST.sha256
```
