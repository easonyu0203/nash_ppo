# Template Run Commands

Useful stuff
```bash
JAX_PLATFORMS=cpu 
CUDA_VISIBLE_DEVICES=0
```

For Nash Policy Gradient
```bash
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=100 \
    algorithm.mag_coef=0.2 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=robot_warehouse/tiny_4ag \
    agent=robot_warehouse/tiny \
    run_name=robot_warehouse/nash_pg/default_run
```

Indenpendent PPO
```bash
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=100 \
    algorithm.mag_coef=0.0 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=robot_warehouse/tiny_4ag \
    agent=robot_warehouse/tiny \
    run_name=robot_warehouse/ippo/default_run
```