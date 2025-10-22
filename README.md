# Template Run Commands

Useful stuff
```bash
JAX_PLATFORMS=cpu 
CUDA_VISIBLE_DEVICES=0
```

Robot Warehouse
```bash
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.05 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=robot_warehouse/tiny_2ag \
    agent=robot_warehouse/tiny \
    run_name=robot_warehouse/nash_pg/default_run

uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.0 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=robot_warehouse/tiny_4ag \
    agent=robot_warehouse/tiny \
    run_name=robot_warehouse/ippo/default_run
```

Connector
```bash
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.05 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=connector/grid10_10ag \
    agent=connector/tiny_10ag \
    run_name=connector_10ag/nash_pg/default_run

uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.0 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=connector/grid10_10ag \
    agent=connector/tiny_10ag \
    run_name=connector_10ag/ippo/default_run
```

LBF
```bash
CUDA_VISIBLE_DEVICES=2 uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.05 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=lbf/tiny_3ag \
    agent=lbf/large \
    run_name=lbf_tiny_3ag/nash_pg/default_run

CUDA_VISIBLE_DEVICES=3 uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.0 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=lbf/tiny_3ag \
    agent=lbf/large \
    run_name=lbf_tiny_3ag/ippo/default_run
```

## Visiualize
```bash
uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/robot_warehouse/nash_pg/default_run --step 10000 --env-config conf/env/robot_warehouse/tiny_4ag.yaml --fps 8 --seed 100

uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/connector_10ag/nash_pg/default_run --step 1000 --env-config conf/env/connector/grid10_10ag.yaml --seed 100

uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/lbf_tiny_3ag/nash_pg/default_run --step 10000 --env-config conf/env/lbf/tiny_3ag.yaml --seed 100
```


JAX_PLATFORMS=cpu uv run train/nash_pg.py \
    algorithm.num_inner_update=100 \
    algorithm.num_outer_update=10 \
    algorithm.mag_coef=0.0 \
    logging.save_interval=-1 \
    logging.log_interval=1 \
    env=unity/editor \
    agent=unity/3d_ball \
    run_name=unity/3d_ball/ippo/norm_log_prob

JAX_PLATFORMS=cpu uv run train/nash_pg.py \
    algorithm.num_inner_update=100 \
    algorithm.num_outer_update=10 \
    algorithm.num_envs=16 \
    algorithm.num_steps=64 \
    algorithm.gamma=0.99 \
    algorithm.normalize_logprob=True \
    algorithm.mag_coef=0.0 \
    logging.save_interval=-1 \
    logging.log_interval=1 \
    env.time_scale=16.0 \
    env=unity/3d_ball \
    agent=unity/3d_ball \
    run_name=unity/3d_ball/ippo/build