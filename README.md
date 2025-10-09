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
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.05 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=lbf/large_6ag \
    agent=lbf/tiny \
    run_name=lbf_large_6ag/nash_pg/default_run

uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=20 \
    algorithm.mag_coef=0.0 \
    logging.save_interval=1000 \
    logging.log_interval=10 \
    env=lbf/large_6ag \
    agent=lbf/tiny \
    run_name=lbf_large_6ag/ippo/default_run
```

## Visiualize
```bash
uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/robot_warehouse/nash_pg/default_run --step 100000 --env-config conf/env/robot_warehouse/tiny_4ag.yaml --seed 100

uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/connector_10ag/nash_pg/default_run --step 1000 --env-config conf/env/connector/grid10_10ag.yaml --seed 100

uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/lbf_large_6ag/nash_pg/default_run --step 10000 --env-config conf/env/lbf/large_6ag.yaml --seed 100
```