# Template Run Commands

```bash
JAX_PLATFORMS=cpu uv run train/nash_pg.py \
                    algorithm.num_inner_update=100 \
                    algorithm.num_outer_update=10 \
                    algorithm.mag_coef=0.2 \
                    logging.save_interval=100 \
                    env=robot_warehouse/tiny_4ag \
                    agent=robot_warehouse/tiny \
                    run_name=robot_warehouse/nash_pg/default_run
```