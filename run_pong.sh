python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True post_synthesis_mode=nothing

python inspect_model.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True post_synthesis_mode=nothing


# Start at 50
python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=20 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=30 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=50 imagined_atari_env.time_lag=0.1

# python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=65 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=85 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=100 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=280 imagined_atari_env.time_lag=0.1

# agent

python run.py task=Pong obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True post_synthesis_mode=agent debug_mode=True agent.n_goals_to_achieve=50

# BASIC2

# Start at 50
python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=20 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=30 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=50 imagined_atari_env.time_lag=0.1

# python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=65 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=85 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=100 imagined_atari_env.time_lag=0.1

python run.py task=Pong obs_suffix=_basic2 world_model_learner.obj_type=all det_world_model=True imagined_atari_env.start_at=280 imagined_atari_env.time_lag=0.1

# run world model
python run.py --config-name=pong
# agent
python run.py --config-name=pong_agent

python run.py --config-name=pong_agent method=worldcoder
python run.py --config-name=pong_alt_agent method=worldcoder
