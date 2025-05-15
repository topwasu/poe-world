python run.py obs_suffix=_basic3 run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_80_basic3_new.pickle det_world_model=False
python run.py obs_suffix=_basic3 run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_100_basic3_new.pickle det_world_model=False
python run.py obs_suffix=_basic3 run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_120_basic3_new.pickle det_world_model=False
python run.py obs_suffix=_basic3 run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_140_basic3_new.pickle det_world_model=False
python run.py obs_suffix=_basic3 run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_200_basic3_new.pickle det_world_model=False
python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic4.pickle det_world_model=True obs_suffix=_basic4
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic4.pickle det_world_model=True obs_suffix=_basic4
python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic4_across_the_room_and_back_v2.pickle det_world_model=True obs_suffix=_basic4
python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic5.pickle det_world_model=True obs_suffix=_basic5
python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic5_twofall.pickle det_world_model=True obs_suffix=_basic5
python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic5_good.pickle det_world_model=True obs_suffix=_basic5

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic5_good.pickle det_world_model=True obs_suffix=_basic5
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic5_good_new.pickle det_world_model=True obs_suffix=_basic5

python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final.pickle det_world_model=True obs_suffix=_basic6

python run.py run_world_model=True load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic6_frameskip.pickle det_world_model=True obs_suffix=_basic6

python run.py load_obj_model=saved_checkpoints_basic5/reasoner_player_20.pickle det_world_model=True obs_suffix=_basic5


python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic7.pickle det_world_model=True obs_suffix=_basic7 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic8_test.pickle det_world_model=True obs_suffix=_basic8 run_world_model=True
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic8.pickle det_world_model=True obs_suffix=_basic8 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic8_good.pickle det_world_model=True obs_suffix=_basic8 run_world_model=True
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic8_good.pickle det_world_model=True obs_suffix=_basic8

python run.py load_obj_model=saved_checkpoints_basic5/reasoner_player_final.pickle det_world_model=True obs_suffix=_basic5

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic5_good.pickle det_world_model=False obs_suffix=_basic5 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_380.pickle det_world_model=True obs_suffix=_basic5 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_640.pickle det_world_model=True obs_suffix=_basic9 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_660.pickle det_world_model=True obs_suffix=_basic9 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9.pickle det_world_model=True obs_suffix=_basic9 run_world_model=True
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9_new.pickle det_world_model=True obs_suffix=_basic9 run_world_model=True
# best
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9_newest.pickle det_world_model=True obs_suffix=_basic9 run_world_model=True
python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9_newest.pickle det_world_model=True obs_suffix=_basic9 run_world_model=False

# python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic10_new.pickle det_world_model=True obs_suffix=_basic10 run_world_model=True
# python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic10_newest.pickle det_world_model=True obs_suffix=_basic10 run_world_model=True

python run.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9_v2.pickle obs_suffix=_basic9_v2

python inspect_model.py load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9_v2.pickle obs_suffix=_basic9_v2

python run.py obs_suffix=_basic9_v2 world_model_learner.obj_type=skull post_synthesis_mode=nothing

python run.py obs_suffix=_basic15 world_model_learner.obj_type=skull imagined_atari_env.time_lag=0.03
python inspect_model.py obs_suffix=_basic15 world_model_learner.obj_type=skull imagined_atari_env.time_lag=0.03


python run.py load_obj_model="saved_checkpoints_MontezumaRevenge_basic15 bad last fit/player/300.pickle" world_model_learner.obj_type=player imagined_atari_env.time_lag=0.03 obs_suffix=_basic15

# run all
python run.py obs_suffix=_basic15 world_model_learner.obj_type=all

# run player
python run.py obs_suffix=_basic15 world_model_learner.obj_type=player imagined_atari_env.time_lag=0.03

# inspect player
python inspect_model.py obs_suffix=_basic15 world_model_learner.obj_type=player

# player dying -- restart
python run.py obs_suffix=_basic15 world_model_learner.obj_type=player obs_index=613 obs_index_length=0

# run world model
python run.py --config-name=montezuma
# agent
python run.py --config-name=montezuma_agent

tensorboard --logdir baseline_logs/MontezumaRevenge-s1
tensorboard --logdir baseline_logs

python run.py --config-name=montezuma_agent method=worldcoder
python run.py --config-name=montezuma_alt_agent method=worldcoder


python run.py --config-name=montezuma_agent method=worldcoder agent.quick_build_graph=True


python run.py --config-name=montezuma_agent post_synthesis_mode=evaluate
python run.py --config-name=montezuma_alt_agent post_synthesis_mode=evaluate
python run.py --config-name=montezuma_agent method=worldcoder post_synthesis_mode=evaluate
python run.py --config-name=montezuma_alt_agent method=worldcoder post_synthesis_mode=evaluate