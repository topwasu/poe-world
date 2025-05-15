python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 load_obj_model=saved_checkpoints_Pitfall_pomdp1/reasoner_player_100.pickle det_world_model=True run_world_model=True
python inspect_model.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 load_obj_model=saved_checkpoints_Pitfall_pomdp1/reasoner_player_100.pickle det_world_model=True run_world_model=True

python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 load_obj_model=saved_checkpoints_Pitfall_pomdp1/reasoner_player_120.pickle det_world_model=True run_world_model=True
python inspect_model.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 load_obj_model=saved_checkpoints_Pitfall_pomdp1/reasoner_player_120.pickle det_world_model=True run_world_model=True

# Full game
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 world_model_learner.obj_type=all

# Second room creation
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=95 obs_index_length=10 world_model_learner.obj_type=all
python inspect_model.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=95 obs_index_length=10 world_model_learner.obj_type=player
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=95 obs_index_length=10 world_model_learner.obj_type=all det_world_model=True run_world_model=True

# Until second room
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=0 obs_index_length=120 world_model_learner.obj_type=all
# inspect
python inspect_model.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=0 obs_index_length=120 world_model_learner.obj_type=all
# run
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=0 obs_index_length=120 world_model_learner.obj_type=all det_world_model=True run_world_model=True

# Rope swinging
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=218 obs_index_length=168 world_model_learner.obj_type=rope det_world_model=True run_world_model=True

# Player hanging on to the rope
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_rope obs_index=218 obs_index_length=339 world_model_learner.obj_type=player det_world_model=True run_world_model=True rope_mode=True
python inspect_model.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_rope obs_index=218 obs_index_length=339 world_model_learner.obj_type=player det_world_model=True run_world_model=True rope_mode=True

# Until third room, before hopping on the rope
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=0 obs_index_length=300 world_model_learner.obj_type=all

# Until third room, after getting off the rope
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_pomdp1 obs_index=0 obs_index_length=360 world_model_learner.obj_type=all

# Pitfall basic1
: '
[2025-02-21 18:06:11,021][main][INFO] - at idx 195: Portal interaction: player portal_1
[2025-02-21 18:06:11,023][main][INFO] - at idx 196: Portal interaction: player portal_2
[2025-02-21 18:06:11,185][main][INFO] - at idx 337: Portal interaction: player portal_3
[2025-02-21 18:06:11,186][main][INFO] - at idx 338: Portal interaction: player portal_4
[2025-02-21 18:06:11,461][main][INFO] - at idx 600: Portal interaction: player portal_5
[2025-02-21 18:06:11,462][main][INFO] - at idx 601: Portal interaction: player portal_6
[2025-02-21 18:06:11,582][main][INFO] - at idx 716: Portal interaction: player portal_7
[2025-02-21 18:06:11,583][main][INFO] - at idx 717: Portal interaction: player portal_7
[2025-02-21 18:06:11,828][main][INFO] - at idx 886: Portal interaction: player portal_9
[2025-02-21 18:06:11,829][main][INFO] - at idx 887: Portal interaction: player portal_9
[2025-02-21 18:06:11,995][main][INFO] - at idx 1057: Portal interaction: player portal_11
[2025-02-21 18:06:11,996][main][INFO] - at idx 1058: Portal interaction: player portal_12
'
# player only
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_basic1 world_model_learner.obj_type=player det_world_model=True run_world_model=True
# all
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_basic1 world_model_learner.obj_type=all det_world_model=True run_world_model=True

# all before fourth room (after rope)
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_basic1 obs_index=0 obs_index_length=600 world_model_learner.obj_type=all det_world_model=True run_world_model=True
python inspect_model.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_basic1 obs_index=0 obs_index_length=600 world_model_learner.obj_type=all det_world_model=True run_world_model=True

# tarpit
python run.py task=Pitfall database_path=completions_atari_pitfall.db obs_suffix=_tarpit obs_index=141 obs_index_length=268 world_model_learner.obj_type=disappearingtarpit det_world_model=True run_world_model=True