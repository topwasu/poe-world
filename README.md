# [PoE-World: Compositional World Modeling with Products of Programmatic Experts](https://arxiv.org/abs/2505.10819)

By [Wasu Top Piriyakulkij](https://www.cs.cornell.edu/~wp237/), [Yichao Liang](https://yichao-liang.github.io/), [Hao Tang](https://haotang1995.github.io/), [Adrian Weller](https://mlg.eng.cam.ac.uk/adrian/), [Marta Kryven](https://marta-kryven.github.io/), [Kevin Ellis](https://www.cs.cornell.edu/~ellisk/)

[![deploy](https://img.shields.io/badge/Project_Page%20%20-8A2BE2)](https://topwasu.github.io/poe-world) [![arXiv](https://img.shields.io/badge/arXiv-2401.02739-red.svg)](https://arxiv.org/abs/2505.10819)


We introduce a novel program synthesis approach to output world models of complex, non-gridworld domains by representing world models as products of programmatic experts.

## Installation

First clone this project and cd to it 
```
git clone https://github.com/topwasu/poe-world.git
cd poe-world
```

Create conda environment 
```
conda create -n poeworld python=3.10
conda activate poeworld
```

Install requirements 
```
pip install -r requirements.txt
```

Initialize all submodules 
```
git submodule update --init --recursive
```

Install the submodule `openai-hf-interface` by calling 
```
cd openai-hf-interface
pip install -e .
# Important: you also need to create a file (secrets.json) containing your OpenAI's API key here -- see instruction below
cd ..
```
`openai-hf-interface` is a package that provides nice abstractions for calling OpenAI's api, so you need to input your OpenAI's API key. To do that, create a file called `secrets.json` inside `openai-hf-interface` directory and set the key called `openai_api_key` to your OpenAI's API key value. See [the package's repo](https://github.com/topwasu/openai-hf-interface) for more information.

Install the submodule OCAtari
```
cd OC_Atari
python setup.py develop 
pip install "gymnasium[atari, accept-rom-license]"
cd ..
```

## Running

Running PoE-World
```
python make_observations.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
python run.py --config-name=pong_agent # choose config-name from [pong_agent, pong_alt_agent, montezuma_agent, montezuma_alt_agent]
```

Running WorldCoder
```
python make_observations.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
python run.py --config-name=pong_agent # choose config-name from [pong_agent, pong_alt_agent, montezuma_agent, montezuma_alt_agent]
```

Running ReAct
```
python run_react.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
```

Running PPO
```
python run_rl.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
```

## Example learned PoE-World world models

**mr_world_model_seed0.txt** and **pong_world_model_seed0.txt** contain learned PoE-World world models for Montezuma's Revenge and Pong, respectively.

## Important Files

**agents/agent.py**
Contains the implementation of the main agent class, which is responsible for interacting with the environment, calling planning algorithms, and calling functions to update world models.

**agents/mcts.py**
Implements the Monte Carlo Tree Search (MCTS) algorithm, which is used by the agent to plan motions.

**classes/envs**
Contains the environment classes in the style of openai's gym

**classes/helper.py**
Contains various helper classes that are interfaces to game objects (Obj), their interactions (Interaction), game states (ObjList), etc.

**learners/world_model_learner.py**
Implements the world model learner, which calls the obj model learner for all object types.

**learners/obj_model_learner.py**
Implements the object model learner, which calls synthesizers to get programs and calls MoEObjModel in learners/models.py to fit the weights of the programs.

**learners/synthesizer.py**
Contains modules that synthesize programs based on observation

**learners/models.py**
Contains classes that let us fit the weights of the programs