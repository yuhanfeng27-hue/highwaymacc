# Vehicle attention actor

The default CACC Hydra configuration now selects `model=layers/entityattention`;
`critic_model` remains `layers/mlp`. The environment, rewards, sharing setting,
and illegal-action masking are unchanged. There are no new dependencies.

For each agent separately, the actor reshapes the flattened PlatoonObservation
into vehicle entities `[entity_type, x, y, vx, vy]`, with ego first. A shared
entity encoder produces 32-dimensional embeddings. Ego queries the visible
entities through four-head attention. Slots with entity_type=0 are masked.
A residual connection retains ego information; LayerNorm produces one 32-D
vector, which a [256, 256] MLP maps to action logits. The existing MAPPO actor
applies the action mask and builds the action distribution. All layers learn
jointly through the PPO loss. Other agents' observation tensors are never read
by this decentralized encoder. Shared policy parameters remain the default;
independent parameters are also supported.

From the repository root, with the development Python environment activated:

```bash
python benchmarl/run.py task=myenv/stage1 model=layers/entityattention
python benchmarl/run.py task=myenv/stage2 model=layers/entityattention
# Baseline comparison (also select MLP for non-vehicle environments):
python benchmarl/run.py task=myenv/stage1 model=layers/mlp
```

The original experiment defaults request CUDA and wandb. For CPU/local logging,
append `experiment.sampling_device=cpu experiment.train_device=cpu
experiment.buffer_device=cpu 'experiment.loggers=[csv]'`. In a headless machine
use SDL_VIDEODRIVER=dummy and SDL_AUDIODRIVER=dummy. See the environment startup
instructions for the tested short-training command.

Configure `model.embed_dim=32 model.num_heads=4` to change attention capacity;
embed_dim must be positive and divisible by num_heads. This model is only for
decentralized vehicle actors, not centralized critics or arbitrary observations.
It accepts different entity counts, but the environment and its declared
observation_space must still agree. The current MyCustomEnv2 wrapper declares
40 features; increasing its HDV count requires a separate coordinated change.

Existing MLP checkpoints cannot initialize the new actor unchanged. Start a new
attention experiment, and use attention checkpoints for subsequent curriculum
warm starts. `run_curriculum.py` contains a hardcoded old checkpoint path: update
it to a compatible attention checkpoint before using that entry point. To use an
old MLP checkpoint, retain the MLP architecture.

A successful short run proves integration, not better learning. Compare MLP and
attention under identical rewards, seeds, frame budgets and evaluation settings.
Track formation success, collision rate, recovery time, lane changes and elapsed
time. The pre-existing reward issues have not been modified in this change.
