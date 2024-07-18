# What this is
These are some notes on how the original research code works that so its easy to modify and debug when it inevitably breaks.

# Docs
So it seems that `accelerate_redteam_ppo_trainer.py` is the primary part of this repositorry, while `self_bleu.py` and `sentence_embed.py` offer a sort of unified interface for rewards, in which you append "references" and then call an object with those references to calculate a reward.

## Runnables
Generally in root we have all these `ppo_*_...py` files, which seem to primarily always consistent of the same thing:
```
def default_redteam_ppo_config():
    ... # return a TRLX config
class RedTeamToxicityRewardModel(object):
    ... # object thatt is also a reward function
    ... # for TRLX (somehow)
def main(hparams={}):
    ... # merge hyperparams with the default config
    ... # set up output directories and files
    ... # call TRLX train

# and in general this will use accelerate_redteam_ppo_trainer.py
```

## Runners
These are in turn called by all the "runners" that are inside `experiments/`. The runners appear to actually defiine some hyperparameters to grid-search or sweep, as well as a script which ends up being run by:
```
|
- launch_jobs(...)
    |
    -  launch(...)
         |
         - local(...)
```
(in `experiments/__init__.py`, which contains the core logic to run the actual experiments)

Runners always have the same core structure:
```
if __name__ == '__main__':
  # parse arguments using experiments/__init__.py
  # define the child subprocess script to run
  # define a grid-test
  # go through all jobs with sweep_with_devices (I don't 100% understand this)
  # set up logging, etc... as far as is necessary
  # parse jobs
  # call all jobs using launch_jobs() from experiments/__init__.py
```

## accelerate_redteam_ppo_trainer
This uses `self_bleu.py` and `sentence_embed.py` to define some value functions. Its only export is `RedteamPPOConfig`. This config is used for `AccelerateRedteamPPOTrainer`. This trainer seems like it's probably called by TRLX. TRLX is probably doing it due to `PYTHONPATH` being modified to include `accelerate_redteam_ppo_trainer` and probably due to the customization Zhang-Wei made. The runner uses the `TextCSVLogger*` objects to log prompts and responses + scores while it trains. It makes use of the rewards/penalties as described before (plus one more): `SelfBleuReward`, `CosineSentenceEmbeddingReward`, and `GiberishPenalty`. Its methods seem to most likely be used by TRLX. The main functionality seems to be the following:
- `evaluate`: for each sweep value (?) it will tokenize all samples, then calculatet the rewards, do some parsing/formatting, and log "metrics" (?). After it's done with this, if its the main process (?) it will display results of some kind. It's unclear why the evaluation only happens if is main process as it does.
- `make_experience` specifically for the main process collect rewards using the stuff mentioned above, then calculate some KLs using `_process_element` (apprently this involves a certain type of reduction in distributed systems?) then push a set of `PPORLElement` to a certain store (?); `_aggregate_traj_reward` is used to seemingly combine scores by taking a linear combination based on the function that is being maximized.

## custom_trlx
Just a slightly modified (I'm not sure what they actually did concretely, though) version of TRLX (which is a little dated) to enable rapid training using RLHF with low effort.

## prompts
These are used by runnables

## scripts
One-offs that are used to make prompts or datasets or other such stuff. These are used before anything else to create promtps and datasets AFAIK.

## TLDR
First, you use the scripts to set up your prompts, datases, etc... Then you will do experiments.

During an experiment, each runner invokes the experiments library to run a subprocess.

That subprocess (runnable):
1. invokes `accelerate_redteam_ppo_trainer.py` as a library, but just to load `RedteamPPOConfig`
2. Sets up default hyperparameters and merges them with the arguments
3. Use prompts
4. Calls TRLX

The library `accelerate_redteam_ppo_trainer` uses `self_bleu.py` and `sentence_embed.py` to define some value functions. Its only important export is `RedteamPPOConfig`. This config is meant to be used by `AccelerateRedteamPPOTrainer` which is probably called by `TRLX`.

Why a subprocess call? Unclear, but maybe parallelism?