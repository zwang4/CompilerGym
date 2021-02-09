# Introducing the Compiler Gym

The machine learning (ML) and compiler research communities barely talk to each other. When they do talk, it is often to use compilers to make ML workloads faster. 

Compiler Gym is a package that aims to direct important compiler optimization problems at the ML research community in a language and vocabulary that they are comfortable with. We hope to act as a catalyst for making compilers faster using ML.

With our package, building ML models for compiler research problems is as easy as building ML models to play video games. Here are some highlights of our package:
- **API:** uses the popular Gym interface from OpenAI. Use Python to write your agent
- **Datasets:** wraps real world programs (C++ programs, TensorFlow programs, programs from Github, etc.) and a mainstream compiler (LLVM).
- **Tasks and Actions:** interfaces the LLVM compiler for one compiler research problem :  *phase ordering* (more to come). It has a large discrete action space
- **Representations:** provides raw representations of progams, as well as multiple kinds of pre-computed features: you can focus on end-to-end deep learning or features + boosted trees.
- **Rewards:** provides appropriate reward functions and loss functions for one end-task, *code size reduction*, with more to come
- **Testing:** provides a validation process for correctness of results
- **Baselines:** provides some baselines and reports their performance
- **Competition:** provides a leaderboard for you to submit your results

![overview](https://facebookresearch.github.io/CompilerGym/_images/overview.png)

The rest of this article explores each of the features in more detail.

## The API

As with the [OpenAI Gym](https://gym.openai.com/), compiler optimization problems are exposed as an environment, that is created with `gym.make`. `Observations` are computer programs sampled from the datasets that we prepackage. `Rewards` are provided if the agent does a better job than the compiler's inbuilt heuristics.

```python
import compiler_gym
import gym
env = gym.make("llvm-autophase-ic-v0")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

We cover more of this in the [Getting Started](https://facebookresearch.github.io/CompilerGym/getting_started.html) documentation.

## Datasets: scale, diversity and complexity

We provide a few datasets that are pre-package and easy to download via our API.
In this initial release, we focus on C and C++ programs and use the LLVM compiler.

Dataset | Description | Num. Benchmarks 
--------|-------------|-----------------
BLAS | | |
cBench | single-threaded C programs with hot-loops | 23
Github | random code from github with diverse code complexity | 50,708
Linux | | 13,920
PolyBench | kernel-space programs | 27
MIBench |  | 40
NAS Parallel Benchmarks | programs with parallelism | 122
OpenCV | programs using OpenCV | 442
TensorFlow | random tensorflow graphs | 1985



To get an intuition of the complexity of the datasets, we can compute a very crude estimate based on the likelihood that a random search will beat the compiler's default heuristics within 30 seconds (on the task of code-size, as described later in the post). For each dataset, `difficulty = 1 - p/n`, where `n` is the number of trials (2000 in the given graph), and `p` is the number of trials that succeeded.

![dataset difficulty](https://github.com/soumith/CompilerGym/raw/development/docs/source/_static/img/dataset_difficulty.png)


## Input Representations

The default representation that LLVM uses internally is low-level and non-linear enough that it is not intuitively useful in it's raw form.

Hence, for `observations`, we provide functionality to extract low-dimensional features from the compiler literature in addition to making the raw LLVM intermediate representation.


Representation | Description | Dimensionality 
--------|-------------|-----------------
LLVM IR | raw IR from the compiler as a string | variable
Autophase | heuristic counting-based features | 56
Inst2Vec | embeds the IR as a sequence of vectors using pre-trained embeddings on LLVM programs | (num_statements x 200)
ProGraML | graph-based representation of the IR

A more detailed description of these representations, along with relevant references is described in the [Observation Spaces](https://facebookresearch.github.io/CompilerGym/llvm/index.html#observation-spaces) section of the documentation.


## Tasks, Action Spaces and Rewards

### Problem 1: Phase Ordering of compiler passes

The first problem we present to the ML community is a classic staple in the compiler research world.

The LLVM compiler runs a set of optimization passes to transform programs from an unoptimized intermediate representation to an optimized one.

The type of passes that are run and the order in which they are run is of importance. Some passes might even be run multiple times.

We focus on emitting an ordered list of passes to be run, optimizing for a `cost function` (also called `loss function`) such as reducing the instruction count, binary size or the runtime of the program.

### Rewards

In our first release, we provide rewards focused on code size reduction. We provide two reward spaces that give you rewards as the compiler generates a smaller program:

1. code size: computationally-expensive but provides exact rewards for the task
2. instruction count: a cheaper-to-compute proxy for code size

The initial reward is `0`. As the result of the current `action`, if you generate a program that is smaller by `20` instructions (with `instruction count` as the reward function), then you get a reward of `20`.

### Actions

The action that the Agent has to take at each `step` is *"what pass to run next?"*. Hence, the actions are discrete.

There are XXX discrete actions available, as [documented](https://facebookresearch.github.io/CompilerGym/llvm/index.html#action-space). An example of an action is to ask the compiler to run `dead-code elimination`, where the compiler removes code that has no effect on the output of the program.

To get an intuitive sense of the action space, we compute the rewards that are attributed to each action (while the agent runs over the cBench dataset).

![reward by action](https://github.com/soumith/CompilerGym/raw/development/docs/source/_static/img/reward_by_action.png)

The graph looks a bit hairy, but we can observe a couple of interesting things:
- the `reg2mem` pass seems to give the least reward, probably increasing the size of the program, and the `mem2reg` pass seems to have the exact opposite effect, giving the agent maximum reward.
- `instsimplify` seems to reduce the size of the program because it removes redundant instructions (by definition)
- `structurizecfg` seems to consistently give `0` reward, and that makes sense because it is simply re-ordering the control flow code in the IR, neither removing nor adding code

The reward spaces are [documented here](https://facebookresearch.github.io/CompilerGym/llvm/index.html#reward-spaces).

## Testing and Validation

An important observation about Phase Ordering is that the compiler can run an order of passes that compile the program but produce nonsense or null output. LLVM by default generates a correct ordering of passes. However, our agent being an ML-based agent, can easily generate something that maximizes reward but produces incorrect compiled code.

Here is a sequence of actions that produces code with 0 instructions (hence maximizing reward) but is effectively reducing the source code to a no-op:

[TODO fill the failure case]

For this reason, we need a validation for our agent against known results.

Regardless of what training data your agent was trained on, We provide a way to validate the correctness of your agent while running the `cBench` dataset.

[TODO: chris to fill this]

## Baselines, Leaderboards and Submitting Results

We provide two baselines as of now, on the `cBench` dataset.

| Method       | Inference time | Geomean reward |
|--------------|----------------|----------------|
| Actor Critic | 2,213 ms       | 0.910          |
| Random agent | 60,261 ms      | 0.800          |

[TODO: fill in the same baselines on some other datasets, like TensorFlow or Github]

[TODO: what is the reward for `-Oz`??]

Once you train an agent that performs better than the baseline, please validate it's correctness and then send us a PR adding an entry to [this](todo: link this to the leaderboard table) table. The PR should contain the CSV file generated during validation. Here is an [example PR](todo: fill this) showcasing a new submission.

## Closing Thoughts

As part of the initial release, we provide one important problem from the compiler research community and make it look like Machine Learning modeling problems. We also provide data, baselines and validation scripts.

Over time, we expect to add more tasks, rewards, observations and actions to bridge the compiler and ML research communities further.
