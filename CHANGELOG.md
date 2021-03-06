## Release 0.1.3 (2021-02-25)

This release adds numerous enhancements aimed at improving ease-of-use. Thanks
to @broune, @hughleat, and @JD-ETH for contributions.

* Added a new `env.validate()` API for validating the state of an environment.
  Added semantics validation for some LLVM benchmarks.
* Added a `env.fork()` method to efficiently duplicate an environment state.
* The `manual_env` environment has been improved with new features such as hill
  climbing search and tab completion.
* Ease of use improvements for string observation space and reward space names:
  Added new getter methods such as `env.observation.Autophase()` and generated
  constants such as `llvm.observation_spaces.autophase`.
* *Breaking change*: Calculation of environment reward has been moved to Python.
  Reward functions have been removed from backend service implementations and
  replaced with equivalent Python classes.
* Various bug fixes and improvements.

## Release 0.1.2 (2021-01-25)

* Add a new `compiler_gym.views.ObservationView.add_derived_space(...)` API
  for constructing derived observation spaces.
* Added default reward and observation values for `env.step()` in case of
  service failure.
* Extended the public `compiler_gym.datasets` API for managing datasets.
* [llvm] Adds `-Norm`-suffixed rewards that are normalized to unoptimized cost.
* Extended documentation and example codes.
* Numerous bug fixes and improvements.

## Release 0.1.1 (2020-12-28)

* Expose the package version through `compiler_gym.__version__`, and
  the compiler version through `CompilerEnv.compiler_version`.
* Add a [notebook
  version](https://colab.research.google.com/github/facebookresearch/CompilerGym/blob/development/examples/getting-started.ipynb)
  of the "Getting Started" guide that can be run in colab.
* [llvm] Reformulate reward signals to be cumulative.
* [llvm] Add a new reward signal based on the size of the `.text`
  section of compiled object files.
* [llvm] Add a `LlvmEnv.make_benchmark()` API for easily constructing
  custom benchmarks for use in environments.
* Numerous bug fixes and improvements.

## Release 0.1.0 (2020-12-21)

Initial release.
