## Reinforcement Learning Experiments

This file contains some of my early experiments with TensorFlow and Reinforcement Learning (which means that the code structuring might be questionable sometimes etc.)

All test are run for 300 steps. No real reason for this number. My observation is that if the agent can balance the pole for 300 steps, it can often do it for longer too.

The code was originally written in TensorFlow 0.12 so a few things have changed (mostly regarding saving and loading models). The algorithmic part should still work though.

If you want to load and run the models with a recent TensorFlow version (>v1.5), I recommend you inspect the checkpoint and load the weights manually. This [Gist](https://gist.github.com/noorvir-a/6473f661cdfe1cb995eaf22b3f72f783) gives an example of how to do this. Alternatively you can rename the variables in the build_graph() methods of each notebook to match those in the checkpoint.
