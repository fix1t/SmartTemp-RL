# Results Branch Overview

This branch serves as a comprehensive repository for all outputs and data derived from the parameter tuning and evaluation sections of the reinforcement learning algorithms studied. This branch is meticulously organized to ensure ease of access and analysis of the data. Below is a detailed description of the contents and structure of this branch:

## Directory Structure

The branch is structured into several key directories and files, each serving a specific purpose in the storage and presentation of results:

- Details Folder: This directory contains detailed textual descriptions and summaries for each parameter combination tested. Each file within this folder provides insights into the setup, execution, and immediate outcomes of the tests.
- Score Plot: For each combination of parameters, a plot file is stored illustrating the progression of scores achieved by the agent during training. These plots are crucial for visualizing the learning effectiveness and stability over time.
- Behaviour Plot: Accompanying each score plot, the behaviour plot provides a visual representation of the agent's decision-making process in the environment. This may include actions taken at various states, changes in strategy, or responses to specific environmental cues, offering a deeper understanding of how parameter changes impact the agent's behaviour.
- Saved Model: Each set of parameters that undergoes evaluation has its corresponding model saved at the conclusion of training. These saved models allow for post-hoc analysis and further experimentation or demonstration without the need to retrain.
