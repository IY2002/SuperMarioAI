# Super Mario Bros PPO Agent 🎮🕹️

This repository showcases a Python implementation of a Proximal Policy Optimization (PPO) agent trained to play the classic Super Mario Bros game using the `gym-super-mario-bros` environment. The agent is built using the `stable-baselines3` library, which provides an advanced reinforcement learning method that combines the strengths of both trust region policy optimization (TRPO) and deep Q-learning (DQN). 🤖🧠

## Features 🌟

- Utilizes the PPO algorithm from the `stable-baselines3` library.
- Implements a custom callback to save model checkpoints during training.
- Uses a CNN policy to process grayscale images of the game.
- Stacks consecutive frames to provide better temporal information to the agent.

## Docker 🐳

This project includes a Dockerfile that allows for easy deployment and management of the agent's environment. The Dockerfile is based on the `python:3.8` image and installs all necessary dependencies, including `ffmpeg`, `libsm6`, `libxext6`, `stable-baselines3`, and `gym_super_mario_bros`.

To build and run the Docker container, use the following commands:

```bash
docker build -t super_mario_ppo_agent .
docker run -it --rm -v "$(pwd)/train:/app/train" -v "$(pwd)/logs:/app/logs" super_mario_ppo_agent
```
