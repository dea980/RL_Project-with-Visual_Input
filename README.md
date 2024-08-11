# Reinforcement Learning with Visual Inputs

This project explores the application of reinforcement learning (RL) to control an agent in an environment using both native and photographic inputs. The goal is to optimize the agent's performance through various experiments involving different neural network architectures and RL algorithms.

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)
- [Research Process](#research-process)
- [Future Work](#future-work)

## Abstract

This project demonstrates the potential of reinforcement learning in environments where agents must learn to control their surroundings without direct instructions. We focus on experiments using non-photographic and photographic datasets to train agents using Convolutional Neural Networks (CNNs) and Variational Autoencoders (VAEs). Our results highlight the challenges and successes of RL with simple visual inputs, emphasizing the importance of data representation in training RL agents.

## Introduction

Reinforcement learning (RL) is a method where an agent learns to interact with its environment through trial and error, aiming to maximize cumulative rewards. Recent successes in RL, such as defeating a world champion in Go, showcase its potential. This project explores RL using visual inputs, comparing native state representations with photographic data processed through CNNs and VAEs.

## Methods

1. **Experiment 1: Learning with Non-Photographic Dataset**
   - Utilized Proximal Policy Optimization (PPO) with a Multi-Layer Perceptron (MLP) for reinforcement learning.

2. **Experiment 2: Learning with Photographic Dataset using CNN (Before Encoding)**
   - Applied CNNs to process photographic inputs for RL.

3. **Experiment 3: Learning with Photographic Dataset using CNN (After Encoding)**
   - Employed VAEs to encode images into lower-dimensional representations before RL training with CNNs.

## Results

- **Experiment 1:** Achieved maximum performance quickly with native data using PPO and MLP.
- **Experiment 2:** Longer training times with visual inputs, but performance improved over time using CNNs.
- **Experiment 3:** Linear VAE did not improve performance significantly, highlighting challenges in encoding meaningful features.

## Conclusion

Our experiments reveal that RL is effective in native spaces but faces challenges with image-based inputs. While CNNs and VAEs can help, careful data representation and network design are crucial. Further exploration of different architectures and training strategies is needed to optimize performance.

## References

- [Convolutional Neural Networks (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Multi-Layer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- [Proximal Policy Optimization (PPO)](https://en.wikipedia.org/wiki/Proximal_Policy_Optimization)
- [Cartpole Environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## Research Process

1. **Reinforcement Learning without CNN:** 
   - Implemented using MLP and PPO with numerical state inputs.

2. **Reinforcement Learning with CNN Before VAE:**
   - Integrated CNNs for visual input processing.

3. **Reinforcement Learning with CNN After VAE:**
   - Utilized VAEs for dimensionality reduction before RL training.

## Future Work

- Increase training steps and explore alternative policy algorithms.
- Experiment with different hyperparameters and preprocessing techniques.
- Explore transfer learning and advanced generative models like VAEs.
