# Reinforcement Learning Project: Cartpole Environment

## Abstract

This project focuses on implementing and tuning the REINFORCE algorithm and the Actor-Critic method to continuous action spaces in the Cartpole problem.

## Table of Contents

- [Introduction](#introduction)
- [The Cartpole Environment](#the-cartpole-environment)
- [Policy Neural Network](#policy-neural-network)
- [REINFORCE](#reinforce)
- [Actor-Critic for Continuous Setting](#actor-critic-for-continuous-setting)
- [Results](#results)

## Introduction

This project explores reinforcement learning algorithms applied to the Cartpole environment, focusing on REINFORCE and Actor-Critic methods for discrete and continuous action spaces.

## The Cartpole Environment

The Cartpole problem involves balancing a pole on a moving cart. The state consists of the cart's position, velocity, pole angle, and angular velocity. The agent learns to apply left or right forces to balance the pole, receiving rewards for successful balancing and penalties for failure.

## Policy Neural Network

The neural network for the actor model is defined to output actions based on the current state, initially using a random policy which is later trained to optimize performance.

## REINFORCE

The REINFORCE algorithm uses a policy gradient method to optimize the agent's actions. Key hyperparameters include the learning rate and discount factor, with optimal values found to ensure stable convergence.

## Actor-Critic for Continuous Setting

The Actor-Critic method combines policy-based and value-based approaches. The actor selects actions based on the policy, while the critic evaluates actions using a value function, updating the policy parameters based on the critic's feedback.

## Results

The Actor-Critic algorithm demonstrated consistent convergence and effective performance in continuous action spaces, outperforming the REINFORCE algorithm in these settings.

---

For more details, refer to the [full project report](ProjectReport.pdf).
