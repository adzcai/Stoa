# JaxEnv

JaxEnv provides a flexible and powerful JAX-native API for reinforcement learning environments, inspired by `dm_env`. It is designed from the ground up to leverage JAX's functional and differentiable programming paradigms, offering several utilities for modern RL research and development. While still under active development, the core API and several key features are taking shape.

## Overview

The primary goal of JaxEnv is to serve as a robust interface layer for integrating various environment implementations within the JAX ecosystem. It aims to provide a familiar yet improved experience compared to existing APIs, with first-class support for features essential for modern RL experimentation.

## Key Features (In Progress)

*   **JAX Native:** Fully compatible with JAX transformations like `jit`, `vmap`, and `grad`.
*   **Stateless `step` Function:** Purely functional step transitions suitable for JAX's programming model.
*   **Automatic Episode Resets:** Designed to handle episode termination and resetting natively.
*   **Enhanced Truncation Support:** More natural handling of episode truncation signals compared to traditional APIs.
*   **Structured State Spaces:** Exploring ways to define and utilize structured environment states, potentially integrated with resets.
*   **Procedural Generation Ready:** Built-in considerations for procedurally generated environments (inspired by [jaxued](https://github.com/DramaCow/jaxued/blob/main/src/jaxued/environments/underspecified_env.py)).
*   **Flexible Spaces:** Adaptable observation and reward space definitions.
*   **Differentiable Operations:** Core `step` and `reset` operations are designed with differentiability in mind.
*   **Wrapper Ecosystem:** Built for wrapper compatibility and aims to include a suite of commonly used wrappers, reducing boilerplate.
*   **Tools:** Planning to include useful tools alongside the core API.

## Getting Started

*(Placeholder)* Installation instructions and basic usage examples will be added as the library matures.

## Wrappers

JaxEnv will include several useful wrappers commonly needed in RL workflows, saving users from rewriting them. Details on available wrappers will be documented here.

## Roadmap / Future Work

*   Solidify the core Environment API.
*   Implement and test essential wrappers.
*   Develop example environments demonstrating the API usage.
*   Integrate with libraries like [Stoix](https://github.com/InstaDeepAI/Stoix) (potentially replacing Jumanji usage).
*   Refine API design based on usage and feedback.
*   Expand tooling and utilities.

Contributions are welcome!