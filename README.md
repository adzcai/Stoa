<p align="center">
  <a href="docs/images/stoa.png">
    <img src="docs/images/stoa.jpeg" alt="Stoa logo" width="30%"/>
  </a>
</p>

<div align="center">
  <a href="https://www.python.org/doc/versions/">
    <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python Versions"/>
  </a>
  <a href="https://github.com/your-org/stoa/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-orange" alt="License"/>
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000" alt="Code Style"/>
  </a>
  <a  href="http://mypy-lang.org/">
    <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="MyPy" />
</a>
</div>

<h2 align="center">
  <p>A JAX-native, Functional Interface for Reinforcement Learning Environments</p>
</h2>

## 🚀 Welcome to **Stoa**

Stoa is a lightweight, flexible, and fully JAX‑native API for RL environments—heavily inspired by `dm_env`. It offers a pure‑functional interface that aligns seamlessly with `jit`, `vmap`, and `grad`, enabling efficient experimentation and environment integration in JAX.

> ⚠️ **Work in progress** – core abstractions and utilities are shaping up fast!

---

## 🎯 What Makes Stoa Useful

* **Fully Functional API**
  Stateless `step` and `reset` operations designed for composability and JAX transformations.

* **First-class JAX Compatibility**
  Use `jit`, `vmap`, and even `grad` over environment interactions and dynamics.

* **Auto-resets & Truncation Signals**
  Native episode restart logic and cleaner support for early termination.

* **Structured States & Spaces**
  Designed to work with nested/structured states and custom observation/reward spaces.

* **Procedural‑Ready**
  Built with procedurally generated environments in mind.

* **Differentiable Environment Ops**
  Core functions are differentiable to support model-based methods and environment gradients.

* **Wrapper Ecosystem**
  Support for standard wrappers—reduction of boilerplate and easy composition.


## ⚡ Quickstart

1. **Install Stoa**
   *(installation instructions coming soon)*

2. **Sketch an environment loop**

   ```python
   import jax
   from stoa import Env, reset, step

   @jax.jit
   def run_episode(params, rng):
       env_state = reset(Env(), rng)
       def body(carry, _):
           state, rng = carry
           action = policy(params, state.obs)
           next_state = step(Env(), state, action)
           return (next_state, rng), next_state.reward
       (_, _), rewards = jax.lax.scan(body, (env_state, rng), None, length=1000)
       return rewards.sum()
   ```

3. **Layer on wrappers**
   Add observation normalization, frame stacking, etc., via native wrappers.

4. **Optimize transforms**
   Wrap loops with `jit`, batch across environments with `vmap`, or distribute via `pmap`.

---

## 🛣️ Roadmap & Growth

* ✅ Define and solidify core abstractions (`Env`, `step`, `reset`, spaces)
* 🚧 Build and test essential wrappers for utility and popular environment suites.
* 🏞️ Provide example/procedural environments
* 🔄 Integrate with Stoix.
* 🏁 Iterate API based on community feedback

---

## 🤝 Contributions Welcome

We’re building Stoa to evolve with the needs of modern RL research. Issues, PRs, ideas—**all welcome**!

---

### 📚 Related Projects

* **Stoix** – distributed single-agent RL in JAX
* **Jumanji**, **Gymnax**, **Brax** – JAX-native environment suites
