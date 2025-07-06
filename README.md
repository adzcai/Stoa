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
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000" alt="Code Style"/>
  </a>
  <a  href="http://mypy-lang.org/">
    <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="MyPy" />
</a>
</div>

<h2 align="center">
  <p>A JAX-Native Interface for Reinforcement Learning Environments</p>
</h2>

## 🚀 Welcome to **Stoa**

Stoa provides a lightweight, JAX-native interface for reinforcement learning environments. It defines a common abstraction layer that enables different environment libraries to work together in JAX workflows.

> ⚠️ **Early Development** – Core abstractions are in place, but the ecosystem is still growing!

---

## 🎯 What Stoa Provides

* **Common Interface**
  A standardized `Environment` base class that defines the contract for RL environments in JAX.

* **JAX-Native Design**
  Pure-functional `step` and `reset` operations that work with JAX transformations like `jit` and `vmap`.

* **Environment Wrappers**
  Infrastructure for composing and extending environments with additional functionality.

* **Space Definitions**
  Structured representations of observation, action, and state spaces that work with JAX arrays.

* **TimeStep Protocol**
  Standardized way to represent environment transitions with proper termination/truncation signals.

---

## 🔧 Current Status

**Core Components:**
- ✅ `Environment` abstract base class
- ✅ `TimeStep` and space definitions
- ✅ Basic wrapper infrastructure
- ✅ Gymnax environment adapter

**In Development:**
- 🚧 Additional environment adapters (Brax, Jumanji, Navix, Mujoco Playground, etc.)
- 🚧 More utility wrappers
- 🚧 Documentation and examples

---

## ⚡ Quick Start

1. **Install Stoa**
   ```bash
   pip install stoa
   ```

2. **Use with Gymnax environments**
   ```python
   import jax
   import gymnax
   from stoa.env_wrappers.gymnax import GymnaxWrapper

   # Wrap a Gymnax environment
   env = GymnaxWrapper(gymnax.environments.CartPole())

   # Reset and step
   rng = jax.random.PRNGKey(0)
   state, timestep = env.reset(rng)

   action = env.action_space().sample(rng)
   next_state, next_timestep = env.step(state, action)
   ```

3. **Compose with JAX transformations**
   ```python
   @jax.jit
   def run_episode(env, rng):
       state, timestep = env.reset(rng)
       # ... episode logic
       return total_reward
   ```

---

## 🛣️ Roadmap

* ✅ Core environment interface
* 🚧 Additional environment adapters (Brax, Jumanji, etc.)
* 🚧 Utility wrappers (observation normalization, frame stacking, etc.)
* 🚧 Documentation and tutorials
* 🚧 Integration examples with RL libraries

---

## 🤝 Contributing

We're building Stoa to provide a common foundation for JAX-based RL research. Contributions are welcome!

---

### 📚 Related Projects

* **Stoix** – distributed single-agent RL in JAX
* **Gymnax** – Classic control environments in JAX
* **Brax** – Physics-based environments in JAX
* **Jumanji** – Board games and optimization problems in JAX
