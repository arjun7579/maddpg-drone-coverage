# 🛩️ MADDPG-Based UAV Swarm Coverage

A research-oriented PyTorch implementation of a **multi-agent UAV swarm** system trained with **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**. This project demonstrates how advanced multi-agent reinforcement learning enables UAVs to:

- 🚫 Avoid collisions with dynamic and static obstacles as well as other drones
- 📡 Maximize user coverage with smooth, continuous control
- 🤖 Coordinate efficiently to cover more area and minimize redundant coverage

---

## What’s in This Project?

- **MADDPG Algorithm:**  
  Each drone is an independent agent with its own actor network and a centralized critic, allowing for cooperative learning in a continuous action space.

- **Custom UAV Environment:**  
  Simulates a 2D grid world featuring:
  - Multiple drones (agents)
  - Static and dynamic obstacles (with movement and state in observations)
  - Randomly placed users to be covered
  - Collision logic and area coverage constraints

- **Replay Buffer:**  
  Stores experiences for efficient, off-policy multi-agent training.

- **Reward Structure:**  
  Designed to balance:
  - Collision avoidance (with obstacles and other drones)
  - Maximizing unique user coverage
  - Penalizing redundant or idle behavior

- **Training Loop:**  
  Runs for thousands of episodes, printing per-episode rewards for each drone and supporting model checkpointing.

- **(Optional) Visualization:**  
  Real-time 2D plotting of drones, users, obstacles, and coverage area for qualitative evaluation after training.

---

## Reward Function (Typical Example)

| Event                                    | Reward / Penalty |
|-------------------------------------------|------------------|
| Collision with obstacle                   | −5               |
| Collision with another drone              | −5               |
| New user covered (not seen before)        | +5               |
| Existing user still covered               | +0.5             |
| Idle (no movement or redundant coverage)  | −0.5             |

---

## Hyperparameters

- Grid size: 20×20
- Drones: 5  
- Users: 10  
- Obstacles: 5  
- Coverage radius: 3  
- Actor/Critic hidden layers: 256/384  
- Exploration noise: 0.1  
- Training episodes: 1000

---

## Project Goals

- **Multi-agent collision avoidance**  
- **Maximize user coverage**  
- **Enable smooth, continuous control**  
- **Demonstrate MADDPG’s effectiveness for swarm coordination**

---

## References

- [Lowe et al., 2017. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.](https://arxiv.org/abs/1706.02275)
- [ICAS 2024: UAV Swarm Area Coverage Reconnaissance](https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_0550_paper.pdf)
- [RAND: Unmanned Aerial Systems Intelligent Swarm Technology](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2300/RRA2380-1/RAND_RRA2380-1.pdf)
- [Exploring the Power of Heterogeneous UAV Swarms Through Reinforcement Learning](https://pdfs.semanticscholar.org/acc1/d8d1c18f119bc4ade25cd1ec5fbc8ece03e6.pdf)

---


