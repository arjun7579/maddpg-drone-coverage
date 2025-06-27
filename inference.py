import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os

from main import MADDPGAgent, UAVEnv, OBS_DIM, STATE_DIM, ACTION_DIM, DRONE_COUNT

GRID_SIZE = 20
DRONE_COUNT = 5
USER_COUNT = 10
OBSTACLE_COUNT = 5
COVERAGE_RADIUS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained agents (from episode 500) ---
agents = [MADDPGAgent(OBS_DIM, STATE_DIM, ACTION_DIM, DRONE_COUNT) for _ in range(DRONE_COUNT)]
for i, agent in enumerate(agents):
    model_path = f"saved_models/actor_agent_{i}_ep500.pth"
    if os.path.exists(model_path):
        agent.actor.load_state_dict(torch.load(model_path, map_location=device))
        agent.actor.eval()
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

# --- Visualization function ---
def render_env(ax, drones, users, static_obs, dynamic_obs, coverage_radius, grid_size, coverage):
    ax.clear()
    if users:
        ux, uy = zip(*users)
        ax.scatter(ux, uy, c='gold', s=30, label='Users')
    if static_obs:
        ox, oy = zip(*static_obs)
        ax.scatter(ox, oy, c='red', s=30, marker='x', label='Static Obstacles')
    if dynamic_obs:
        dx, dy = zip(*dynamic_obs)
        ax.scatter(dx, dy, c='magenta', s=30, marker='D', label='Dynamic Obstacles')
    if drones:
        px, py = zip(*drones)
        ax.scatter(px, py, c='blue', s=80, label='Drones')
        for x, y in drones:
            circle = plt.Circle((x, y), coverage_radius, fill=False, linestyle='--', alpha=0.5, color='green')
            ax.add_artist(circle)
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_title(f"Users Covered: {coverage}/{USER_COUNT}")
    ax.legend(loc='lower right')
    ax.text(0.02, 0.95, f"Users Covered: {coverage}/{USER_COUNT}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.pause(0.001)

def users_covered(drones, users, coverage_radius):
    covered = set()
    for dx, dy in drones:
        for idx, (ux, uy) in enumerate(users):
            if np.hypot(dx-ux, dy-uy) <= coverage_radius:
                covered.add(idx)
    return len(covered)

# --- Run visualization ---
def run_visualization():
    env = UAVEnv()
    obs_n, state = env.reset()
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()
    try:
        while plt.fignum_exists(fig.number):
            actions = [agent.select_action(obs, noise=0.0) for agent, obs in zip(agents, obs_n)]
            next_obs_n, next_state, rewards, done, _ = env.step(actions)
            obs_n, state = next_obs_n, next_state

            drones = [tuple(d) for d in env.drones]
            users = [tuple(u) for u in env.users]
            static_obs = [tuple(o['pos']) for o in env.obstacles if not o['dynamic']]
            dynamic_obs = [tuple(o['pos']) for o in env.obstacles if o['dynamic']]
            coverage = users_covered(drones, users, COVERAGE_RADIUS)

            render_env(ax, drones, users, static_obs, dynamic_obs, COVERAGE_RADIUS, GRID_SIZE, coverage)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    run_visualization()
