import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 설정
G = 10000  # Gravity of Center Mass
g = 1  # Gravity of Particles
num_particles = 10
time_step = 0.1
total_time = 100

# Initialize positions and velocities
center_mass = np.array([0.0, 0.0, 0.0])
particles_pos = np.random.uniform(-50, 50, (num_particles, 3))
particles_vel = np.random.uniform(-10, 10, (num_particles, 3))

print(particles_pos)
print(particles_vel)


def calculate_gravity(pos1, pos2, mass1, mass2):
    direction = pos2 - pos1
    distance = np.linalg.norm(direction)  # get the distance between two objects
    if distance < 1e-5:  # ignore if the distance is too small
        return np.zeros(3)
    force_magnitude = (mass1 * mass2) / (distance**2)  # calculate the force magnitude
    unit_vec = direction / distance
    force = force_magnitude * unit_vec  # calculate the force vector
    return force


# 시뮬레이션 업데이트 함수
def update(frame):
    global particles_pos, particles_vel

    for i in range(num_particles):
        # 중심 물체로부터의 중력 계산
        force = calculate_gravity(particles_pos[i], center_mass, g, G)
        acceleration = force / g
        particles_vel[i] += acceleration * time_step
        particles_pos[i] += particles_vel[i] * time_step

    ax.cla()
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.set_zlim(-120, 120)
    ax.scatter(0, 0, 0, color="red", s=100, label="Center Mass (G)")
    ax.scatter(
        particles_pos[:, 0],
        particles_pos[:, 1],
        particles_pos[:, 2],
        color="blue",
        s=20,
        label="Particles (g)",
    )
    ax.legend()
    ax.set_title(f"3D Gravity Simulation - Time: {frame * time_step:.1f}s")
    return ax.artists


# 3D 플롯 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ani = FuncAnimation(fig, update, frames=int(total_time / time_step), interval=50)
plt.show()
