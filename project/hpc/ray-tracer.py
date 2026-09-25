import numpy as np
import matplotlib.pyplot as plt

# Geometry (mm)
width = 3.0
length = 15.0
x_min, x_max = -width/2, width/2
z_min, z_max = -length/2, length/2

# Interaction point
x0 = x_max   # try different values
z0 = 5.0

# Optical attenuation length (mm)
atten_length = 300.0

# Ray settings
n_rays = 500000
n_plot = 20  # number of rays to visualize

# Generate isotropic directions toward +z
theta = np.random.uniform(-np.pi/2, np.pi/2, n_rays)
dx = np.sin(theta)
dz = np.cos(theta)

# Storage
x_hits = []
weights = []

# For plotting ray paths
ray_paths = []

for i in range(n_rays):
    x, z = x0, z0
    vx, vz = dx[i], dz[i]

    path = [(x, z)]
    total_length = 0

    while True:
        # Distance to top (SiPM)
        if vz > 0:
            t_z = (z_max - z) / vz
        else:
            break

        # Distance to side walls
        if vx > 0:
            t_x = (x_max - x) / vx
        elif vx < 0:
            t_x = (x_min - x) / vx
        else:
            t_x = np.inf

        # Next boundary
        t = min(t_z, t_x)

        # Advance
        x_new = x + vx * t
        z_new = z + vz * t
        total_length += t

        path.append((x_new, z_new))

        # Check if hit SiPM plane
        if t == t_z:
            x_hits.append(x_new)
            weights.append(np.exp(-total_length / atten_length))
            if len(ray_paths) < n_plot:
                ray_paths.append(path)
            break

        # Otherwise reflect on side wall
        vx = -vx
        x, z = x_new, z_new

# ---- Plot geometry + rays ----
plt.figure(figsize=(5, 8))

# Draw crystal rectangle
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [z_min, z_min, z_max, z_max, z_min],
         'k-')

# Plot rays
for path in ray_paths:
    xs, zs = zip(*path)
    plt.plot(xs, zs, alpha=0.3)

# Mark interaction point
plt.scatter([x0], [z0], color='red', label='Interaction')

plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.title("2D Ray tracing in 3×15 mm crystal")
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

# ---- Plot intensity at SiPM ----
bins = np.linspace(x_min, x_max, 60)
hist, _ = np.histogram(x_hits, bins=bins, weights=weights)
centers = 0.5*(bins[:-1] + bins[1:])

plt.figure()
#plt.plot(centers, hist / hist.sum())

plt.bar(
    bins[:-1],
    hist / hist.sum(),
    width=np.diff(bins),
    align="edge",
    alpha=0.8,
    edgecolor="black",
)




plt.xlabel("x at SiPM (mm)")
plt.ylabel("Normalized intensity")
plt.title("Light distribution at SiPM")
plt.show()