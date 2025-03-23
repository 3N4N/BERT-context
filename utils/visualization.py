import matplotlib.pyplot as plt
import numpy as np

def get_vector_points(v1,v2, normalize=False):
    # Normalize the data if required (along maximum)
    if normalize:
        max_value = max(max(v1), max(v2))
        v1 = [x / max_value for x in v1]
        v2 = [x / max_value for x in v2]

    # We must assume both vectors are the same length
    N = len(v1)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Calculate the points in relation to the circle
    radii_1 = 1 + np.array(v1)
    x1 = radii_1 * np.cos(angles)
    y1 = radii_1 * np.sin(angles)

    radii_2 = 1 + np.array(v2)
    x2 = radii_2 * np.cos(angles)
    y2 = radii_2 * np.sin(angles)

    return x1, y1, x2, y2

def plot_vectors(ax, v1, v2, col, show_labels=False, normalize=False):

    x1,y1, x2,y2 = get_vector_points(v1,v2, normalize)

    # Plot the circle
    circle = plt.Circle((0, 0), 1, edgecolor='gray', facecolor='none')
    ax.add_artist(circle)

    # Plot the points
    ax.scatter(x1, y1, marker='o', color=col[0])
    ax.scatter(x2, y2, marker='o', color=col[1])

    # Draw lines between v1 and v2
    for px, py, tx, ty in zip(x1, y1, x2, y2):
        ax.plot([px, tx], [py, ty], col[2], linestyle='-')

    # Optionally write labels
    if show_labels:
        for i, (xi, yi) in enumerate(zip(x1, y1)):
            ax.text(xi, yi, f'{v1[i]}', fontsize=8, ha='right' if xi < 0 else 'left', va='bottom' if yi < 0 else 'top')
        for i, (xi, yi) in enumerate(zip(x2, y2)):
            ax.text(xi, yi, f'{v2[i]}', fontsize=8, ha='right' if xi < 0 else 'left', va='bottom' if yi < 0 else 'top')

    # Set limits and aspect
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
