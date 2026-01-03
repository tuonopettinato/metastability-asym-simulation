import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from modules.activation import step_function
from parameters import g_q, g_x, multiple_dir_name, N, seed

# --------------------------------------------------
# FIXED SEED
# --------------------------------------------------
np.random.seed(seed)


# --------------------------------------------------
# OUTPUT SETTINGS
# --------------------------------------------------

output_dir = f"{multiple_dir_name}_{N}"
output_txt = "neuron_pattern_activity.txt"
subset_txt = "subset_neurons.txt"
grid_img_all = "activity_grid_all.png"
grid_img_subset = "activity_grid_subset.png"

K = 200 # total number of neurons in subset to analyze

# --------------------------------------------------
# LOAD PATTERNS
# --------------------------------------------------
pattern_path = f"{multiple_dir_name}_{N}/npy/phi_memory_patterns.npy"
phi = np.load(pattern_path).transpose()
phi_step = step_function(phi, q_f=g_q, x_f=g_x)

N_total, P = phi_step.shape
assert P == 4

# --------------------------------------------------
# BUILD ACTIVITY GRID
# --------------------------------------------------
activity = (phi_step > 0).astype(int)

# --------------------------------------------------
# SAVE ACTIVITY GRID TO TXT
# --------------------------------------------------
with open(output_txt, "w") as f:
    f.write("neuron_id p0 p1 p2 p3\n")
    for n in range(N_total):
        f.write(f"{n} {activity[n,0]} {activity[n,1]} {activity[n,2]} {activity[n,3]}\n")

print(f"Saved activity grid to {output_txt}")

# --------------------------------------------------
# STATISTICS FUNCTION
# --------------------------------------------------
def compute_stats(activity_matrix):
    sums = activity_matrix.sum(axis=1)
    return {
        "P1_only": np.sum((sums == 1) & (activity_matrix[:,0] == 1)),
        "P2_only": np.sum((sums == 1) & (activity_matrix[:,1] == 1)),
        "P3_only": np.sum((sums == 1) & (activity_matrix[:,2] == 1)),
        "P4_only": np.sum((sums == 1) & (activity_matrix[:,3] == 1)),
        "2_patterns": np.sum(sums == 2),
        "3_patterns": np.sum(sums == 3),
        "4_patterns": np.sum(sums == 4),
        "inactive": np.sum(sums == 0)
    }

# --------------------------------------------------
# GLOBAL STATISTICS
# --------------------------------------------------
global_stats = compute_stats(activity)

print("\nGLOBAL ACTIVITY STATISTICS")
for k, v in global_stats.items():
    print(f"{k}: {v}")

# --------------------------------------------------
# GRID IMAGE (ALL NEURONS)
# --------------------------------------------------
plt.figure(figsize=(6, 8))
plt.imshow(activity, aspect="auto", cmap="gray_r")
plt.xlabel("Pattern")
plt.ylabel("Neuron")
plt.title("Neuron × Pattern Activity Grid (All)")
plt.colorbar(label="Active")
plt.tight_layout()
plt.savefig(grid_img_all)
plt.close()

print(f"Saved grid image: {grid_img_all}")

# --------------------------------------------------
# BUILD SUBSET OF K NEURONS
# --------------------------------------------------
subset = np.random.choice(np.arange(N_total), K, replace=False)
subset = np.sort(subset)  # readability
subset_activity = activity[subset]

# --------------------------------------------------
# PRINT SUBSET (COMMA SEPARATED)
# --------------------------------------------------
print(f"\nSUBSET OF {K} NEURONS (comma-separated):")
print(",".join(map(str, subset)))

# --------------------------------------------------
# SAVE SUBSET TO TXT
# --------------------------------------------------
with open(subset_txt, "w") as f:
    for n in subset:
        f.write(f"{n}\n")

print(f"Saved subset neuron list to {subset_txt}")

# --------------------------------------------------
# SUBSET STATISTICS
# --------------------------------------------------
subset_stats = compute_stats(subset_activity)

print(f"\nSUBSET ({K}) ACTIVITY STATISTICS")
for k, v in subset_stats.items():
    print(f"{k}: {v}")

# --------------------------------------------------
# GRID IMAGE (SUBSET)
# --------------------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(subset_activity, aspect="auto", cmap="gray_r")
plt.xlabel("Pattern")
plt.ylabel("Subset neuron index")
plt.title(f"Neuron × Pattern Activity Grid (Subset of size {K})")
plt.colorbar(label="Active")
plt.tight_layout()
plt.savefig(grid_img_subset)
plt.close()

print(f"Saved subset grid image: {grid_img_subset}")

# --------------------------------------------------
# TRIANGULAR HISTOGRAM (PERCENTAGES)
# --------------------------------------------------
labels = [
    "P1 only", "P2 only", "P3 only", "P4 only",
    "2 patterns", "3 patterns", "4 patterns",
    "Inactive"
]

global_counts = [
    global_stats["P1_only"],
    global_stats["P2_only"],
    global_stats["P3_only"],
    global_stats["P4_only"],
    global_stats["2_patterns"],
    global_stats["3_patterns"],
    global_stats["4_patterns"],
    global_stats["inactive"]
]

subset_counts = [
    subset_stats["P1_only"],
    subset_stats["P2_only"],
    subset_stats["P3_only"],
    subset_stats["P4_only"],
    subset_stats["2_patterns"],
    subset_stats["3_patterns"],
    subset_stats["4_patterns"],
    subset_stats["inactive"]
]

global_perc = np.array(global_counts) / N_total * 100
subset_perc = np.array(subset_counts) / K * 100

fig = go.Figure()

fig.add_bar(x=labels, y=global_perc, name="All neurons (%)")
fig.add_bar(x=labels, y=subset_perc, name=f"Subset ({K}) (%)")

fig.update_layout(
    title=f"Activity Class Distribution: All Neurons vs Subset ({K})",
    xaxis_title="Class",
    yaxis_title="Percentage (%)",
    barmode="group",
    height=650,
    width=1100
)

fig.show()

# --------------------------------------------------
# FINAL PRINT
# --------------------------------------------------
print(f"\nFINAL COMPARISON (percentages) for subset size {K}")
for lbl, g, s in zip(labels, global_perc, subset_perc):
    print(f"{lbl:12s} | global: {g:6.2f}% | subset: {s:6.2f}%")
