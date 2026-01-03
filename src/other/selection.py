import numpy as np
import plotly.graph_objects as go
from modules.activation import step_function
from parameters import g_q, g_x, multiple_dir_name, N, p, seed

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
np.random.seed(seed)

K = 80 # total number of active neurons to select
K_low = K // p # number of low-activity neurons
shared_fraction = 1 / 2  # max fraction of multi-pattern neurons
max_trials = 20000

output_txt = "selected_neurons.txt"

# --------------------------------------------------
# LOAD PATTERNS
# --------------------------------------------------
pattern_path = f"{multiple_dir_name}_{N}/npy/phi_memory_patterns.npy"
phi = np.load(pattern_path).transpose()
phi_step = step_function(phi, q_f=g_q, x_f=g_x)

N_total, P = phi_step.shape
assert P == 4

# activity[n, p] = 1 if neuron n is active in pattern p
activity = (phi_step > 0).astype(int)

# --------------------------------------------------
# DEFINE NEURON POOLS
# --------------------------------------------------
low_neurons = np.where(np.all(activity == 0, axis=1))[0]
active_neurons = np.where(np.any(activity == 1, axis=1))[0]

# --------------------------------------------------
# SEARCH FOR BALANCED ACTIVE SET
# --------------------------------------------------
found = False
max_shared = int(K * shared_fraction)

for _ in range(max_trials):

    candidate = np.random.choice(active_neurons, K, replace=False)

    # % active per pattern
    perc_active = activity[candidate].sum(axis=0) / K * 100
    if not np.all((20 <= perc_active) & (perc_active <= 30)):
        continue

    # multi-pattern constraint
    n_shared = np.sum(activity[candidate].sum(axis=1) >= 2)
    if n_shared > max_shared:
        continue

    selected_active = candidate
    found = True
    break

if not found:
    raise RuntimeError("No valid neuron set found under the given constraints")

# --------------------------------------------------
# ADD LOW NEURONS
# --------------------------------------------------
if len(low_neurons) < K_low:
    raise RuntimeError("Not enough low neurons available")

selected_low = np.random.choice(low_neurons, K_low, replace=False)

final_neurons = np.sort(np.concatenate([selected_active, selected_low]))

# --------------------------------------------------
# CLASSIFY FINAL NEURONS
# --------------------------------------------------
groups = {p: [] for p in range(P)}
inactive = []

for n in final_neurons:
    if np.all(activity[n] == 0):
        inactive.append(n)
    else:
        for p in range(P):
            if activity[n, p]:
                groups[p].append(n)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
print("\nNeuron groups:")
for p in range(P):
    print(f"P{p+1} ({len(groups[p])}): {groups[p]}")
print(f"Inactive ({len(inactive)}): {inactive}")

print("\nComma-separated list:")
print(",".join(map(str, final_neurons)))

# --------------------------------------------------
# SAVE TXT
# --------------------------------------------------
with open(output_txt, "w") as f:
    for n in final_neurons:
        f.write(f"{n}\n")

print(f"\nSaved neuron list to {output_txt}")

# --------------------------------------------------
# PLOTLY HISTOGRAM
# --------------------------------------------------
labels = [f"P{p+1}" for p in range(P)] + ["Inactive"]
counts = [len(groups[p]) for p in range(P)] + [len(inactive)]
hover = (
    ["<br>".join(map(str, groups[p])) for p in range(P)]
    + ["<br>".join(map(str, inactive))]
)

fig = go.Figure(
    go.Bar(
        x=labels,
        y=counts,
        hovertext=hover,
        hoverinfo="text+y"
    )
)

fig.update_layout(
    title=f"Neuron activity distribution (shared_fraction = {shared_fraction})",
    xaxis_title="Group",
    yaxis_title="Number of neurons"
)

fig.show()
