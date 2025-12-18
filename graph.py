import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# 1. Load Accuracy Data
data_acc = """algorithm,num_clients,dataset,run,test_accuracy,val_accuracy,best_round,total_train_time,avg_time_per_round,test_time,num_rounds
fedala,5,Cora,0,82.42,84.22,28,11.12,0.11,0.55,100
fedala,5,Cora,1,81.60,84.04,50,10.77,0.10,0.53,100
fedala,5,Cora,2,82.32,84.50,27,9.90,0.09,0.49,100
newala,5,Cora,0,81.69,84.32,34,13.04,0.13,0.65,100
newala,5,Cora,1,82.05,84.50,27,13.32,0.13,0.66,100
newala,5,Cora,2,81.78,84.04,28,12.68,0.12,0.63,100
fedala,10,Cora,0,81.15,82.14,21,20.32,0.20,1.01,100
fedala,10,Cora,1,81.33,82.32,21,20.77,0.20,1.03,100
fedala,10,Cora,2,81.87,82.05,19,20.20,0.20,1.01,100
newala,10,Cora,0,81.33,82.32,15,25.74,0.25,1.28,100
newala,10,Cora,1,80.97,82.60,15,25.56,0.25,1.27,100
newala,10,Cora,2,80.79,82.23,16,25.56,0.25,1.27,100
fedala,15,Cora,0,77.72,80.64,73,30.85,0.30,1.54,100
fedala,15,Cora,1,77.98,80.37,65,30.42,0.30,1.52,100
fedala,15,Cora,2,77.63,80.47,74,30.01,0.30,1.50,100
newala,15,Cora,0,77.45,79.90,79,37.76,0.37,1.88,100
newala,15,Cora,1,77.27,79.81,86,37.17,0.37,1.85,100
newala,15,Cora,2,77.45,79.81,90,37.64,0.37,1.88,100
fedala,20,Cora,0,78.02,79.43,85,39.87,0.39,1.99,100
fedala,20,Cora,1,78.10,79.25,69,38.97,0.38,1.94,100
fedala,20,Cora,2,76.70,79.53,25,40.79,0.40,2.03,100
newala,20,Cora,0,77.84,79.07,32,49.90,0.49,2.49,100
newala,20,Cora,1,77.40,79.62,29,49.97,0.49,2.49,100
newala,20,Cora,2,77.39,79.25,72,50.08,0.50,2.50,100
fedala,25,Cora,0,74.79,80.64,39,49.52,0.49,2.47,100
fedala,25,Cora,1,74.97,81.00,37,49.79,0.49,2.48,100
fedala,25,Cora,2,75.49,80.92,36,50.29,0.50,2.51,100
newala,25,Cora,0,75.32,80.61,26,60.54,0.60,3.02,100
newala,25,Cora,1,75.05,80.44,50,61.16,0.61,3.05,100
newala,25,Cora,2,75.06,80.61,35,62.22,0.62,3.11,100
fedala,30,Cora,0,73.99,77.03,64,59.84,0.59,2.99,100
fedala,30,Cora,1,74.60,77.31,82,58.90,0.58,2.94,100
fedala,30,Cora,2,74.77,77.40,77,58.33,0.58,2.91,100
newala,30,Cora,0,75.21,77.59,40,72.36,0.72,3.61,100
newala,30,Cora,1,75.38,77.59,55,74.18,0.74,3.70,100
newala,30,Cora,2,75.29,77.50,37,72.89,0.72,3.64,100
"""

df_acc = pd.read_csv(io.StringIO(data_acc))
grouped_acc = df_acc.groupby(['algorithm', 'num_clients'])['test_accuracy'].agg(['mean', 'std']).reset_index()

# 2. Project Parameter Data Linearly
base_fedala = 32259
base_newala = 18316
clients_list = [5, 10, 15, 20, 25, 30]

data_params = []
for c in clients_list:
    data_params.append({
        'num_clients': c,
        'fedala_params': base_fedala * c,
        'newala_params': base_newala * c
    })
df_params = pd.DataFrame(data_params)

# Reshape
params_melted = []
for idx, row in df_params.iterrows():
    params_melted.append({'algorithm': 'fedala', 'num_clients': row['num_clients'], 'params': row['fedala_params']})
    params_melted.append({'algorithm': 'newala', 'num_clients': row['num_clients'], 'params': row['newala_params']})
df_params_melted = pd.DataFrame(params_melted)

# Merge
merged = pd.merge(grouped_acc, df_params_melted, on=['algorithm', 'num_clients'])

# 3. Plotting
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.35
clients = sorted(merged['num_clients'].unique())
x = np.arange(len(clients))

# Scale Factor: Map 1 Million params to -100 on Y-axis
param_scale_factor = 100 / 1000000

# Define colors
color_fedala = 'tab:blue'
color_newala = 'tab:red'

for i, algo in enumerate(['fedala', 'newala']):
    subset = merged[merged['algorithm'] == algo].sort_values('num_clients')
    offset = bar_width / 2 if i == 1 else -bar_width / 2

    # Determine Color and Label Name
    if algo == 'fedala':
        current_color = color_fedala
        display_name = 'FedALA'
    else:
        current_color = color_newala
        display_name = r'LoRA-CA$^3$'

        # Upper Bars (Accuracy)
    bars_upper = ax.bar(x + offset, subset['mean'], bar_width,
                        label=display_name,
                        color=current_color,
                        edgecolor='black', alpha=0.8)

    # Labels for Upper Bars
    for bar in bars_upper:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    # Lower Bars (Parameters) - Negative direction
    scaled_params = -1 * subset['params'] * param_scale_factor
    bars_lower = ax.bar(x + offset, scaled_params, bar_width,
                        color=current_color,
                        edgecolor='black', alpha=0.5, hatch='//')

    # Labels for Lower Bars
    for bar, val in zip(bars_lower, subset['params']):
        height = bar.get_height()
        if val >= 1000000:
            label_text = f'{val / 1000000:.1f}M'
        else:
            label_text = f'{val / 1000:.0f}k'
        ax.text(bar.get_x() + bar.get_width() / 2., height - 8,
                label_text, ha='center', va='top', fontsize=8, fontweight='bold', rotation=90)

# Axes Setup
ax.set_xticks(x)
ax.set_xticklabels(clients)
ax.set_xlabel('Number of Clients', fontsize=12)

# Y-Axis Setup
ax.set_ylim(-130, 110)
ax.axhline(0, color='black', linewidth=1)

# Custom Ticks
yticks_pos = [0, 20, 40, 60, 80, 100]
yticklabels_pos = [str(y) for y in yticks_pos]
param_ticks = [200000, 400000, 600000, 800000, 1000000]
yticks_neg = [-1 * p * param_scale_factor for p in param_ticks]
yticklabels_neg = ['200k', '400k', '600k', '800k', '1M']

ax.set_yticks(yticks_neg + yticks_pos)
ax.set_yticklabels(yticklabels_neg + yticklabels_pos)

# --- CORRECTED LABELS SECTION ---
# Used transform=ax.transAxes to place labels relative to the axes box, not data.
# -0.08 puts it safely to the left of the axis spine.
# 0.75 is roughly the vertical center of the upper (accuracy) section.
# 0.25 is roughly the vertical center of the lower (parameters) section.
ax.text(-0.08, 0.75, 'Test Accuracy (%)', rotation=90, va='center', ha='center', fontsize=12, transform=ax.transAxes)
ax.text(-0.08, 0.25, 'Total Parameters', rotation=90, va='center', ha='center', fontsize=12, transform=ax.transAxes)

# Legend
custom_lines = [
    Line2D([0], [0], color=color_fedala, lw=4),
    Line2D([0], [0], color=color_newala, lw=4)
]
ax.legend(custom_lines, ['FedALA', r'LoRA-CA$^3$'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

plt.title('Performance vs. Complexity: Accuracy & Parameter Count', fontsize=14, y=1.05)

# Ensure layout accounts for the left labels
plt.tight_layout()
plt.subplots_adjust(left=0.12)  # Manually reserve 12% space on the left

# Save
plt.savefig('diverging_plot_extended_fixed.pdf', format='pdf', dpi=300)
print("Saved to diverging_plot_extended_fixed.pdf")