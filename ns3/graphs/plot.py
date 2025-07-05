import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Load data into DataFrame
df = pd.read_csv('fbs_graph.csv')
groups = df.groupby(['gnb_power', 'fbs_power'])

def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i]+1, f"{round(y[i],2)}%", ha='center', fontsize=15, color='black')
        plt.text(i, y[i] // 2, f"{int(round(y[i] * 12/100,0))}/12", ha='center', fontsize=15, color='black')

# Accuracy per Group
accuracies = []
group_labels = []
for (gnb_power, fbs_power), group in groups:
    accuracy = (group['actual_fbs'] == group['detected_fbs']).mean() * 100
    accuracies.append(accuracy)
    group_labels.append(f"{gnb_power} dBm\ngNB\n{fbs_power} dBm\nFBS\n{fbs_power/gnb_power*100:.1f}% fbs power")

plt.figure(figsize=(12, 6))
bars = plt.bar(group_labels, accuracies, color='skyblue')
bars[0].set_color('skyblue')
bars[1].set_color('#60a3d9')
bars[2].set_color('#0074b7')

bars[3].set_color('skyblue')
bars[4].set_color('#60a3d9')
bars[5].set_color('#0074b7')
add_labels([0,10,20,30,40,50], accuracies)
plt.ylabel('Accuracy (%)')
plt.title('Detection Accuracy per Power Configuration')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('accuracy_per_group.png', dpi=300)
plt.close()

# Actual vs. Detected Indices
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
for i, ((gnb_power, fbs_power), group) in enumerate(groups):
    group = group.sort_values('actual_fbs')
    plt.plot(group['actual_fbs'], group['detected_fbs'], 
             'o-', color=colors[i], 
             label=f"gNB: {gnb_power:.1f} dBm, FBS: {fbs_power:.1f} dBm")
plt.plot([0, 11], [0, 11], 'k--', alpha=0.3, label='Ideal Detection')
plt.xlabel('Actual FBS Cell ID')
plt.ylabel('Detected FBS Cell ID')
plt.xticks(range(0, 12))
plt.yticks(range(0, 12))
plt.title('Actual vs. Detected FBS Cell ID')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('actual_vs_detected.png', dpi=300)
plt.close()

# Absolute Error per Index
plt.figure(figsize=(10, 6))
for i, ((gnb_power, fbs_power), group) in enumerate(groups):
    group = group.sort_values('actual_fbs')
    error = np.abs(group['actual_fbs'] - group['detected_fbs'])
    plt.plot(group['actual_fbs'], error, 's-', 
             label=f"gNB: {gnb_power} dBm, FBS: {fbs_power} dBm")
plt.xlabel('Actual FBS Index')
plt.ylabel('|Detected - Actual|')
plt.title('Absolute Detection Error per FBS Index')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('absolute_error.png', dpi=300)
plt.close()

# Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, ((gnb_power, fbs_power), group) in enumerate(groups):
    conf_mat = pd.crosstab(
        group['actual_fbs'], 
        group['detected_fbs'],
        rownames=['Actual'],
        colnames=['Detected']
    ).reindex(index=range(0,12), columns=range(0,12), fill_value=0)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'gNB: {gnb_power} dBm\nFBS: {fbs_power} dBm')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.close()
