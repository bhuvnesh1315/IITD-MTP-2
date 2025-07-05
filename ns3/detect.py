import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os

sns.set_style("whitegrid")
sns.set_palette("colorblind")

def read_topology(file_path):
    df = pd.read_csv(file_path, header=None, sep=' ', names=['x', 'y'])
    gnb_locs = df.iloc[:12].values
    ue_locs = df.iloc[12:].values
    return gnb_locs, ue_locs

def read_rsrp(file_path):
    df = pd.read_csv(file_path, header=None, sep=',')
    
    num_gnbs = 12  
    rsrp = df.iloc[:, :num_gnbs].values
    
    return rsrp

def create_features(ue_loc, gnb_loc):
    dx = ue_loc[0] - gnb_loc[0]
    dy = ue_loc[1] - gnb_loc[1]
    distance = np.sqrt(dx**2 + dy**2)
    log_distance = np.log10(distance + 1)  # Avoid log(0)
    return [ue_loc[0], ue_loc[1], gnb_loc[0], gnb_loc[1], dx, dy, distance, log_distance]

def calculate_optimal_threshold(ue_locs, gnb_locs, percentile=90):
    all_distances = []
    for ue_loc in ue_locs:
        dists = cdist([ue_loc], gnb_locs).flatten()
        closest_dists = np.partition(dists, 4)[:4]
        all_distances.extend(closest_dists)
    
    threshold = np.percentile(all_distances, percentile)
    print(f"Optimal distance threshold: {threshold:.2f} meters (P{percentile})")
    return threshold

def clean_data(X, y):
    valid_mask = ~np.isnan(y)
    valid_mask &= ~np.isinf(y)
    valid_mask &= ~np.any(np.isnan(X), axis=1)
    valid_mask &= ~np.any(np.isinf(X), axis=1)
    
    return X[valid_mask], y[valid_mask]

def train_regression_clustering(ue_locs, gnb_locs, rsrp, distance_threshold):
    models = {}
    kmeans_models = {}
    num_gnbs = len(gnb_locs)
    
    for C in range(num_gnbs): 
        valid_ues = []
        for ue_idx, ue_loc in enumerate(ue_locs):
            dist = np.linalg.norm(ue_loc - gnb_locs[C])
            if dist <= distance_threshold:
                valid_ues.append(ue_idx)
        
        if not valid_ues:
            continue
            
        y = rsrp[valid_ues, C]
        
        nan_mask = ~np.isnan(y)
        y_clean = y[nan_mask]
        valid_ues_clean = [valid_ues[i] for i in range(len(valid_ues)) if nan_mask[i]]
        
        if len(y_clean) < 3:  
            continue
            
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(y_clean.reshape(-1, 1))
        kmeans_models[C] = kmeans
        cluster_models = {}
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            ue_indices_cluster = [valid_ues_clean[i] for i in cluster_indices]
            y_cluster = y_clean[cluster_indices]
            
            X_cluster = []
            for ue_idx in ue_indices_cluster:
                features = create_features(ue_locs[ue_idx], gnb_locs[C])
                X_cluster.append(features)
            
            X_cluster = np.array(X_cluster)
            y_cluster = np.array(y_cluster)
            
            X_cluster, y_cluster = clean_data(X_cluster, y_cluster)
            
            if len(X_cluster) == 0:
                continue
                
            model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                         min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(X_cluster, y_cluster)
            cluster_models[cluster_id] = model
        
        models[C] = cluster_models
    
    return models, kmeans_models

def test_regression_clustering(ue_locs, gnb_locs, rsrp, distance_threshold, 
                              train_models, kmeans_models):
    num_gnbs = len(gnb_locs)
    num_ues = len(ue_locs)
    all_residues = np.zeros((num_gnbs, num_ues)) * np.nan
    mean_residuals = np.zeros(num_gnbs)
    predicted_rsrp = np.zeros((num_gnbs, num_ues)) * np.nan
    
    for C in range(num_gnbs):
        if C not in train_models or C not in kmeans_models:
            continue
            
        valid_ues = []
        for ue_idx, ue_loc in enumerate(ue_locs):
            dist = np.linalg.norm(ue_loc - gnb_locs[C])
            if dist <= distance_threshold:
                valid_ues.append(ue_idx)
        
        if not valid_ues:
            continue
            
        residues = []
        for ue_idx in valid_ues:
            if np.isnan(rsrp[ue_idx, C]):
                continue
                
            features = create_features(ue_locs[ue_idx], gnb_locs[C])
            X = np.array([features])
            
            try:
                cluster_id = kmeans_models[C].predict([[rsrp[ue_idx, C]]])[0]
            except:
                continue
                
            if cluster_id not in train_models[C]:
                continue
                
            model = train_models[C][cluster_id]
            
            try:
                y_pred = model.predict(X)[0]
                predicted_rsrp[C, ue_idx] = y_pred
            except:
                continue
            
            residue = np.abs(rsrp[ue_idx, C] - y_pred)
            residues.append(residue)
            all_residues[C, ue_idx] = residue
        
        if residues:
            mean_residuals[C] = np.mean(residues)
    
    return all_residues, mean_residuals, predicted_rsrp

gnb_locs_train, ue_locs_train = read_topology('topology_train.csv')
rsrp_train = read_rsrp('rsrp_training.csv')

gnb_locs_test, ue_locs_test = read_topology('topology_test.csv')
rsrp_test = read_rsrp('rsrp_test.csv')


import sys

if len(sys.argv) > 1:
    distance_threshold = int(sys.argv[1])
else:
    print("threshold not provided.")


print("Training regression clustering models...")
train_models, kmeans_models = train_regression_clustering(
    ue_locs_train, gnb_locs_train, rsrp_train, distance_threshold
)

print("Testing regression clustering...")
all_residues, mean_residuals, predicted_rsrp = test_regression_clustering(
    ue_locs_test, gnb_locs_test, rsrp_test, distance_threshold,
    train_models, kmeans_models
)

suspicion_scores = mean_residuals
fake_idx = np.nanargmax(suspicion_scores)
actual_fbs_idx = 5

print("\n" + "="*60)
print("FAKE BASE STATION DETECTION RESULTS")
print("="*60)
print(f"Using distance threshold: {distance_threshold:.2f} meters")
print(f"{'gNB Index':<10} {'Mean Residual (dBm)':<20} {'Samples':<10} Status")
for idx in range(len(gnb_locs_test)):
    status = "FAKE!" if idx == fake_idx else "Genuine"
    samples = np.sum(~np.isnan(all_residues[idx]))
    print(f"{idx:<10} {mean_residuals[idx]:<20.3f} {samples:<10} {status}")

print("\n" + "="*60)
print(f"Detected Fake gNB: Index {fake_idx}") # (Actual Fake: Index {actual_fbs_idx})")
print("="*60)

os.makedirs("detection_plots", exist_ok=True)

plt.figure(figsize=(12, 6))
x = np.arange(len(gnb_locs_test))
bars = plt.bar(x, mean_residuals, color='skyblue')
if fake_idx < len(bars):
    bars[fake_idx].set_color('red')
if actual_fbs_idx < len(bars):
    bars[actual_fbs_idx].set_edgecolor('black')
    bars[actual_fbs_idx].set_linewidth(2)

plt.axhline(y=np.nanmean(mean_residuals), color='gray', linestyle='--', alpha=0.7)
plt.title('Fake gNB Detection - Mean Residuals', fontsize=16)
plt.xlabel('gNB Index', fontsize=12)
plt.ylabel('Mean Residual (dBm)', fontsize=12)
plt.xticks(x)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('detection_plots/mean_residuals.png')
plt.close()

plt.figure(figsize=(12, 6))
genuine_residuals = all_residues[0][~np.isnan(all_residues[0])]
fake_residuals = all_residues[fake_idx][~np.isnan(all_residues[fake_idx])]

if len(genuine_residuals) > 0 and len(fake_residuals) > 0:
    sns.kdeplot(genuine_residuals, label=f'Genuine gNB (0)', fill=True, alpha=0.5)
    sns.kdeplot(fake_residuals, label=f'Fake gNB ({fake_idx})', fill=True, alpha=0.5)

    plt.title('Residual Distribution Comparison', fontsize=16)
    plt.xlabel('Residual (dBm)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('detection_plots/residual_distribution.png')
    plt.close()

plt.figure(figsize=(12, 10))
valid_indices = ~np.isnan(rsrp_test[:, 0]) & ~np.isnan(predicted_rsrp[0])
if np.sum(valid_indices) > 0:
    plt.subplot(2, 1, 1)
    sns.scatterplot(x=rsrp_test[valid_indices, 0], y=predicted_rsrp[0][valid_indices], 
                    alpha=0.6, label=f'Genuine gNB (0)')
    plt.plot([-150, -60], [-150, -60], 'r--', alpha=0.5)
    plt.title('Actual vs Predicted RSRP (Genuine gNB)', fontsize=14)
    plt.xlabel('Actual RSRP (dBm)', fontsize=12)
    plt.ylabel('Predicted RSRP (dBm)', fontsize=12)
    plt.grid(alpha=0.3)

valid_indices = ~np.isnan(rsrp_test[:, fake_idx]) & ~np.isnan(predicted_rsrp[fake_idx])
if np.sum(valid_indices) > 0:
    plt.subplot(2, 1, 2)
    sns.scatterplot(x=rsrp_test[valid_indices, fake_idx], y=predicted_rsrp[fake_idx][valid_indices], 
                    alpha=0.6, label=f'Fake gNB ({fake_idx})')
    plt.plot([-150, -60], [-150, -60], 'r--', alpha=0.5)
    plt.title('Actual vs Predicted RSRP (Fake gNB)', fontsize=14)
    plt.xlabel('Actual RSRP (dBm)', fontsize=12)
    plt.ylabel('Predicted RSRP (dBm)', fontsize=12)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('detection_plots/actual_vs_predicted.png')
plt.close()

plt.figure(figsize=(14, 8))
residual_data = []
labels = []
for i in range(len(gnb_locs_test)):
    residuals = all_residues[i][~np.isnan(all_residues[i])]
    if len(residuals) > 0:
        residual_data.append(residuals)
        labels.append(f"gNB {i}")

if residual_data:
    box_colors = ['skyblue'] * len(labels)
    if fake_idx < len(labels):
        box_colors[fake_idx] = 'red'
    
    bplot = plt.boxplot(residual_data, patch_artist=True, tick_labels=labels)

    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)

    plt.title('Residual Distribution per gNB', fontsize=16)
    plt.xlabel('gNB Index', fontsize=12)
    plt.ylabel('Residual (dBm)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('detection_plots/residual_boxplot.png')
    plt.close()

plt.figure(figsize=(10, 8))
plt.scatter(gnb_locs_test[:, 0], gnb_locs_test[:, 1], s=150, c='blue', 
            marker='^', label='Genuine gNB', alpha=0.8)

if actual_fbs_idx < len(gnb_locs_test):
    plt.scatter(gnb_locs_test[actual_fbs_idx, 0], gnb_locs_test[actual_fbs_idx, 1], 
                s=300, c='red', marker='X', label='Actual Fake gNB', alpha=0.9)
if fake_idx < len(gnb_locs_test):
    plt.scatter(gnb_locs_test[fake_idx, 0], gnb_locs_test[fake_idx, 1], 
                s=300, c='purple', marker='*', label='Detected Fake gNB', alpha=0.9)

plt.scatter(ue_locs_test[:, 0], ue_locs_test[:, 1], s=50, c='green', 
            marker='o', label='UEs', alpha=0.6)

plt.title('Network Topology with Fake gNB Detection', fontsize=16)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('detection_plots/topology.png')
plt.close()

print("\nDetection results saved to 'detection_plots' directory")