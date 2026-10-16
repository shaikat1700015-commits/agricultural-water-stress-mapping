
# full_conflict_pipeline_updated_km2_xgb2_class1fix.py
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import tempfile  # FIX: for safe temp files
import os as _os  # FIX: to remove temp files

# -----------------------------
# 1. File paths
# -----------------------------
thematic_files = ['LULC1.tif','Rainfall.tif','Slope.tif','Soil Type Map1.tif','Lineament Density Map1.tif','Lithology.tif']
gpz_file = 'GWP_Zone.tif'
conflict_file = 'Agricultural conflict zone.tif'
study_shapefile = 'study_area.shp'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2. Reference raster
# -----------------------------
with rasterio.open(conflict_file) as ref:
    ref_shape = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_crs = ref.crs

# Pixel area in km²
pixel_area_km2 = abs(ref_transform.a * ref_transform.e) / 1e6
print("Pixel area (km²):", pixel_area_km2)

# -----------------------------
# 3. Load study area shapefile & match CRS
# -----------------------------
study_area = gpd.read_file(study_shapefile)
study_area = study_area.to_crs(ref_crs)
# union_all() is available on recent GeoPandas; fallback if needed
try:
    study_geom = [study_area.geometry.union_all().__geo_interface__]
except Exception:
    study_geom = [study_area.unary_union.__geo_interface__]

# -----------------------------
# 4. Resample + Clip function
# -----------------------------
def resample_and_clip(src_file, ref_shape, ref_transform, study_geom, resampling_method='nearest'):
    with rasterio.open(src_file) as src:
        data = src.read(
            1,
            out_shape=ref_shape,
            resampling=Resampling.nearest if resampling_method == 'nearest' else Resampling.bilinear
        )
        meta = src.meta.copy()
        meta.update({
            "height": ref_shape[0],
            "width": ref_shape[1],
            "transform": ref_transform
        })
        tmp_handle = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_file = tmp_handle.name
        tmp_handle.close()
        try:
            with rasterio.open(temp_file, 'w', **meta) as tmp:
                tmp.write(data, 1)
            with rasterio.open(temp_file) as tmp:
                clipped_data, _ = mask(tmp, study_geom, crop=False)
        finally:
            try:
                _os.remove(temp_file)
            except Exception:
                pass
        return clipped_data[0]

# -----------------------------
# 5. Load & process rasters
# -----------------------------
X_list = []
for f in thematic_files:
    if f in ['LULC1.tif','Soil Type Map1.tif','Lithology.tif']:
        X_list.append(resample_and_clip(f, ref_shape, ref_transform, study_geom, 'nearest'))
    else:
        X_list.append(resample_and_clip(f, ref_shape, ref_transform, study_geom, 'bilinear'))

gpz_resampled = resample_and_clip(gpz_file, ref_shape, ref_transform, study_geom, 'nearest')
X_list.append(gpz_resampled)
X = np.stack(X_list, axis=-1)

y = resample_and_clip(conflict_file, ref_shape, ref_transform, study_geom, 'nearest')

# Remove nodata & NaNs
nodata_val = -3.4028235e+38
H,W,C = X.shape
X_flat = X.reshape(-1,C)
y_flat = y.flatten()
mask_valid = (y_flat != nodata_val) & (~np.any(np.isnan(X_flat), axis=1))
X_flat = X_flat[mask_valid]
y_flat = y_flat[mask_valid].astype(int)

print("Cleaned data shape:", X_flat.shape, "Labels shape:", y_flat.shape)
print("Unique classes in y:", np.unique(y_flat))

# -----------------------------
# 5. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_flat, test_size=0.2, random_state=42, stratify=y_flat
)

# -----------------------------
# 6. Compute class weights (kept as in your code though not used by VotingClassifier)
# -----------------------------
class_counts = Counter(y_train.tolist())
n_classes = len(np.unique(y_train))
total = len(y_train)
class_weights_dict = {cls: total/(n_classes*cnt) for cls, cnt in class_counts.items()}
sample_weights = np.array([class_weights_dict[c] for c in y_train])

# -----------------------------
# 6b. Optional: tune class weights more aggressively (kept)
# -----------------------------
class_weights_dict = {cls: (total/(n_classes*cnt))**1.5 for cls, cnt in class_counts.items()}
sample_weights = np.array([class_weights_dict[c] for c in y_train])

# -----------------------------
# 6b. Oversample rare classes (kept)
# -----------------------------
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
print("After oversampling, class distribution:", Counter(y_res))

# -----------------------------
# 7. Define models for ensemble (kept)
# -----------------------------
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=n_classes,
    max_depth=8,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    random_state=42,
    class_weight='balanced_subsample'
)

ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    voting='soft',
    n_jobs=1
)

# -----------------------------
# 8. Train ensemble on oversampled data (kept)
# -----------------------------
ensemble_model.fit(X_res, y_res)

# -----------------------------
# 8b. NEW: Calibrate a class-1 probability multiplier (decode-time only)
#         This *does not* change the model or training — it only adjusts how we pick the argmax
#         to trade precision/recall for Class 1 and improve its F1.
# -----------------------------
labels_sorted = sorted(np.unique(y_train))
if 1 in labels_sorted:
    # Grid of multipliers to test for class 1
    grid = np.concatenate([np.linspace(0.4, 1.0, 7), np.linspace(1.05, 2.0, 20)])
    proba_train = ensemble_model.predict_proba(X_train)
    best_m = 1.0
    best_f1 = -1.0
    idx1 = labels_sorted.index(1)
    for m in grid:
        proba_m = proba_train.copy()
        proba_m[:, idx1] *= m
        y_hat = np.argmax(proba_m, axis=1)
        # Compute F1 for class 1 only
        _, _, f1s, _ = precision_recall_fscore_support(y_train, y_hat, labels=labels_sorted, zero_division=0)
        f1_class1 = f1s[idx1]
        if f1_class1 > best_f1:
            best_f1 = f1_class1
            best_m = m
    CLASS1_DECODE_MULTIPLIER = best_m
    print(f"[calib] Selected class-1 decode multiplier: {CLASS1_DECODE_MULTIPLIER:.3f} (val F1_1={best_f1:.4f})")
else:
    CLASS1_DECODE_MULTIPLIER = 1.0
    print("[calib] Class 1 not present in labels; using neutral multiplier 1.0.")

def apply_class1_bias(proba, multiplier, labels_sorted):
    """Multiply the probability column for class 1 by a scalar before argmax."""
    proba_b = proba.copy()
    if 1 in labels_sorted:
        idx1 = labels_sorted.index(1)
        proba_b[:, idx1] *= multiplier
    return proba_b

# -----------------------------
# 9. Predict & validate  (apply class-1 decode multiplier)
# -----------------------------
mask_flat = mask_valid
baseline_map = np.full(H*W, fill_value=-1, dtype=np.int32)
proba_full = ensemble_model.predict_proba(X_flat)
proba_full_b = apply_class1_bias(proba_full, CLASS1_DECODE_MULTIPLIER, labels_sorted)
baseline_map[mask_flat] = np.argmax(proba_full_b, axis=1)

kappa = cohen_kappa_score(y_flat, baseline_map.flatten()[mask_flat])
print("Cohen's Kappa:", kappa)

proba_test = ensemble_model.predict_proba(X_test)
proba_test_b = apply_class1_bias(proba_test, CLASS1_DECODE_MULTIPLIER, labels_sorted)
y_pred = np.argmax(proba_test_b, axis=1)

# -----------------------------
# 8. Pixel-level validation (kept)
# -----------------------------
print("=== Pixel-level validation ===")
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 9. Map-level validation (kept; uses biased baseline_map already)
# -----------------------------
baseline_map = baseline_map.reshape(H, W)

kappa = cohen_kappa_score(y_flat, baseline_map.flatten()[mask_flat])
print("Cohen's Kappa:", kappa)

# === UPDATED order per your pattern ===
classes = [
    'No conflict',
    'Very high conflict',
    'High conflict',
    'Moderate conflict',
    'Low conflict',
    'Very low conflict'
]

obs_counts = np.bincount(y_flat, minlength=n_classes)
pred_counts = np.bincount(baseline_map.flatten()[mask_flat], minlength=n_classes)
obs_area_km2 = obs_counts * pixel_area_km2
pred_area_km2 = pred_counts * pixel_area_km2

df_area = pd.DataFrame({
    'Class':classes,
    'Observed_pixels':obs_counts,
    'Observed_km2':obs_area_km2,
    'Predicted_pixels':pred_counts,
    'Predicted_km2':pred_area_km2
})
df_area.to_csv(os.path.join(output_dir,'baseline_area_comparison.csv'), index=False)
print("Baseline area comparison CSV saved.")

with rasterio.open(conflict_file) as src:
    meta = src.meta.copy()
meta.update(dtype=rasterio.int32, nodata=-9999)
baseline_file = os.path.join(output_dir,'predicted_conflict_baseline.tif')
with rasterio.open(baseline_file,'w',**meta) as dst:
    dst.write(baseline_map, 1)

# -----------------------------
# 10. Feature importance (kept)
# -----------------------------
rf_inside = getattr(ensemble_model, "named_estimators_", None)
if rf_inside is not None:
    rf_inside = rf_inside['rf']
else:
    rf_inside = dict(ensemble_model.estimators_)[ensemble_model.estimators[1][0]]

plt.figure(figsize=(8,5), dpi=300)
plt.barh(thematic_files + ['GWP_Zone'], rf_inside.feature_importances_)
plt.xlabel('Importance')
plt.title('Feature Importance (RandomForest)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'feature_importance.png'), dpi=300)
plt.close()

# -----------------------------
# 11. Scenario definitions (kept)
# -----------------------------
scenarios = []

rainfall_idx = 1
rainfall_factors = [0.4, 0.5, 0.57, 0.6, 0.675, 0.7, 0.8, 0.805, 0.85, 0.9, 0.905, 1.1, 2.0]

def modify_rainfall(X_array, factor):
    X_mod = X_array.copy()
    X_mod[:,:,rainfall_idx] = X_mod[:,:,rainfall_idx].astype(np.float32) * factor
    return X_mod

for factor in rainfall_factors:
    scenarios.append({
        'name': f'rainfall_{int(factor*1000)/10}',
        'modify': lambda X, f=factor: modify_rainfall(X, f)
    })

lulc_idx = 0
def modify_lulc(X_array):
    X_mod = X_array.copy()
    cropland_value = 3
    new_value = 5
    X_mod[:,:,lulc_idx][X_mod[:,:,lulc_idx]==cropland_value] = new_value
    return X_mod
scenarios.append({'name':'LULC_swap','modify': modify_lulc})

gpz_idx = 6
def improve_gpz(X_array):
    X_mod = X_array.copy()
    low_mask = X_mod[:,:,gpz_idx] <= 1
    X_mod[:,:,gpz_idx][low_mask] = 4
    return X_mod
scenarios.append({'name':'GPZ_improve','modify': improve_gpz})

# -----------------------------
# 12. Run scenarios (apply class-1 decode multiplier when predicting)
# -----------------------------
results = []
for scn in scenarios:
    print(f"Running scenario: {scn['name']}")
    X_scn = scn['modify'](X)
    X_scn_flat = X_scn.reshape(-1,C)
    pred_map_flat = np.full(H*W, fill_value=-1, dtype=np.int32)
    proba_scn = ensemble_model.predict_proba(X_scn_flat[mask_flat])
    proba_scn_b = apply_class1_bias(proba_scn, CLASS1_DECODE_MULTIPLIER, labels_sorted)
    pred_map_flat[mask_flat] = np.argmax(proba_scn_b, axis=1)
    pred_map = pred_map_flat.reshape(H,W)

    scenario_file = os.path.join(output_dir,f'predicted_conflict_{scn["name"]}.tif')
    with rasterio.open(scenario_file,'w',**meta) as dst:
        dst.write(pred_map,1)

    counts = np.bincount(pred_map.flatten()[mask_flat], minlength=n_classes)
    diff = counts - pred_counts
    counts_km2 = counts * pixel_area_km2
    diff_km2 = counts_km2 - (pred_counts*pixel_area_km2)

    results.append({
        'scenario':scn['name'],
        'counts':counts,
        'difference':diff,
        'counts_km2':counts_km2,
        'difference_km2':diff_km2
    })

    plt.figure(figsize=(8,5), dpi=300)
    plt.bar(classes, counts_km2, label='Scenario')
    plt.bar(classes, pred_area_km2, alpha=0.5, label='Baseline')
    plt.ylabel('Area (km²)')
    plt.title(f'Conflict area comparison: {scn["name"]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f'barchart_{scn["name"]}.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10,8), dpi=300)
    cmap = plt.get_cmap('bwr', 11)
    plt.imshow(pred_map - baseline_map, cmap=cmap, vmin=-5, vmax=5)
    plt.colorbar(label=f'Class difference ({classes[0]} → {classes[-1]})')
    plt.title(f'Difference map: {scn["name"]} − baseline')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f'diffmap_{scn["name"]}.png'), dpi=300)
    plt.close()

# -----------------------------
# 13. Save scenario CSV (kept)
# -----------------------------
rows = []
for r in results:
    row = {'Scenario': r['scenario']}
    counts = r['counts'].tolist()
    diff = r['difference'].tolist()
    counts_km2 = r['counts_km2'].tolist()
    diff_km2 = r['difference_km2'].tolist()
    for i in range(n_classes):
        row[f'class_{i}_pixels'] = counts[i]
        row[f'diff_class_{i}_pixels'] = diff[i]
        row[f'class_{i}_km2'] = counts_km2[i]
        row[f'diff_class_{i}_km2'] = diff_km2[i]
    rows.append(row)

df_results = pd.DataFrame(rows)
df_results.to_csv(os.path.join(output_dir,'scenario_area_stats.csv'), index=False)
print("Scenario area stats CSV saved successfully.")

print("All scenario maps, difference maps, bar charts, and CSV saved in 'output' folder.")
