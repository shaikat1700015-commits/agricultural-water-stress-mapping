# Groundwater-related agricultural conflict mapping with machine learning

This repo holds the code I used to map zones of agricultural conflict over groundwater and to see how those zones shift under rainfall, land-use and groundwater-potential scenarios. The model is a soft-voting ensemble of Random Forest and XGBoost, trained on pixels from co-registered raster layers.



## What the script does

1. Resamples every input raster to the grid of the conflict-zone raster and clips it to the study area.
2. Splits valid pixels into train, validation and test sets. The default is a spatial block split, so neighbouring pixels do not end up on both sides of the split.
3. Oversamples the training set (random oversampling) and fits the RF + XGBoost ensemble.
4. Picks a multiplier for one class's predicted probability (default: class 1) using the validation set only.
5. Reports accuracy, kappa and a per-class report on the held-out test set.
6. Predicts the full map and writes area tables (km2) and rasters.
7. Re-predicts under scenarios: 13 rainfall factors (0.4x to 2.0x), a cropland-to-other-class swap, and a groundwater-potential upgrade.

## Input data



 Land use & land cover.tif,   
,Rainfall.tif
,Slope.tif
,Soil Type Map1.tif
,Lineament Density Map1.tif
,Lithology.tif
,GWP_Zone.tif
,Agricultural conflict zone.tif
study_area.shp

Class order in the target raster: 0 No conflict, 1 Very high, 2 High, 3 Moderate, 4 Low, 5 Very low.

The `.tif` and `.shp` files are large to redistribute, so they are ignored by git. 
## Run it

```bash
pip install -r requirements.txt
python conflict_prediction_pipeline.py
```

Settings (file names, split type, scenario values) sit in the CONFIG block at the top of the script. Results land in `output/`:

- `baseline_area_comparison.csv`: observed vs predicted area per class
- `scenario_area_stats.csv`: area and change per class for every scenario
- `predicted_conflict_*.tif`: baseline and scenario maps
- `diffmap_*.png`, `barchart_*.png`: scenario minus baseline
- `feature_importance.png`: RF and XGBoost importances



## Things to keep in mind

- The full-map kappa printed by the script includes training pixels. It describes agreement, not accuracy. Use the test-set numbers for performance.
- The class multiplier is tuned on the validation set, so it does not touch the test set.
- `GWP_Zone` is used as a predictor. If the conflict classes were built partly from the same layers that produced `GWP_Zone`, the model can pick up that link instead of real spatial patterns.
- Scenario runs change one input at a time and hold the others fixed. They show how the model responds, not a forecast.
- Pixel areas assume the rasters are in a projected (metric) CRS. The script warns if they are not.

## Contact

Md. Shaikat Chowdhury, Civil Engineering, RUET, Bangladesh
[LinkedIn: www.linkedin.com/in/shaikat-chowdhury-a86182329]
