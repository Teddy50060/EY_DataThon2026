# Water Quality Prediction: Model Explanation
---
## Model Development Approach 
1. Data Wrangling 
The notebook aligns the EY training and submission files with the provided TerraClimate and Landsat tables using Latitude, Longitude, and Sample Date. Dates are standardized, and coordinates are rounded before merging. GLORICH and DWS features are then added using nearest-station spatial matching and nearest-date temporal matching, as those datasets don’t share the same coordinates or dates as the EY records. The final merged tables used for modeling contain 9,319 rows × 75 columns for the train and 200 rows × 75 columns for the test.

2. Feature Engineering 
Landsat: We fetched the raw bands nir, green, swir16, swir22, and derived indices NDMI and MNDWI, and later create nir_green_ratio. 
GLORICH: These explained the chemistry and static watershed descriptors. Additional engineered variables are log_Popdens_00, log_SOC, log_dist_km, year, month. 
DWS: We added the nearest measured water-quality variables, and created dws_P_modified by capping the original DWS phosphorus values to 0.01–0.2 mg/L and converting them to 10–200 µg/L
Dates are standardized, and coordinates are rounded before merging. 

3. Modeling 
Before we model, we prune the highly correlated features at a 0.95 correlation threshold; the final modeling feature set contains 49 features. For model development, the final notebook trains both CatBoostRegressor and LightGBMRegressor for each target. We tuned the hyperparameters with Optuna and used a 5-fold GroupKFold with groups defined by 5 spatial clusters from KMeans on latitude/longitude for validation. 

DRP is modeled on a log1p scale and transformed back after prediction. We got CatBoost cross-validation performance of R² = 0.9869 for Total Alkalinity, 0.8971 for Electrical Conductance, and 0.6301 for DRP. LightGBM is also trained and ensembled in the final submission, the final printed LightGBM CV values R² = 0.9888 for Total Alkalinity, 0.9184 for Electrical Conductance, and 0.6698 for DRP.

The final submission uses a 50/50 CatBoost–LightGBM ensemble for all three targets, with one final rule-based override: Total Alkalinity is replaced with DWS TAL wherever DWS data are available. This reflects the earlier model development logic noted in the notebook: DWS-dominated features worked especially well for alkalinity and conductance, while DRP benefited more from combining DWS with broader environmental features such as Landsat and GLORICH. For DRP, the final notebook keeps the ensemble prediction. 

## Datasets Used 
---
Beyond the Terraclimate features given, we added more features in Landsat data and used other data from publicly available datasets as follows: 

Landsat data 
We added Landsat spectral information beyond the provided baseline tables. The notebook used nir, green, swir16, and swir22. NDMI, MNDWI, and nir_green_ratio are derived.
Data imputation: Missing Landsat values are first searched through the Microsoft Planetary Computer API using cloud/QA filtering and widening date windows; any remaining gaps are filled with a KNN imputer based on latitude, longitude, and PET. This provides atmospherically corrected surface reflectance products and QA bands that support cloud-aware pixel selection.

GLORICH - Global River Chemistry Database 
(Hartmann et al., 2014; URL https://doi.org/10.1594/PANGAEA.902360)
We used 3 datasets:  hydrochemistry.csv, catchment_properties.csv, and sampling_locations.csv. 
Data filtering: Hydrochemistry is filtered to years after 2000, aggregated by station and month, and restricted to South African stations. After filtering, the Africa-filtered locations have 1,542 rows and 8 columns, and the selected catchment table contains 1,210 unique stations. Chemistry variables retained are pH, SpecCond25C, Alkalinity, Cl, SO4, and DIP.
Data extrapolation to 2015: As GLORICH data ends in 2011, we need to impute time-based values. Imputation is based on per-station stationarity diagnostics: stationary series use a historical mean, diff-stationary series use the last value, trend-stationary series use linear extrapolation, and non-stationary series use a median-based fallback. Numerical reliability columns are also created for each chemistry variable, based on the type of imputation. 
Data merge: The GLORICH is merged by nearest station by spatial join, then nearest date within station by merge_asof. In the final merge, the notebook reports 150 GLORICH stations matched in training and 23 in testing, with a median lag of 4 days.

South African Department of Water and Sanitation (DWS) National Surface Water
URL: https://www.dws.gov.za/iwqs/wms/data/{region_letter}_reg_WMS_nobor.htm 
The data was systematically scraped across all 22 drainage regions (labelled A through X), accessible via a region-specific URL. Each region page lists monitoring stations with their station ID, geographic coordinates, sample count, and date range, along with a downloadable ZIP file of the station's full historical record. After filtering from 2010-01-01 through 2016-12-31, the notebook retains 168,999 records across 3,844 stations.
Data extraction and cleaning process: The DWS variables used are TAL, EC, PO4_P, pH, Ca, Mg, Na, Cl, SO4, and P_Tot. Each EY row is linked to the nearest DWS station and then to the nearest observation in time. 
