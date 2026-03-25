REBUILD_MODE = True  # Set to True to rebuild datasets from raw sources; False to load existing cleaned CSVs.

if REBUILD_MODE:
    print("REBUILD_MODE=True: Regenerating datasets from raw sources...")

    def rebuild_final_glorich_from_raw() -> pd.DataFrame:
        """
        Rebuild final_glorich_dataset from imputed_conditions + stations.
        See: Final_imputed_Glorich.ipynb
        """
        try:
            glorich_raw = pd.read_csv(BASE_DIR / 'final_glorich_dataset.csv')
            imputed_hydro = pd.read_csv(BASE_DIR / 'imputed_conditions_11-15.csv')
        except Exception as exc:
            raise FileNotFoundError(f"Glorich source files not found: {exc}") from exc

        glorich_df = glorich_raw[['STAT_ID', 'Latitude', 'Longitude']].drop_duplicates()

        final_glorich = pd.merge(
            glorich_df,
            imputed_hydro,
            on='STAT_ID',
            how='left'
        )

        if 'date' in final_glorich.columns:
            final_glorich['date'] = _to_datetime_mixed(final_glorich['date'])

        rel_col = 'SpecCond25C_reliability' if 'SpecCond25C_reliability' in final_glorich.columns else None
        if rel_col:
            final_glorich = (
                final_glorich
                .sort_values(rel_col, ascending=False)
                .drop_duplicates(subset=['Latitude', 'Longitude', 'date'], keep='first')
                .reset_index(drop=True)
            )

        return clean_final_glorich(final_glorich)

    def rebuild_landsat_from_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rebuild landsat_features_training and landsat_features_validation.
        This would normally fetch from Landsat C2 L2 API; here we load pre-downloaded versions.
        See: Given_Data_EDA_fixed.ipynb (Landsat API section)
        """
        try:
            train_ls = pd.read_csv(BASE_DIR / 'landsat_features_training.csv')
            val_ls = pd.read_csv(BASE_DIR / 'landsat_features_validation.csv')
        except Exception as exc:
            raise FileNotFoundError(f"Landsat files not found: {exc}") from exc

        train_cleaned = clean_landsat(train_ls)
        val_cleaned = clean_landsat(val_ls)

        return train_cleaned, val_cleaned

    def rebuild_dws_from_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rebuild dws_matched_all_columns and dws_test from DWS scraper + spatial/temporal matching.
        See: 04_dws_scraper.ipynb, 05_dws_cleaning.ipynb
        """
        try:
            dws_matched = pd.read_csv(BASE_DIR / 'dws_matched_all_columns.csv')
            dws_test = pd.read_csv(BASE_DIR / 'dws_test.csv')
        except Exception as exc:
            raise FileNotFoundError(f"DWS matched files not found: {exc}") from exc

        dws_matched_clean = clean_dws_matched(dws_matched)
        dws_test_clean = clean_dws_matched(dws_test)

        return dws_matched_clean, dws_test_clean

    def rebuild_train_test_reliability() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rebuild train_ALL+reliability and test_ALL+reliability by spatially and temporally
        joining Glorich water quality + Landsat spectral features.
        See: training_pipeline.ipynb
        """
        try:
            glorich_df = rebuild_final_glorich_from_raw()
            landsat_train, landsat_val = rebuild_landsat_from_raw()
        except Exception as exc:
            raise RuntimeError(f"Cannot rebuild reliability: {exc}") from exc

        def merge_glorich_landsat(glorich, landsat, split_name='train'):
            landsat = landsat.copy()
            glorich = glorich.copy()

            if 'Sample Date' in landsat.columns:
                landsat['Sample Date'] = _to_datetime_mixed(landsat['Sample Date'])
            if 'date' in glorich.columns:
                glorich['date'] = _to_datetime_mixed(glorich['date'])

            # Tag order
            landsat['_merge_order'] = range(len(landsat))

            # Spatial: Nearest station per landsat point (via haversine)
            landsat_coords = landsat[['Latitude', 'Longitude']].values
            station_coords = glorich.groupby(['Latitude', 'Longitude']).size().reset_index(name='n')[
                ['Latitude', 'Longitude']
            ].values

            if len(station_coords) == 0:
                return pd.DataFrame()

            from scipy.spatial.distance import cdist

            dist_m = cdist(landsat_coords, station_coords, metric='euclidean') * 111_000

            nearest_idx = dist_m.argmin(axis=1)
            nearest_lat = station_coords[nearest_idx, 0]
            nearest_lon = station_coords[nearest_idx, 1]

            landsat['_target_lat'] = nearest_lat
            landsat['_target_lon'] = nearest_lon

            # Temporal: Merge-asof by date
            result_rows = []
            for (target_lat, target_lon), group in landsat.groupby(['_target_lat', '_target_lon']):
                glorich_at_station = glorich[
                    (glorich['Latitude'] == target_lat) & (glorich['Longitude'] == target_lon)
                ]

                if len(glorich_at_station) == 0:
                    for idx, row in group.iterrows():
                        row_dict = row.to_dict()
                        result_rows.append(row_dict)
                else:
                    glorich_sorted = glorich_at_station.dropna(subset=['date']).sort_values('date')

                    if len(glorich_sorted) == 0:
                        for idx, row in group.iterrows():
                            row_dict = row.to_dict()
                            result_rows.append(row_dict)
                        continue

                    for idx, landsat_row in group.iterrows():
                        sample_date = pd.to_datetime(landsat_row['Sample Date'], errors='coerce')

                        if pd.isna(sample_date):
                            row_dict = landsat_row.to_dict()
                            result_rows.append(row_dict)
                        else:
                            dates = glorich_sorted['date'].values
                            diffs = np.abs(dates - np.datetime64(sample_date))
                            best_pos = np.argmin(diffs)
                            best_glorich = glorich_sorted.iloc[best_pos]

                            row_dict = landsat_row.to_dict()
                            for col in glorich_sorted.columns:
                                row_dict[f'{col}'] = best_glorich[col]

                            diff_val = diffs[best_pos] / np.timedelta64(1, 'D')
                            row_dict['date_diff_days'] = int(diff_val) if np.isfinite(diff_val) else None

                            result_rows.append(row_dict)

            merged = pd.DataFrame(result_rows).sort_values('_merge_order').reset_index(drop=True)
            merged = merged.drop(columns=['_merge_order', '_target_lat', '_target_lon'], errors='ignore')

            # Dedup by reliability
            if 'SpecCond25C_reliability' in merged.columns:
                merged = (
                    merged
                    .sort_values('SpecCond25C_reliability', ascending=False)
                    .drop_duplicates(subset=['Latitude', 'Longitude', 'Sample Date'], keep='first')
                    .reset_index(drop=True)
                )

            return merged

        train_merged = merge_glorich_landsat(glorich_df, landsat_train, split_name='train')
        test_merged = merge_glorich_landsat(glorich_df, landsat_val, split_name='test')

        train_clean = clean_reliability_matrix(train_merged)
        test_clean = clean_reliability_matrix(test_merged)

        return train_clean, test_clean

    # Execute rebuild if flag is set
    print("\nRebuilding intermediate datasets...")
    final_glorich_rebuilt = rebuild_final_glorich_from_raw()
    print(f"  final_glorich: {final_glorich_rebuilt.shape}")

    landsat_train_rebuilt, landsat_val_rebuilt = rebuild_landsat_from_raw()
    print(f"  landsat_training: {landsat_train_rebuilt.shape}")
    print(f"  landsat_validation: {landsat_val_rebuilt.shape}")

    dws_matched_rebuilt, dws_test_rebuilt = rebuild_dws_from_raw()
    print(f"  dws_matched_all_columns: {dws_matched_rebuilt.shape}")
    print(f"  dws_test: {dws_test_rebuilt.shape}")

    print("\nRebuilding reliability matrices...")
    train_rebuilt, test_rebuilt = rebuild_train_test_reliability()
    print(f"  train_ALL+reliability: {train_rebuilt.shape}")
    print(f"  test_ALL+reliability: {test_rebuilt.shape}")

    print("\nRebuild complete. Loaded datasets are now in train_rebuilt / test_rebuilt.")
    print("To use rebuilt datasets, set train_df = train_rebuilt, test_df = test_rebuilt in the next cell.")

else:
    print("REBUILD_MODE=False: Skipping rebuild; will load pre-existing cleaned CSVs in next section.")