import os
import time
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np

# ------------------------------------------------------------
# Imports from custom modules
# ------------------------------------------------------------
from stronghold_utils import (
    export_top_clusters_to_shapefile,
    export_cluster_summary_csv,
    plot_cluster_priority_gradient,
    clean_raw_density,
    clean_managed_strongholds,
    score_managed_strongholds,
    plot_master_summary_with_managed,
    plot_three_panel_migration_smoothed,
    filter_best_clusters_per_patch
)
from mcda_methods import (
    score_all_patches,
    score_clusters,
)
from cluster_methods import (
    build_rtree_index,
    build_connectivity_graph,
    collect_clusters,
    filter_top_n_unique_clusters
)


import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent
from scipy.ndimage import gaussian_filter
from matplotlib_scalebar.scalebar import ScaleBar


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PRIORITIES = [
    {
        "name": "ecology_large",
        "weights": {
            "total_core": 1.0,
            "feasibility_score": 0.8,
            "suppression_score": 1.0
        }
    },
]

SCENARIOS = {
    "cc_42": "../data/new/pm_cc_suppression_42.tif",
    "cc_83": "../data/new/pm_cc_suppression_83.tif",
    "cc_125": "../data/new/pm_cc_suppression_125.tif",
    "mean_42":"../data/new/pm_cc_suppression_42.tif",
    "mean_83": "../data/new/pm_cc_suppression_42.tif",
    "mean_125": "../data/new/pm_cc_suppression_42.tif",
}

AREA_FLOOR                   = 200    # minimum cluster area (ha)
AREA_LIMIT                   = 2500 # maximum cluster area (ha)
MAX_DISTANCE_M               = 300    # connectivity threshold (m)
BUFFER_DISTANCE              = 3000   # buffer for isolation cost (m)
FEASIBILITY_LAYER            = '../data/new/feasibility_focal.tif'
HABITAT_PATH                 = "../data/base/red_habitat_patches.shp"
MANAGED_STRONGHOLDS_SHP      = "../data/base/strongholds_in_cumbria.shp"
MANAGED_MASTER_SUMMARY_CSV   = "../outputs/managed_master_summary.csv"
ALGORITHMIC_MASTER_SUMMARY_CSV = "../outputs/algorithmic_master_stronghold_summary.csv"


def main():
    """
    Main workflow: loop through each strategy, then each scenario.
    Score patches, build clusters, pick top 5,
    then evaluate & plot managed strongholds.
    """
    start_time = time.time()

    # 1) Load habitat patches
    patches = gpd.read_file(HABITAT_PATH).reset_index(drop=True)
    assert patches.crs.is_projected, "Patches must be in a projected CRS."

    # 2) Load feasibility raster and replace nodata with zeros
    with rasterio.open(FEASIBILITY_LAYER) as src_feas:
        feas = src_feas.read(1)
        nodata = src_feas.nodata
        if nodata is not None:
            feas[feas == nodata] = 0

    # 3) Clean & filter managed Cumbria strongholds
    managed_gdf = clean_managed_strongholds(
        MANAGED_STRONGHOLDS_SHP,
        sitenames=["Whinfell", "Thirlmere", "Whinlatter", "Greystoke"],
        target_crs=patches.crs
    )

    # 4) Ensure master summary CSVs exist
    if not os.path.exists(MANAGED_MASTER_SUMMARY_CSV):
        pd.DataFrame(columns=[
            "scenario", "sitename",
            "total_area_ha", "total_core_ha",
            "suppression_score", "feasibility_score",
            "final_score", "percentile"
        ]).to_csv(MANAGED_MASTER_SUMMARY_CSV, index=False)

    if not os.path.exists(ALGORITHMIC_MASTER_SUMMARY_CSV):
        pd.DataFrame(columns=[
            "cluster_id", "scenario", "strategy",
            "total_area_ha", "total_core_ha",
            "grey_cost", "buffer_cost",
            "final_score", "feasibility_score"
        ]).to_csv(ALGORITHMIC_MASTER_SUMMARY_CSV, index=False)

    combined_gdfs = []  # for final shapefile export

    # 5) Loop through each strategy, then each scenario
    for strat in PRIORITIES:
        strat_name = strat["name"]
        weights    = strat["weights"]

        for scenario_name, grey_path in SCENARIOS.items():
            print(f"Running strategy {strat_name} | scenario {scenario_name}")

            # load and clean suppression raster for this scenario
            with rasterio.open(grey_path) as src:
                grey = clean_raw_density(src)
                transform = src.transform

            export_dir = f"../outputs/{scenario_name}_{strat_name}_buffer/"
            os.makedirs(export_dir, exist_ok=True)

            # 5a) Score all patches
            scored_patches = score_all_patches(
                patches, grey, feas, weights, transform
            )

            # 5b) Build spatial graph
            idx = build_rtree_index(scored_patches)
            G   = build_connectivity_graph(
                scored_patches, idx, MAX_DISTANCE_M
            )

            # 5c) Collect & score clusters
            clusters = collect_clusters(
                G, AREA_LIMIT, AREA_FLOOR,
                grey, transform, feas, BUFFER_DISTANCE
            )
            scored_clusters = score_clusters(
                clusters, weights, G
            )
            # 👉 Print number of clusters before filtering
            print(f"Scenario {scenario_name}: {len(scored_clusters)} clusters generated (pre-filtering)")


            # 5d) Pick top 5 non-overlapping clusters
            top_clusters = filter_top_n_unique_clusters(
                scored_clusters,
                n=len(scored_clusters),
                area_floor=AREA_FLOOR,
                area_limit=AREA_LIMIT
            )
            
            filtered_clusters = filter_best_clusters_per_patch(scored_clusters)
            
            # 👉 Print number of clusters after filtering
            print(f"Scenario {scenario_name}: {len(filtered_clusters)} clusters retained (post-filtering)")

            # 5e) Export top-5 shapefiles
            top_gdf = export_top_clusters_to_shapefile(
                patches_gdf=scored_patches,
                scored_clusters=top_clusters,
                export_dir=export_dir,
                buffer_distance=BUFFER_DISTANCE,
                top_n=len(top_clusters),
                shapefile_name=f"top5_strongholds_{scenario_name}_{strat_name}"
            )
            combined_gdfs.append(top_gdf)

            # 5f) Summary CSV of all clusters
            summary_csv = os.path.join(
                export_dir,
                f"summary_top5_{scenario_name}_{strat_name}.csv"
            )
            export_cluster_summary_csv(
                scored_clusters=scored_clusters,
                export_dir=export_dir,
                top_n=len(scored_clusters),
                PRIORITIES=PRIORITIES,
                filename=os.path.basename(summary_csv),
                scenario_name=scenario_name,
                strategy_name=strat_name,
                master_csv_path=ALGORITHMIC_MASTER_SUMMARY_CSV
            )

            # 5g) Score managed strongholds
            managed_df = score_managed_strongholds(
                managed_gdf,
                scored_patches,
                scored_clusters,
                weights
            )
            managed_csv = os.path.join(export_dir, "managed_strongholds_scores.csv")
            managed_df.to_csv(managed_csv, index=False)
            managed_df["scenario"] = scenario_name
            managed_df.to_csv(
                MANAGED_MASTER_SUMMARY_CSV,
                mode="a",
                header=False,
                index=False
            )
            
            # 5k) Plot 3‐panel migration map for this strategy
            plot_three_panel_migration_smoothed(
                scenarios=SCENARIOS,
                strategy_weights=strat["weights"],
                patches_gdf=scored_patches,
                managed_gdf=managed_gdf,
                area_floor=AREA_FLOOR,
                area_limit=AREA_LIMIT,
                max_distance=MAX_DISTANCE_M,
                buffer_distance=BUFFER_DISTANCE,
                clean_raw_density=clean_raw_density,
                build_rtree_index=build_rtree_index,
                build_connectivity_graph=build_connectivity_graph,
                collect_clusters=collect_clusters,
                score_clusters=score_clusters,
                filter_top_n_unique_clusters=filter_top_n_unique_clusters,
                output_path=f"../outputs/migration_{strat_name}.png"
            )
            
            # 5h) Plot percentile vs score
            plot_master_summary_with_managed(
                summary_csv=summary_csv,
                managed_scores_csv=managed_csv,
                scenario_name=scenario_name,
                output_path=os.path.join(
                    "../outputs", f"{scenario_name}_{strat_name}_score_vs_percentile.png"
                )
            )
            
    # 6) Export combined shapefile
    if combined_gdfs:
        all_clusters_gdf = gpd.GeoDataFrame(
            pd.concat(combined_gdfs, ignore_index=True),
            crs=patches.crs
        )
        all_clusters_gdf.to_file(
            "../outputs/all_strongholds_with_cluster_id.shp"
        )

    # 7) Report runtime
    print(f"Completed in {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
