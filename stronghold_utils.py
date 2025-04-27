import os
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.mask
import numpy as np
from shapely.ops import unary_union
from shapely.errors import TopologicalError
from rtree import index
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mcda_methods import compute_z_stats, mcda_score
import rasterio
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent
from scipy.ndimage import gaussian_filter
from matplotlib_scalebar.scalebar import ScaleBar
# ------------------------------------------------------------
# Clean raw grey-squirrel density raster
# ------------------------------------------------------------
def clean_raw_density(src):
    """
    Read a grey-squirrel density raster, mask invalid/outlier values, and
    return a cleaned numpy array.
    """
    density = src.read(1)
    nodata = src.nodata

    # Mask NaNs and implausible values
    density = np.where((np.isnan(density)) | (density < 0) | (density > 9998), 0, density)

    # Remove urban density outliers (>4) as per domain knowledge
    density = np.where(density > 4.0, 0, density)

    # Ensure nodata is zeroed
    if nodata is not None:
        density[density == nodata] = 0
    return density

# ------------------------------------------------------------
# Export top clusters as shapefiles
# ------------------------------------------------------------
def export_top_clusters_to_shapefile(
    patches_gdf,
    scored_clusters,
    export_dir,
    buffer_distance,
    top_n,
    shapefile_name="top_clusters"
):
    """
    Export patch-level and cluster-level shapefiles for the top N clusters.
    """
    os.makedirs(export_dir, exist_ok=True)

    # Copy and clean geometries
    patches = patches_gdf.copy()
    patches['geometry'] = patches['geometry'].buffer(0)
    patches['cluster_id'] = None

    # Assign cluster IDs
    for i, cluster in enumerate(scored_clusters[:top_n], start=1):
        for node in cluster.nodes:
            patches.at[node, 'cluster_id'] = i

    # Export patch-level shapefile
    patch_cols = ['geometry', 'cluster_id', 'total_core', 'density_cost', 'suppression_score']
    patches_top = patches.dropna(subset=['cluster_id'])[patch_cols].copy()
    patches_top['cluster_id'] = patches_top['cluster_id'].astype(int)
    shp_patches = os.path.join(export_dir, f"{shapefile_name}_patches.shp")
    patches_top.to_file(shp_patches, driver="ESRI Shapefile")

    # Export multipolygon shapefile of strongholds
    stronghold_rows = [c.to_geodataframe_row(i+1, 'stronghold') for i, c in enumerate(scored_clusters[:top_n])]
    gdf_strong = gpd.GeoDataFrame(stronghold_rows, crs=patches.crs, geometry='geometry')
    shp_multi = os.path.join(export_dir, f"{shapefile_name}_multipolygon.shp")
    gdf_strong.to_file(shp_multi, driver="ESRI Shapefile")

    # Build a combined buffer+core shapefile if desired
    buffer_polys = [c.buffer_ring for c in scored_clusters[:top_n]]
    union_core = unary_union([row['geometry'] for row in stronghold_rows])
    union_buffer = unary_union(buffer_polys)
    buffer_with_holes = union_buffer.difference(union_core)
    buffer_row = {
        'cluster_id': 0,
        'geom_type': 'buffer',
        'geometry': buffer_with_holes
    }
    gdf_combined = gpd.GeoDataFrame(stronghold_rows + [buffer_row], crs=patches.crs, geometry='geometry')
    shp_combined = os.path.join(export_dir, f"{shapefile_name}_combined.shp")
    gdf_combined.to_file(shp_combined, driver="ESRI Shapefile")

    return patches_top

# ------------------------------------------------------------
# Export cluster summary to CSV
# ------------------------------------------------------------
def export_cluster_summary_csv(
    scored_clusters,
    export_dir,
    top_n,
    PRIORITIES,
    filename="stronghold_summary.csv",
    scenario_name=None,
    strategy_name=None,
    master_csv_path=None
):
    """
    Write a per-scenario CSV of cluster summaries (top_n clusters) and append
    to a master CSV with consistent headers.
    Columns: cluster_id, scenario, strategy,
    total_area_ha, total_core_ha,
    grey_cost, buffer_cost, final_score, feasibility_score
    """
    # Build rows for the top N clusters
    rows = []
    for i, cluster in enumerate(scored_clusters[:top_n]):
        # Determine strategy weights (for naming consistency only)
        weights = next((s['weights'] for s in PRIORITIES if s['name']==strategy_name), {})

        rows.append({
            'cluster_id': f"{scenario_name}_{strategy_name}_{i}",
            'scenario': scenario_name,
            'strategy': strategy_name,
            'total_area_ha': round(cluster.total_area, 2),
            'total_core_ha': round(cluster.total_core, 2),
            # Use density_cost as the raw grey-squirrel cost metric
            'grey_cost': round(cluster.density_cost, 2),
            'buffer_cost': round(cluster.buffer_cost, 2),
            'final_score': round(cluster.final_score, 4),
            'feasibility_score': round(cluster.feasibility_score or 0, 4)
        })

    # Create DataFrame with explicit column ordering
    cols = [
        'cluster_id','scenario','strategy',
        'total_area_ha','total_core_ha',
        'grey_cost','buffer_cost',
        'final_score','feasibility_score'
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Write per-scenario CSV
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, filename)
    df.to_csv(path, index=False)

    # Append to master CSV, ensuring headers only once
    if master_csv_path:
        write_mode = 'a' if os.path.exists(master_csv_path) else 'w'
        header = not os.path.exists(master_csv_path)
        df.to_csv(master_csv_path, mode=write_mode, header=header, index=False, columns=cols)

    return df

# ------------------------------------------------------------
# (Other utilities unchanged)
# ------------------------------------------------------------

def read_raster_as_array(raster_path, mask_geom=None):
    with rasterio.open(raster_path) as src:
        if mask_geom:
            out_image, _ = rasterio.mask.mask(src, mask_geom, crop=True)
        else:
            out_image = src.read(1)
    data = out_image[0].astype('float32')
    data[data==9999] = np.nan
    return data

FOREST_CODES = [311, 312, 313]

def calculate_forest_cover_percentage(cluster, raster_path):
    raster = read_raster_as_array(raster_path, [cluster.combined_geom])
    total = np.count_nonzero(~np.isnan(raster))
    forest_pixels = sum(np.count_nonzero(raster==code) for code in FOREST_CODES)
    return (forest_pixels/total*100) if total>0 else 0

def filter_clusters_by_forest_cover(clusters, raster_path, threshold=20):
    filtered = []
    for c in clusters:
        pct = calculate_forest_cover_percentage(c, raster_path)
        if pct>=threshold:
            filtered.append(c)
        else:
            print(f"[INFO] Excluding cluster {c} with {pct:.1f}% forest cover")
    return filtered
# ------------------------------------------------------------
# Plotting:
# ------------------------------------------------------------

def plot_all_strategies_overlay(scenario_name, strategies, output_dir="../outputs", cumbria_shapefile="cumbria.shp"):
    """
    Plots all strongholds from different strategies on a single map for a given scenario.
    Colour-coded by strategy. No text labels. Saves a combined overlay plot.
    """

    # Define colour scheme for strategies
    strategy_colors = {
        "total_core_priority": "blue",
        "maximise_value": "green",
        "balanced": "red"
    }

    # Load base layer
    cumbria = gpd.read_file(cumbria_shapefile)

    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot base boundary
    cumbria.boundary.plot(ax=ax, color='black', linewidth=1)

    # Loop through strategies and overlay each
    for strategy in strategies:
        shapefile_path = os.path.join(
            output_dir,
            f"{scenario_name}_{strategy}_buffer",
            f"top5_strongholds_{scenario_name}_{strategy}_multipolygon.shp"
        )

        if os.path.exists(shapefile_path):
            gdf = gpd.read_file(shapefile_path)
            gdf = gdf[gdf["geom_type"] == "stronghold"]
            color = strategy_colors.get(strategy, 'grey')

            gdf.plot(ax=ax, color=color, alpha=0.4, label=strategy.replace("_", " ").title())
        else:
            print(f"[WARN] Missing shapefile: {shapefile_path}")

    # Finalise plot
    ax.set_title(f"Stronghold Strategies — {scenario_name}", fontsize=14)
    ax.axis('off')
    ax.legend(title="Strategy", loc='upper right')

    # Save combined overlay map
    save_path = os.path.join(output_dir, f"strongholds_overlay_{scenario_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[PLOT SAVED] {save_path}")
    
def filter_best_clusters_per_patch(scored_clusters):
    """
    Filters scored clusters so that each patch is only assigned to the highest-scoring cluster it appears in.
    Returns a list of filtered clusters with no overlapping patches.
    
    Parameters:
        scored_clusters (list): List of cluster objects, each with `nodes` (set of patch IDs) and `final_score`.
    
    Returns:
        filtered_clusters (list): Non-overlapping, high-scoring clusters.
    """
    
    # Dictionary to track the best cluster each patch appears in
    best_cluster_per_patch = {}
    
    # Step 1: Assign each patch to the best-scoring cluster it belongs to
    for cluster in scored_clusters:
        for patch_id in cluster.nodes:
            current_best = best_cluster_per_patch.get(patch_id)
            if current_best is None or cluster.final_score > current_best.final_score:
                best_cluster_per_patch[patch_id] = cluster
    
    # Step 2: Collect only unique clusters that "own" at least one patch
    filtered_clusters = []
    seen_clusters = set()
    
    for cluster in best_cluster_per_patch.values():
        cluster_key = id(cluster)
        if cluster_key not in seen_clusters:
            filtered_clusters.append(cluster)
            seen_clusters.add(cluster_key)

    return filtered_clusters

def plot_cluster_priority_gradient(scored_clusters, base_shapefile_path, output_path):
    """
    Plots all scored clusters coloured by priority rank.
    Top 15% = dark blue, next 15% = lighter blue, etc.
    """
    # Load base map
    base = gpd.read_file(base_shapefile_path)

    # Sort clusters by score (highest first)
    scored_clusters = sorted(scored_clusters, key=lambda c: c.final_score, reverse=True)

    # Convert clusters to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        [c.to_geodataframe_row(i, geom_type='stronghold') for i, c in enumerate(scored_clusters)],
        crs=base.crs
    )

    # Calculate rank percentiles
    gdf['rank'] = range(1, len(gdf) + 1)
    gdf['percentile'] = gdf['rank'] / len(gdf)

    # Assign category
    def assign_category(p):
        if p <= 0.15:
            return 'top_15'
        elif p <= 0.30:
            return 'top_30'
        elif p <= 0.45:
            return 'top_45'
        elif p <= 0.60:
            return 'top_60'
        else:
            return 'other'

    gdf['category'] = gdf['percentile'].apply(assign_category)

    # Colour map by category
    color_map = {
        'top_5' : 'red',
        'top_15': cm.Blues(0.8),
        'top_30': cm.Blues(0.6),
        'top_45': cm.Blues(0.4),
        'top_60': cm.Blues(0.2),
        'other': '#cccccc'
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    base.boundary.plot(ax=ax, color='black', linewidth=1)

    # Plot in order: lowest priority first
    for cat in ['other', 'top_60', 'top_45', 'top_30', 'top_15']:
        gdf[gdf['category'] == cat].plot(
            ax=ax,
            color=color_map[cat],
            label=cat.replace('_', ' ').title()
            # No alpha = fully opaque
        )

    ax.set_title("Cluster Priority Gradient", fontsize=14)
    ax.axis('off')
    ax.legend(title="Cluster Tier", loc='upper right')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

def score_managed_strongholds(
    managed_shp: str,
    scored_patches: gpd.GeoDataFrame,
    scored_clusters: list,
    weights: dict
) -> pd.DataFrame:
    """
    Load managed strongholds shapefile and compute an MCDA final_score
    and percentile for each, using the same stats as scored_clusters.
    Returns a DataFrame with columns:
      ['stronghold_id','total_area_ha','final_score','percentile']
    """
    # 1) Read & reproject
    managed = gpd.read_file(managed_shp).to_crs(scored_patches.crs)
    managed['stronghold_id'] = managed.index

    # 2) Ensure suppression_score on patches
    patches = scored_patches.copy()
    if 'suppression_score' not in patches.columns:
        patches['suppression_score'] = patches['density_cost']

    # 3) Compute raw cluster metrics for each managed stronghold
    rows = []
    for _, row in managed.iterrows():
        geom = row.geometry
        sel = patches[patches.geometry.intersects(geom)]
        total_area = sel.geometry.area.sum() / 10000
        total_core = sel['total_core'].sum()
        suppression_score = sel['suppression_score'].sum()
        feasibility_score = sel['feasibility_score'].mean() if len(sel) else 0
        rows.append({
            'stronghold_id': row.stronghold_id,
            'total_area_ha': total_area,
            'total_core': total_core,
            'suppression_score': suppression_score,
            'feasibility_score': feasibility_score
        })
    df = pd.DataFrame(rows)

    # 4) Build the stats (means, stds, zmins, zmaxs) from your algorithm's clusters
    cluster_data = pd.DataFrame([
        {
            'total_core': c.total_core,
            'feasibility_score': c.feasibility_score,
            # matches how score_clusters did it:
            'suppression_score': sum(scored_patches.loc[n, 'density_cost'] for n in c.nodes)
        }
        for c in scored_clusters
    ])
    means, stds, z_mins, z_maxs = compute_z_stats(cluster_data, weights.keys())

    # 5) Compute final_score for each managed stronghold
    def _score(rec):
        attrs = {
            'total_core': rec.total_core,
            'feasibility_score': rec.feasibility_score,
            'suppression_score': rec.suppression_score
        }
        return mcda_score(attrs, weights, means, stds, z_mins, z_maxs)

    df['final_score'] = df.apply(_score, axis=1)

    # 6) Percentile vs your scored_clusters
    all_scores = np.array([c.final_score for c in scored_clusters])
    df['percentile'] = df['final_score'].apply(
        lambda x: float((all_scores <= x).sum()) / len(all_scores)
    )

    return df

'''Investigating managed strongholds'''
def clean_managed_strongholds(
    managed_shp: str,
    sitenames: list = None,
    target_crs=None
) -> gpd.GeoDataFrame:
    if sitenames is None:
        sitenames = ["Whinfell","Thirlmere","Whinlatter","Greystoke"]
    gdf = gpd.read_file(managed_shp)
    gdf = gdf[gdf["sitename"].isin(sitenames)].copy()
    gdf = gdf[gdf.geometry.notnull()]
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid].copy()
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


def score_managed_strongholds(
    managed_gdf: gpd.GeoDataFrame,
    scored_patches: gpd.GeoDataFrame,
    scored_clusters: list,
    weights: dict
) -> pd.DataFrame:
    # ensure suppression_score
    patches = scored_patches.copy()
    if 'suppression_score' not in patches.columns:
        patches['suppression_score'] = patches['density_cost']

    rows = []
    for _, row in managed_gdf.iterrows():
        geom = row.geometry
        sel = patches[patches.geometry.intersects(geom)]
        rows.append({
            'sitename': row['sitename'],
            'total_area_ha': sel.geometry.area.sum()/10000,
            'total_core_ha': sel['total_core'].sum(),
            'suppression_score': sel['suppression_score'].sum(),
            'feasibility_score': sel['feasibility_score'].mean() if len(sel) else 0
        })
    df = pd.DataFrame(rows)

    cluster_df = pd.DataFrame([
        {
            'total_core': c.total_core,
            'feasibility_score': c.feasibility_score,
            'suppression_score': sum(
                scored_patches.loc[n,'density_cost'] for n in c.nodes
            )
        } for c in scored_clusters
    ])
    means, stds, zmins, zmaxs = compute_z_stats(cluster_df, weights.keys())
    df['final_score'] = df.apply(
        lambda r: mcda_score({
            'total_core': r['total_core_ha'],
            'feasibility_score': r['feasibility_score'],
            'suppression_score': r['suppression_score']
        }, weights, means, stds, zmins, zmaxs),
        axis=1
    )

    all_scores = np.array([c.final_score for c in scored_clusters])
    df['percentile'] = df['final_score'].apply(
        lambda x: float((all_scores <= x).sum()) / len(all_scores)
    )
    return df


def plot_master_summary_with_managed(
    summary_csv: str,
    managed_scores_csv: str,
    scenario_name: str,
    output_path: str = None,
    area_floor: float = 450,
    area_limit: float = 1100
):
    """
    Plot percentile vs final_score for one scenario:
      • algorithm clusters (×)
      • managed strongholds (●), only those within [area_floor,area_limit],
        labeled by sitename.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1) Load and filter algorithm clusters to the area bounds
    df_alg = pd.read_csv(summary_csv)
    df_alg = df_alg[
        (df_alg['total_area_ha'] >= 0) &
        (df_alg['total_area_ha'] <= 1200)
    ].copy()
    df_alg['percentile'] = df_alg['final_score'].rank(pct=True)

    # 2) Load and filter the four managed strongholds to the same bounds
    df_exist = pd.read_csv(managed_scores_csv)
    df_exist = df_exist[
        (df_exist['total_area_ha'] >= 0) &
        (df_exist['total_area_ha'] <= 1200)
    ].copy()

    # 3) Recompute their percentiles against the filtered algorithm distribution
    all_scores = df_alg['final_score']
    df_exist['percentile'] = df_exist['final_score'].apply(
        lambda x: float((all_scores <= x).sum()) / len(all_scores)
    )

    # 4) Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df_alg['percentile'], df_alg['final_score'],
        marker='x', s=50, color='black', alpha=0.6,
        label='Algorithm clusters'
    )
    ax.scatter(
        df_exist['percentile'], df_exist['final_score'],
        marker='o', s=200, color='green', edgecolor='black',
        label='Managed strongholds'
    )

    # 5) Label each green point by sitename
    for _, row in df_exist.iterrows():
        ax.annotate(
            row['sitename'],
            xy=(row['percentile'], row['final_score']),
            xytext=(-5, 5),
            textcoords='offset points',
            ha='left',
            va='bottom',
            fontsize=10,
            color='darkgreen'
        )

    ax.set_xlabel('Percentile among clusters\n(size between ' +
                  f'100-1200 ha)')
    ax.set_ylabel('MCDA Final Score')
    ax.set_title(f"{scenario_name} — Score vs Percentile")
    ax.legend(loc='best')

    # 6) Save
    if output_path is None:
        output_path = os.path.join("..", "outputs", f"{scenario_name}_score_vs_percentile.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_three_panel_migration(
    scenarios, strategy_weights, patches_gdf, managed_gdf,
    area_floor, area_limit, max_distance, buffer_distance,
    clean_raw_density, build_rtree_index,
    build_connectivity_graph, collect_clusters, score_clusters,
    filter_top_n_unique_clusters,
    output_path
):
    """
    Same as before, but uses filter_top_n_unique_clusters() to ensure
    both the 15%-set and the 5-cluster set are non-overlapping.
    """
    panel_order = ["mean_42", "mean_83", "mean_125"]
    s_labels    = ["S = 42", "S = 83", "S = 125"]
    prev_top5   = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, scen, slog in zip(axes, panel_order, s_labels):
        # 1) Load & clean raster
        with rasterio.open(scenarios[scen]) as src:
            grey      = clean_raw_density(src)
            extent    = plotting_extent(src)
            transform = src.transform

        # 2) Build & score clusters
        idx      = build_rtree_index(patches_gdf)
        G        = build_connectivity_graph(patches_gdf, idx, max_distance)
        clusters = collect_clusters(
            G, area_limit, area_floor,
            grey, transform,
            np.zeros_like(grey), buffer_distance
        )
        scored = score_clusters(clusters, strategy_weights, G)

        # 3) Pick top 15% *unique* and top 5 *unique*
        n15 = max(1, int(len(scored) * 0.15))
        top15 = filter_top_n_unique_clusters(scored, n=n15,
                     area_floor=area_floor, area_limit=area_limit)
        top5  = filter_top_n_unique_clusters(scored, n=5,
                     area_floor=area_floor, area_limit=area_limit)

        # 4) Convert to GeoDataFrames
        g15 = gpd.GeoDataFrame(
            [c.to_geodataframe_row(i+1,'stronghold') for i,c in enumerate(top15)],
            crs=patches_gdf.crs
        )
        g5p = (None if prev_top5 is None else gpd.GeoDataFrame(
            [c.to_geodataframe_row(i+1,'stronghold') for i,c in enumerate(prev_top5)],
            crs=patches_gdf.crs
        ))

        # 5) Plot layers
        ax.imshow(grey, cmap='viridis_r', vmin=0, vmax=12,
                  extent=extent, zorder=1)

        if g5p is not None:
            g5p.plot(ax=ax, facecolor='none', edgecolor='black',
                     linewidth=2, alpha=0.2, zorder=2)

        g15.plot(ax=ax, facecolor='none', edgecolor='#e66101',
                 linewidth=2, alpha=1.0, zorder=3)

        managed_gdf.plot(ax=ax, facecolor='none', edgecolor='blue',
                         linewidth=2, zorder=4)

        if ax is axes[0]:
            sb = ScaleBar(dx=src.res[0], units='m', length_fraction=0.2,
                          location='lower left')
            ax.add_artist(sb)

        ax.set_title(slog, fontsize=14, pad=8)
        ax.set_axis_off()

        prev_top5 = top5

    # Shared legend & title
    handles = [
        plt.Line2D([], [], color='black', lw=2, alpha=0.2, label='prev top 5'),
        plt.Line2D([], [], color='#e66101', lw=2, label='current top 15%'),
        plt.Line2D([], [], color='blue', lw=2, label='managed strongholds')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               frameon=True, framealpha=0.8, fontsize=12)

    fig.suptitle("Migration of Top Clusters Across Mean Scenarios",
                 fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    
def plot_three_panel_migration_smoothed(
    scenarios,
    strategy_weights,
    patches_gdf,
    managed_gdf,
    area_floor, area_limit,
    max_distance, buffer_distance,
    clean_raw_density,
    build_rtree_index,
    build_connectivity_graph,
    collect_clusters,
    score_clusters,
    filter_top_n_unique_clusters,
    output_path
):
    panel_order = ["mean_42", "mean_83", "mean_125"]
    s_labels    = ["S = 42", "S = 83", "S = 125"]
    prev_top5   = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im = None  # for colorbar

    for ax, scen, slog in zip(axes, panel_order, s_labels):
        # 1) load raw grey raster and print stats
        with rasterio.open(scenarios[scen]) as src:
            grey_raw    = clean_raw_density(src)
            print(f"{scen} stats → min: {np.nanmin(grey_raw):.2f}, "
                  f"mean: {np.nanmean(grey_raw):.2f}, "
                  f"max: {np.nanmax(grey_raw):.2f}")
            grey_smooth = gaussian_filter(grey_raw.astype(float), sigma=3)
            extent      = plotting_extent(src)
            transform   = src.transform

        # 2) build & score clusters
        idx      = build_rtree_index(patches_gdf)
        G        = build_connectivity_graph(patches_gdf, idx, max_distance)
        clusters = collect_clusters(
            G, area_limit, area_floor,
            grey_raw, transform,
            np.zeros_like(grey_raw), buffer_distance
        )
        scored = score_clusters(clusters, strategy_weights, G)

        # 3) pick unique top‐15% & top‐5
        n15  = max(1, int(len(scored) * 0.15))
        top15 = filter_top_n_unique_clusters(scored, n=n15,
                                             area_floor=area_floor,
                                             area_limit=area_limit)
        top5  = filter_top_n_unique_clusters(scored, n=5,
                                             area_floor=area_floor,
                                             area_limit=area_limit)

        # 4) to GeoDataFrames
        g15 = gpd.GeoDataFrame(
            [c.to_geodataframe_row(i+1,'stronghold') for i,c in enumerate(top15)],
            crs=patches_gdf.crs
        )
        g5p = None
        if prev_top5:
            g5p = gpd.GeoDataFrame(
                [c.to_geodataframe_row(i+1,'stronghold') for i,c in enumerate(prev_top5)],
                crs=patches_gdf.crs
            )

        # 5) Plot smoothed background
        im = ax.imshow(
            grey_smooth,
            cmap='viridis_r',
            vmin=np.nanpercentile(grey_smooth, 5),
            vmax=np.nanpercentile(grey_smooth, 95),
            extent=extent,
            zorder=1
        )

        # 6) overlay previous top‐5 (faded)
        if g5p is not None:
            g5p.plot(ax=ax, facecolor='none', edgecolor='#666',
                     linewidth=2, alpha=0.2, zorder=2)

        # 7) overlay current top‐15%
        g15.plot(ax=ax, facecolor='none', edgecolor='#FF7F0E',
                 linewidth=3, alpha=1.0, zorder=3)

        # 8) overlay managed strongholds
        managed_gdf.plot(ax=ax, facecolor='none', edgecolor='#1F77B4',
                         linewidth=2, zorder=4)

        # 9) scalebar on first panel
        if ax is axes[0]:
            sb = ScaleBar(dx=src.res[0], units='m',
                          length_fraction=0.15, location='lower left')
            ax.add_artist(sb)

        # annotate title with scenario code
        ax.set_title(f"{slog}\n({scen})", fontsize=14, pad=8)
        ax.set_axis_off()
        prev_top5 = top5

    # shared colorbar for the background
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Smoothed grey density", size=12)

    # shared legend
    handles = [
        plt.Line2D([], [], color='#666',  lw=2, alpha=0.2, label='prev top 5'),
        plt.Line2D([], [], color='#FF7F0E',lw=3, label='current top 15%'),
        plt.Line2D([], [], color='#1F77B4',lw=2, label='managed strongholds')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               frameon=True, framealpha=0.8, fontsize=12)

    fig.suptitle("Migration of Top Clusters Across Mean Scenarios",
                 fontsize=18, y=1.02)
    plt.tight_layout(rect=[0,0.02,1,0.95])
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)