"""
mcda_methods.py

This module contains all core geographical analysis logic for stronghold prioritisation,
including patch scoring with MCDA, graph construction, bfs cluster formation,
and cluster  scoring. These methods form the basis of the analytical approach
used in my dissertation.
"""
# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
import networkx as nx
from shapely.ops import unary_union
from shapely.errors import TopologicalError
import rasterio.features
import pandas as pd
import geopandas as gpd

# ------------------------------------------------------------
# Z-Score Utilities — used for both patches and clusters
# ------------------------------------------------------------

def compute_z_stats(df, criteria):
    # Calculate means, standard deviations, and z-score min/max for each criterion
    means = {c: df[c].mean() for c in criteria}
    stds = {c: df[c].std() for c in criteria}
    z_mins, z_maxs = {}, {}

    for c in criteria:
        z_scores = (df[c] - means[c]) / stds[c] if stds[c] != 0 else df[c] * 0
        z_mins[c], z_maxs[c] = z_scores.min(), z_scores.max()

    return means, stds, z_mins, z_maxs

# ------------------------------------------------------------
# MCDA Scoring Function (used for both patches and clusters)
# ------------------------------------------------------------

def mcda_score(attributes, weights, means, stds, z_mins, z_maxs):

    score = 0
    for crit in weights:
        # Get raw attribute value
        raw = attributes.get(crit, 0)

        # Z-score standardisation
        z = 0 if stds[crit] == 0 else (raw - means[crit]) / stds[crit]

        # Min-max scale z-score to [0, 1]
        if z_maxs[crit] - z_mins[crit] == 0:
            z_scaled = 0.5
        else:
            z_scaled = (z - z_mins[crit]) / (z_maxs[crit] - z_mins[crit])
            z_scaled = max(0, min(1, z_scaled))

        # Rescale to [0.1, 1.0]
        z_scaled = 0.1 + z_scaled * 0.9

        # For the suppression-related criteria, invert score so lower grey density = higher suitability
        if crit == 'suppression_score':
            z_scaled = 1.1 - z_scaled

        score += weights[crit] * z_scaled

    return score / sum(weights.values())

# ------------------------------------------------------------
# Score Individual Patch Attributes
# ------------------------------------------------------------

def score_patch_attributes(patch, grey_density_raster, feas_raster, transform):
    geom = patch.geometry
    area_ha = geom.area / 10000

    # Buffer inwards to estimate interior (core) habitat
    try:
        core = geom.buffer(-100).buffer(0)
        total_core = max(core.area / 10000, 0) if core.is_valid and not core.is_empty else 0
    except TopologicalError:
        # Handle cases where buffer leads to no interior
        total_core = 0

    # Rasterise patch geometry and extract intersecting pixels
    mask = rasterio.features.rasterize(
        [(geom, 1)], out_shape=grey_density_raster.shape, transform=transform, fill=0
    ) == 1

    # Extract grey squirrel density values and compute cost
    grey_vals = grey_density_raster[mask]
    grey_vals = grey_vals[(~np.isnan(grey_vals)) & (grey_vals < 9999)]
    density_cost = grey_vals.mean() * area_ha if len(grey_vals) > 0 else 0

    # Extract feasibility score from raster
    feas_vals = feas_raster[mask]
    feas_vals = feas_vals[~np.isnan(feas_vals)]
    feas_score = feas_vals.mean() if len(feas_vals) > 0 else 0

    return {
        'geometry': geom,
        'total_core': total_core,                # Interior red squirrel habitat
        'density_cost': density_cost,            # Raw cost (mean grey × area)
        'suppression_score': density_cost,       # Duplicate for inversion in MCDA
        'feasibility_score': feas_score,         # Normalised score of practical feasibility
    }

# ------------------------------------------------------------
# Apply MCDA Scoring to All Patches
# ------------------------------------------------------------

def score_all_patches(patches_gdf, grey_density_raster, feas_raster, weights, transform):
    """
    Calculates MCDA priority scores for each habitat patch.
    Also adds a 'suppression_score' column (needed downstream in build_connectivity_graph).
    """
    # Scoring criteria now directly use suppression_score
    scoring_criteria = list(weights.keys())

    # Compute raw attribute scores per patch
    patch_data = [
        score_patch_attributes(row, grey_density_raster, feas_raster, transform)
        for _, row in patches_gdf.iterrows()
    ]
    df = pd.DataFrame(patch_data)

    # Compute Z-score statistics
    means, stds, z_mins, z_maxs = compute_z_stats(df, scoring_criteria)

    # Calculate MCDA score using original weights
    scores = [
        mcda_score(attr, weights, means, stds, z_mins, z_maxs)
        for attr in patch_data
    ]
    df['priority_score'] = scores

    # Attach identifiers & geometry
    df['patch_id'] = patches_gdf.index
    df['geometry'] = patches_gdf['geometry']

    return gpd.GeoDataFrame(df, crs=patches_gdf.crs)

# ------------------------------------------------------------
# Apply MCDA Scoring to Cluster Objects
# ------------------------------------------------------------

def score_clusters(clusters, weights, G):
    if not clusters:
        print("[WARN] No clusters were generated — skipping scoring.")
        return []

    # Compute mean feasibility score for each cluster from constituent patches
    for cluster in clusters:
        patch_scores = [G.nodes[n].get('feasibility_score', 0) for n in cluster.nodes]
        cluster.feasibility_score = (
            sum(patch_scores) / len(patch_scores)
            if patch_scores else 0
        )

    # Build dataframe of cluster-level attributes for scoring
    cluster_data = pd.DataFrame([
        {
            'total_core': c.total_core,
            'feasibility_score': c.feasibility_score,
            'suppression_score': sum(
                G.nodes[n]['density_cost'] for n in c.nodes
            )  # raw pre-MCDA value
        }
        for c in clusters
    ])

    # Compute normalised z-stats
    means, stds, z_mins, z_maxs = compute_z_stats(cluster_data, weights.keys())

    # Apply MCDA to each cluster
    for cluster, attr in zip(clusters, cluster_data.to_dict('records')):
        cluster.final_score = mcda_score(attr, weights, means, stds, z_mins, z_maxs)

    # Return clusters sorted descending by final_score
    return sorted(clusters, key=lambda c: c.final_score, reverse=True)
