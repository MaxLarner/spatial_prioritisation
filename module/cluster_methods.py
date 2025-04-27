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
from rtree import index

# ------------------------------------------------------------
# Cluster Class: Each cluster is one candidate stronghold
# ------------------------------------------------------------

class Cluster:
    def __init__(self, nodes, G, grey_density_raster, transform, buffer_distance, feas_raster):
        # A cluster object represents a cluster of connected habitat patches

        self.nodes = list(nodes)

        # Combine geometries of all nodes in the cluster into a single multipolygon
        self.geometry = unary_union([G.nodes[n]['geometry'] for n in self.nodes])

        # Aggregate key attributes from the patches
        self.total_area = sum(G.nodes[n]['area'] for n in self.nodes)                # Area in hectares
        self.total_core = sum(G.nodes[n]['total_core'] for n in self.nodes)         # Core red squirrel habitat area
        self.density_cost = sum(G.nodes[n]['density_cost'] for n in self.nodes)     # Raw cost (mean grey × area)
        self.suppression_score = sum(G.nodes[n]['suppression_score'] for n in self.nodes)  # Raw suppression cost

        # Generate the surrounding buffer ring used for isolation cost estimation
        self.buffer_ring = self.geometry.buffer(buffer_distance).difference(self.geometry)

        # Union of core geometry + buffer ring for later suppression analysis
        self.combined_geom = self.geometry.union(self.buffer_ring)

        # Calculate the average grey squirrel density within the buffer ring (used in scoring)
        self.buffer_cost = self.calculate_buffer_cost(grey_density_raster, transform)

        # Placeholder for feasibility (mean of patch-level feasibility scores)
        self.feasibility_score = None

        # Placeholder for final MCDA score
        self.final_score = None

    def calculate_buffer_cost(self, grey_density_raster, transform):
        """
        Calculates the mean management cost within the buffer ring surrounding the cluster
        Returns 0 if geometry is invalid or no meaningful values
        """
        try:
            if self.buffer_ring.is_empty or not self.buffer_ring.is_valid:
                return 0

            # Rasterise buffer ring geometry and extract intersecting pixel values
            mask = rasterio.features.rasterize(
                [(self.buffer_ring, 1)], out_shape=grey_density_raster.shape, transform=transform, fill=0
            ) == 1
            values = grey_density_raster[mask]
            values = values[(~np.isnan(values)) & (values < 9999)]
            return values.mean() if len(values) > 0 else 0

        except Exception as e:
            print(f"[ERROR] Buffer cost calc failed: {e}")
            return 0

    def to_geodataframe_row(self, cluster_id, geom_type='stronghold'):
        """
        Constructs a row of GeoDataFrame-ready attributes for exporting cluster geometry.
        """
        row = self.summary_dict(cluster_id)
        row['geom_type'] = geom_type

        if geom_type == 'stronghold':
            row['geometry'] = self.geometry
        elif geom_type == 'buffer':
            row['geometry'] = self.buffer_ring
        elif geom_type == 'combined':
            row['geometry'] = self.combined_geom

        return row

    def summary_dict(self, cluster_id):
        """
        Creates the summary dictionary of core attributes for export to shapefile.
        """
        return {
            "cluster_id": cluster_id,
            "total_area_ha": round(self.total_area, 2),
            "total_core_ha": round(self.total_core, 2),
            "density_cost": round(self.density_cost, 2),
            "suppression_score": round(self.suppression_score, 2),
            "buffer_cost": round(self.buffer_cost, 2),
            "final_score": round(self.final_score, 4) if self.final_score is not None else None,
            "feasibility_score": round(self.feasibility_score, 4) if self.feasibility_score is not None else None
        }

# ------------------------------------------------------------
# Graph Construction and Spatial Data Structures
# ------------------------------------------------------------

def build_rtree_index(scored_patches):
    # Create a spatial index of patches using an R-tree for fast proximity lookups
    rtree = index.Index()
    for idx, patch in scored_patches.iterrows():
        rtree.insert(idx, patch.geometry.bounds)
    return rtree


def build_connectivity_graph(patches_gdf, rtree, max_distance):
    """
    Constructs an undirected graph of patches within `max_distance` of each other.
    Each node carries 'density_cost', 'suppression_score', 'priority_score', and 'feasibility_score'.
    """
    G = nx.Graph()

    # Add all patches as nodes
    for idx, row in patches_gdf.iterrows():
        G.add_node(
            idx,
            area=row['geometry'].area / 10000,
            total_core=row['total_core'],
            density_cost=row['density_cost'],
            suppression_score=row.get('suppression_score', row['density_cost']),
            priority_score=row['priority_score'],
            geometry=row['geometry'],
            feasibility_score=row['feasibility_score']
        )

    # Connect nodes whose geometries are within max_distance
    for idx_a, patch_a in patches_gdf.iterrows():
        geom_a = patch_a.geometry
        candidates = list(rtree.intersection(geom_a.buffer(max_distance).bounds))
        for idx_b in candidates:
            if idx_b == idx_a:
                continue
            geom_b = patches_gdf.loc[idx_b].geometry
            if geom_a.distance(geom_b) <= max_distance:
                G.add_edge(idx_a, idx_b)

    return G

# ------------------------------------------------------------
# Cluster Growing/Collection Algorithm
# ------------------------------------------------------------

def grow_cluster_bfs(
    G, start_node, area_limit, area_floor,
    grey_density_raster, transform, feas_raster, buffer_distance
):
    '''
    Greedy BFS cluster growth from each starting patch, respecting area limits.
    '''
    cluster_nodes = {start_node}
    total_area = G.nodes[start_node]['area']

    # Breadth-first traversal excluding start node
    reachable = [n for n in nx.bfs_tree(G, start_node) if n != start_node]
    # Prioritise nodes by descending priority_score
    reachable.sort(key=lambda n: G.nodes[n]['priority_score'], reverse=True)

    clusters = []
    for node in reachable:
        if node in cluster_nodes:
            continue
        patch_area = G.nodes[node]['area']
        if total_area + patch_area > area_limit:
            continue
        cluster_nodes.add(node)
        total_area += patch_area
        # Snapshot clusters once minimum area reached
        if total_area >= area_floor:
            clusters.append(
                Cluster(set(cluster_nodes), G,
                        grey_density_raster, transform,
                        buffer_distance, feas_raster)
            )
    return clusters


def collect_clusters(
    G, area_limit, area_floor,
    grey_density_raster, transform, feas_raster, buffer_distance
):
    '''
    Generate all unique clusters by BFS from every patch, deduplicating by membership.
    '''
    clusters = []
    seen = set()
    for node in G.nodes:
        for cluster in grow_cluster_bfs(
            G, node, area_limit, area_floor,
            grey_density_raster, transform, feas_raster, buffer_distance
        ):
            key = tuple(sorted(cluster.nodes))
            if key not in seen:
                clusters.append(cluster)
                seen.add(key)
    return clusters


def filter_top_n_unique_clusters(
    scored_clusters,
    n=5,
    area_floor=450,
    area_limit=1100
):
    """
    Return up to `n` non-overlapping clusters, preferring higher final_score clusters.
    """
    # Enforce area bounds
    valid = [c for c in scored_clusters if area_floor <= c.total_area <= area_limit]
    selected = []
    for c in valid:
        if any(c.geometry.intersects(sel.geometry) for sel in selected):
            continue
        selected.append(c)
        if len(selected) >= n:
            break
    return selected
