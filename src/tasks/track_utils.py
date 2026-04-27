"""
Utility functions for track-level metrics and reporting.
"""
import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def load_track_names(track_names_file, num_tracks):
    """Load track names from file or generate default names.
    
    Args:
        track_names_file: Path to text file with one track name per line, or None
        num_tracks: Total number of tracks
        
    Returns:
        List of track names
    """
    if track_names_file and os.path.exists(track_names_file):
        try:
            with open(track_names_file, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
            if len(names) == num_tracks:
                log.info(f"Loaded {len(names)} track names from {track_names_file}")
                return names
            else:
                log.warning(
                    f"Track names file has {len(names)} entries but expected {num_tracks}. "
                    f"Using default names."
                )
        except Exception as e:
            log.warning(f"Failed to load track names from {track_names_file}: {e}. Using default names.")
    
    # Generate default names
    return [f"track_{i}" for i in range(num_tracks)]


def log_top_tracks(pcc_per_track, track_names, top_k=3, step=None, prefix=""):
    """Log top K tracks by PCC value.
    
    Args:
        pcc_per_track: Tensor of PCC values per track (num_tracks,)
        track_names: List of track names
        top_k: Number of top tracks to report
        step: Training step number (optional)
        prefix: Prefix for log message (e.g., "Train", "Val")
    
    Returns:
        List of (track_name, pcc_value) tuples for top K tracks
    """
    top_pcc_values, top_indices = torch.topk(pcc_per_track, k=min(top_k, len(pcc_per_track)), largest=True)
    
    top_tracks = []
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        pcc_val = top_pcc_values[i].item()
        track_name = track_names[idx] if idx < len(track_names) else f"track_{idx}"
        top_tracks.append((track_name, pcc_val))
    
    # Format log message
    info_strs = [f"{name}: {pcc:.4f}" for name, pcc in top_tracks]
    step_str = f"[Step {step}] " if step is not None else ""
    prefix_str = f"{prefix} " if prefix else ""
    log_msg = f"{step_str}{prefix_str}Top {len(top_tracks)} tracks: {', '.join(info_strs)}"
    log.info(log_msg)
    
    return top_tracks


def save_pcc_ranking(pcc_per_track, track_names, output_dir, epoch, prefix=""):
    """Save PCC ranking to CSV file.
    
    Args:
        pcc_per_track: Averaged PCC values per track (num_tracks,)
        track_names: List of track names
        output_dir: Directory to save the CSV file
        epoch: Current epoch number
        prefix: Prefix for filename (e.g., "val", "test")
    """
    import pandas as pd
    
    # Convert to numpy if needed
    if torch.is_tensor(pcc_per_track):
        pcc_array = pcc_per_track.cpu().numpy()
    else:
        pcc_array = pcc_per_track
    
    # Create ranking
    indices = torch.argsort(torch.tensor(pcc_array), descending=True).numpy()
    
    # Prepare data
    ranking_data = []
    for rank, idx in enumerate(indices, 1):
        track_name = track_names[idx] if idx < len(track_names) else f"track_{idx}"
        ranking_data.append({
            'rank': rank,
            'track_index': idx,
            'track_name': track_name,
            'pcc': float(pcc_array[idx])
        })
    
    # Save to CSV
    df = pd.DataFrame(ranking_data)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix_str = f"{prefix}_" if prefix else ""
    csv_path = output_path / f'{prefix_str}pcc_ranking_epoch{epoch}.csv'
    df.to_csv(csv_path, index=False)
    log.info(f"Saved PCC ranking to {csv_path}")
    
    return csv_path


def classify_track(track_name):
    """Classify track by category based on its name.
    
    Args:
        track_name: Track name string
        
    Returns:
        Category name: 'CAGE', 'ChIP-Histone', 'ChIP-TF', 'DNase/ATAC', or 'Other'
    """
    track_upper = track_name.upper()
    
    # CAGE tracks
    if 'CAGE' in track_upper:
        return 'CAGE'
    
    # DNase/ATAC tracks  
    if 'DNASE' in track_upper or 'DNS' in track_upper or 'ATAC' in track_upper:
        return 'DNase/ATAC'
    
    # Histone modification tracks (ChIP-seq with histone marks)
    histone_marks = ['H2AFZ', 'H3K4ME1', 'H3K4ME2', 'H3K4ME3', 
                     'H3K9AC', 'H3K9ME3', 'H3K27AC', 'H3K27ME3',
                     'H3K36ME3', 'H3K79ME2', 'H4K20ME1']
    for mark in histone_marks:
        if mark in track_upper:
            return 'ChIP-Histone'
    
    # Transcription Factor ChIP-seq (will catch remaining ChIP tracks)
    if 'CHIP' in track_upper or track_name.startswith('CHIP'):
        return 'ChIP-TF'
    
    return 'Other'


def compute_category_statistics(pcc_per_track, track_names):
    """Compute PCC statistics grouped by track categories.
    
    Args:
        pcc_per_track: Tensor of PCC values per track (num_tracks,)
        track_names: List of track names
        
    Returns:
        Dictionary with category statistics:
        {
            'CAGE': {'max': float, 'mean': float, 'count': int, 'tracks': list},
            'ChIP-Histone': {...},
            'ChIP-TF': {...},
            'DNase/ATAC': {...},
            'Other': {...},
            'All': {'mean': float, 'count': int}
        }
    """
    import numpy as np
    
    # Convert to numpy if needed
    if torch.is_tensor(pcc_per_track):
        pcc_array = pcc_per_track.cpu().numpy()
    else:
        pcc_array = pcc_per_track
    
    # Group tracks by category
    categories = {
        'CAGE': [],
        'ChIP-Histone': [],
        'ChIP-TF': [],
        'DNase/ATAC': [],
        'Other': []
    }
    
    for idx, pcc_val in enumerate(pcc_array):
        if not np.isfinite(pcc_val):
            continue
            
        track_name = track_names[idx] if idx < len(track_names) else f"track_{idx}"
        category = classify_track(track_name)
        categories[category].append({
            'name': track_name,
            'pcc': float(pcc_val),
            'index': idx
        })
    
    # Compute statistics for each category
    stats = {}
    for cat_name, tracks in categories.items():
        if len(tracks) > 0:
            pcc_values = [t['pcc'] for t in tracks]
            stats[cat_name] = {
                'max': float(np.max(pcc_values)),
                'mean': float(np.mean(pcc_values)),
                'count': len(tracks),
                'tracks': tracks
            }
        else:
            stats[cat_name] = {
                'max': 0.0,
                'mean': 0.0,
                'count': 0,
                'tracks': []
            }
    
    # Overall statistics
    valid_pcc = pcc_array[np.isfinite(pcc_array)]
    stats['All'] = {
        'mean': float(np.mean(valid_pcc)) if len(valid_pcc) > 0 else 0.0,
        'count': len(valid_pcc)
    }
    
    return stats


def print_category_statistics(stats):
    """Print formatted category statistics.
    
    Args:
        stats: Dictionary from compute_category_statistics
    """
    log.info("=" * 80)
    log.info("Track PCC Statistics by Category")
    log.info("=" * 80)
    
    # Print category-wise statistics
    categories = ['CAGE', 'ChIP-Histone', 'ChIP-TF', 'DNase/ATAC', 'Other']
    for cat in categories:
        if cat in stats and stats[cat]['count'] > 0:
            log.info(f"\n{cat}:")
            log.info(f"  Count: {stats[cat]['count']}")
            log.info(f"  Max PCC:  {stats[cat]['max']:.4f}")
            log.info(f"  Mean PCC: {stats[cat]['mean']:.4f}")
    
    # Print overall statistics
    log.info(f"\nAll Tracks:")
    log.info(f"  Count: {stats['All']['count']}")
    log.info(f"  Mean PCC: {stats['All']['mean']:.4f}")
    log.info("=" * 80)


def save_category_statistics(stats, output_dir, prefix="test"):
    """Save category statistics to JSON and CSV files.
    
    Args:
        stats: Dictionary from compute_category_statistics
        output_dir: Directory to save files
        prefix: Prefix for filename
    """
    import json
    import pandas as pd
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics to JSON
    summary = {}
    for cat in ['CAGE', 'ChIP-Histone', 'ChIP-TF', 'DNase/ATAC', 'Other', 'All']:
        if cat in stats:
            if cat == 'All':
                summary[cat] = {
                    'mean': stats[cat]['mean'],
                    'count': stats[cat]['count']
                }
            else:
                summary[cat] = {
                    'max': stats[cat]['max'],
                    'mean': stats[cat]['mean'],
                    'count': stats[cat]['count']
                }
    
    json_path = output_path / f'{prefix}_category_stats.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved category summary to {json_path}")
    
    # Save detailed track-level data to CSV
    all_tracks = []
    for cat in ['CAGE', 'ChIP-Histone', 'ChIP-TF', 'DNase/ATAC', 'Other']:
        if cat in stats:
            for track in stats[cat]['tracks']:
                all_tracks.append({
                    'category': cat,
                    'track_name': track['name'],
                    'track_index': track['index'],
                    'pcc': track['pcc']
                })
    
    if all_tracks:
        df = pd.DataFrame(all_tracks)
        # Sort by category and PCC (descending)
        df = df.sort_values(['category', 'pcc'], ascending=[True, False])
        csv_path = output_path / f'{prefix}_tracks_by_category.csv'
        df.to_csv(csv_path, index=False)
        log.info(f"Saved detailed track data to {csv_path}")
