"""Test script for SF-Fuse models.

This script evaluates a trained model on a test dataset and computes:
- Overall PCC statistics
- Category-wise PCC statistics (CAGE, ChIP-Histone, ChIP-TF, DNase/ATAC)
- Detailed per-track results

Usage:
    # Single file mode (sequences and targets in one H5 file)
    python test.py --ckpt_path /path/to/checkpoint.ckpt \
                   --test_data /path/to/human_test.h5 \
                   --output_dir ./test_results

    # Two file mode (separate sequences and targets)
    python test.py --ckpt_path /path/to/checkpoint.ckpt \
                   --test_data /path/to/test_sequences.h5 \
                   --test_targets /path/to/test_targets.h5 \
                   --output_dir ./test_results
"""

import argparse
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloaders.genomic_dataset import GenomicTracksDataset, worker_init_fn
from src.tasks.track_utils import (
    compute_category_statistics,
    load_track_names,
    print_category_statistics,
    save_category_statistics,
    save_pcc_ranking,
)
from src.utils.config import get_logger

log = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test SF-Fuse model")
    
    # Required arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test H5 file (can contain both sequences and targets, or just sequences)"
    )
    parser.add_argument(
        "--test_targets",
        type=str,
        default=None,
        help="Path to test targets H5 file (optional, if None will use test_data for both sequences and targets)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (optional, will use checkpoint config if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--track_names",
        type=str,
        default=None,
        help="Path to track names file (one track name per line)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=131072,
        help="Maximum sequence length"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path, config=None):
    """Load model from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        config: Optional config dict/DictConfig to override checkpoint config
        
    Returns:
        Loaded model (LightningModule)
    """
    log.info(f"Loading model from checkpoint: {ckpt_path}")

    # Determine task class first (so we can import torch later)
    # Load checkpoint lightly to get task class info
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Get model class from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        log.info(f"Checkpoint hyperparameters: {list(hparams.keys())}")
    else:
        raise ValueError("Checkpoint does not contain hyper_parameters")
    
    # Determine task class
    # Try to infer from checkpoint or use SFFuseTask by default
    task_class_name = hparams.get('_target_', 'src.tasks.unified_task.SFFuseTask')
    
    if 'unified' in task_class_name.lower():
        from src.tasks.unified_task import SFFuseTask
        task_class = SFFuseTask
    elif 'cnn_stem' in task_class_name.lower():
        from src.tasks.sf_fuse_cnn_stem_ft import SFFuseCNNStemFineTuning
        task_class = SFFuseCNNStemFineTuning
    elif 'sandwich' in task_class_name.lower():
        from src.tasks.sf_fuse_sandwich_ft import SFFuseSandwichFineTuning
        task_class = SFFuseSandwichFineTuning
    else:
        from src.tasks.unified_task import SFFuseTask
        task_class = SFFuseTask
        log.warning(f"Unknown task class {task_class_name}, using SFFuseTask")
    
    log.info(f"Using task class: {task_class.__name__}")
    
    # Load model with checkpoint
    try:
        model = task_class.load_from_checkpoint(ckpt_path, strict=False, weights_only=False)
        model.eval()
        log.info("Model loaded successfully")
        return model
    except ImportError as e:
        log.error(f"\n{'='*80}")
        log.error(f"ImportError: {e}")
        log.error(f"{'='*80}")
        log.error("SOLUTION: Package version conflict detected!")
        log.error("")
        log.error("This error occurs when mamba-ssm requires a specific transformers version.")
        log.error("Please try ONE of these solutions:")
        log.error("")
        log.error("Option 1 (Recommended):")
        log.error("  pip install transformers==4.36.0")
        log.error("")
        log.error("Option 2:")
        log.error("  pip install transformers==4.31.0")
        log.error("")
        log.error("Option 3 (If mamba-ssm not needed):")
        log.error("  pip uninstall mamba-ssm -y")
        log.error(f"{'='*80}\n")
        raise
    except Exception as e:
        log.error(f"Error loading model: {e}")
        raise


@torch.no_grad()
def test_model(model, test_loader, device, track_names):
    """Run model inference on test data and compute metrics.
    
    Uses online (streaming) PCC computation to avoid memory overflow.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run on
        track_names: List of track names
        
    Returns:
        Dictionary with PCC per track
    """
    log.info("Starting model evaluation on test set...")
    
    model = model.to(device)
    model.eval()
    
    # Initialize accumulators for online PCC computation
    num_tracks = None
    sum_pred = None
    sum_target = None
    sum_pred_sq = None
    sum_target_sq = None
    sum_pred_target = None
    n_samples = 0
    
    # Run inference with online statistics computation
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        y_pred = model(x)
        
        # Move to CPU and flatten spatial dimension
        y_pred = y_pred.cpu().float()  # (batch, 896, num_tracks)
        y = y.cpu().float()  # (batch, 896, num_tracks)
        
        # Flatten batch and spatial dimensions: (batch * 896, num_tracks)
        batch_size, seq_len, num_tracks_batch = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, num_tracks_batch)  # (batch*896, num_tracks)
        y_flat = y.reshape(-1, num_tracks_batch)  # (batch*896, num_tracks)
        
        # Initialize accumulators on first batch
        if sum_pred is None:
            num_tracks = num_tracks_batch
            sum_pred = torch.zeros(num_tracks, dtype=torch.float64)
            sum_target = torch.zeros(num_tracks, dtype=torch.float64)
            sum_pred_sq = torch.zeros(num_tracks, dtype=torch.float64)
            sum_target_sq = torch.zeros(num_tracks, dtype=torch.float64)
            sum_pred_target = torch.zeros(num_tracks, dtype=torch.float64)
        
        # Accumulate statistics for PCC computation
        # Convert to float64 for numerical stability
        y_pred_flat = y_pred_flat.double()
        y_flat = y_flat.double()
        
        sum_pred += y_pred_flat.sum(dim=0)
        sum_target += y_flat.sum(dim=0)
        sum_pred_sq += (y_pred_flat ** 2).sum(dim=0)
        sum_target_sq += (y_flat ** 2).sum(dim=0)
        sum_pred_target += (y_pred_flat * y_flat).sum(dim=0)
        n_samples += y_pred_flat.shape[0]
        
        # Free memory
        del x, y, y_pred, y_pred_flat, y_flat
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    log.info(f"Processed {n_samples} data points across {num_tracks} tracks")
    
    # Compute PCC from accumulated statistics
    log.info("Computing Pearson correlation per track...")
    pcc_per_track = compute_pcc_from_stats(
        sum_pred, sum_target, sum_pred_sq, sum_target_sq, sum_pred_target, n_samples
    )
    
    log.info(f"PCC per track shape: {pcc_per_track.shape}")
    log.info(f"Overall mean PCC: {pcc_per_track.mean().item():.4f}")
    
    # Return only PCC, not the full predictions/targets
    return {
        'pcc_per_track': pcc_per_track,
    }


def compute_pcc_from_stats(sum_x, sum_y, sum_x_sq, sum_y_sq, sum_xy, n):
    """Compute Pearson correlation coefficient from accumulated statistics.
    
    PCC = cov(X,Y) / (std(X) * std(Y))
        = [E(XY) - E(X)E(Y)] / sqrt([E(X²) - E(X)²] * [E(Y²) - E(Y)²])
    
    Args:
        sum_x: Sum of predictions per track
        sum_y: Sum of targets per track
        sum_x_sq: Sum of squared predictions per track
        sum_y_sq: Sum of squared targets per track
        sum_xy: Sum of prediction * target per track
        n: Total number of samples
        
    Returns:
        PCC per track (num_tracks,)
    """
    # Compute means
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    # Compute covariance
    cov_xy = (sum_xy / n) - (mean_x * mean_y)
    
    # Compute standard deviations
    var_x = (sum_x_sq / n) - (mean_x ** 2)
    var_y = (sum_y_sq / n) - (mean_y ** 2)
    std_x = torch.sqrt(torch.clamp(var_x, min=1e-10))
    std_y = torch.sqrt(torch.clamp(var_y, min=1e-10))
    
    # Compute PCC
    pcc = cov_xy / (std_x * std_y + 1e-10)
    
    # Handle invalid values
    pcc = torch.where(torch.isfinite(pcc), pcc, torch.zeros_like(pcc))
    pcc = torch.clamp(pcc, -1.0, 1.0)  # Ensure [-1, 1] range
    
    return pcc.float()


def main():
    """Main testing function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log.info("=" * 80)
    log.info("SF-Fuse Model Testing")
    log.info("=" * 80)
    log.info(f"Checkpoint: {args.ckpt_path}")
    log.info(f"Test data: {args.test_data}")
    if args.test_targets:
        log.info(f"Test targets: {args.test_targets} (separate file mode)")
    else:
        log.info(f"Test targets: Using test_data (single file mode)")
    log.info(f"Output directory: {args.output_dir}")
    log.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model_from_checkpoint(args.ckpt_path, args.config)
    
    # Get number of tracks from model
    if hasattr(model, 'head'):
        if hasattr(model.head, 'num_tracks'):
            num_tracks = model.head.num_tracks
        elif hasattr(model.head, 'final_project'):
            num_tracks = model.head.final_project.out_features
        else:
            num_tracks = 5313  # Default
    else:
        num_tracks = 5313  # Default
    
    log.info(f"Number of tracks: {num_tracks}")
    
    # Load track names
    track_names = load_track_names(args.track_names, num_tracks)
    log.info(f"Loaded {len(track_names)} track names")
    
    # Create test dataset and dataloader
    log.info("Creating test dataset...")
    test_dataset = GenomicTracksDataset(
        data_file=args.test_data,
        targets_file=args.test_targets,
        seq_key='sequences',
        tgt_key='targets',
        max_length=args.max_length,
        rc_augment=False,  # No augmentation for testing
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    log.info(f"Test dataset size: {len(test_dataset)}")
    
    # Determine device
    if torch.cuda.is_available() and args.devices > 0:
        device = torch.device('cuda')
        log.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        log.info("Using CPU")
    
    # Run testing
    results = test_model(model, test_loader, device, track_names)
    
    pcc_per_track = results['pcc_per_track']
    
    # Compute category statistics
    log.info("\nComputing category-wise statistics...")
    stats = compute_category_statistics(pcc_per_track, track_names)
    
    # Print results
    print_category_statistics(stats)
    
    # Save results
    log.info(f"\nSaving results to {output_dir}...")
    
    # Save category statistics
    save_category_statistics(stats, output_dir, prefix="test")
    
    # Save full PCC ranking
    save_pcc_ranking(pcc_per_track, track_names, output_dir, epoch=0, prefix="test")
    
    # Save overall summary
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SF-Fuse Model Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Test data: {args.test_data}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Number of tracks: {num_tracks}\n\n")
        
        f.write("Category-wise Statistics:\n")
        f.write("-" * 80 + "\n")
        for cat in ['CAGE', 'ChIP-Histone', 'ChIP-TF', 'DNase/ATAC', 'Other']:
            if stats[cat]['count'] > 0:
                f.write(f"\n{cat}:\n")
                f.write(f"  Count:     {stats[cat]['count']}\n")
                f.write(f"  Max PCC:   {stats[cat]['max']:.4f}\n")
                f.write(f"  Mean PCC:  {stats[cat]['mean']:.4f}\n")
        
        f.write(f"\nAll Tracks:\n")
        f.write(f"  Count:     {stats['All']['count']}\n")
        f.write(f"  Mean PCC:  {stats['All']['mean']:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    log.info(f"Saved summary to {summary_path}")
    
    log.info("\n" + "=" * 80)
    log.info("Testing completed successfully!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
