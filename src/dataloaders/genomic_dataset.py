import logging

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

log = logging.getLogger(__name__)

# Reverse complement mapping: A(0)<->T(3), C(1)<->G(2)
_RC_MAP = np.array([10, 9, 8, 7], dtype=np.int64)  # Complement map for token IDs 7-10 (A=7,C=8,G=9,T=10)


def worker_init_fn(worker_id):
    """Re-open HDF5 file handles in each DataLoader worker process.
    
    HDF5 file objects are NOT fork-safe. When ``num_workers > 0``,
    each worker is a forked subprocess and inherits the parent's file
    descriptors, which can silently corrupt reads.  Call this function
    via ``DataLoader(worker_init_fn=worker_init_fn)`` so that every
    worker opens its own independent file handles.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, GenomicTracksDataset):
            dataset._open_files()


class GenomicTracksDataset(Dataset):
    """
    Dataset to load paired DNA sequences and Genomic Tracks from H5 files.
    Compatible with 'yangyz1230/space' dataset format.
    
    Features:
    - Lazy file handle opening (re-opened in each worker for fork safety)
    - Optional reverse-complement (RC) augmentation for training
    """
    def __init__(
        self,
        data_file,
        targets_file=None,
        seq_key='sequences',
        tgt_key='targets',
        max_length=131072,
        rc_augment=False,
    ):
        """
        Args:
            data_file (str): Path to H5 file containing sequences (and optionally targets).
            targets_file (str, optional): Path to H5 file containing targets. If None, uses data_file.
            seq_key (str): Key for sequences in H5 file. Default 'sequences'.
            tgt_key (str): Key for targets in H5 file. Default 'targets'.
            max_length (int): Expected sequence length.
            rc_augment (bool): Whether to apply reverse-complement augmentation
                with 50% probability. Default False.
        """
        self.data_file = data_file
        self.targets_file = targets_file if targets_file is not None else data_file
        self.seq_key = seq_key
        self.tgt_key = tgt_key
        self.max_length = max_length
        self.rc_augment = rc_augment
        
        # File handles (lazily opened, re-opened per worker)
        self._h5_seq = None
        self._h5_tgt = None
        
        # Verify file existence and get length
        try:
            with h5py.File(self.data_file, 'r') as f:
                if self.seq_key not in f:
                    raise KeyError(f"Key '{self.seq_key}' not found in {self.data_file}. Available keys: {list(f.keys())}")
                self.dataset_len = f[self.seq_key].shape[0]
                self.seq_shape = f[self.seq_key].shape
                log.info(f"Loaded dataset from {self.data_file}. Shape: {self.seq_shape}. Length: {self.dataset_len}")
        except Exception as e:
            log.error(f"Error initializing dataset: {e}")
            raise

    def _open_files(self):
        """Open (or re-open) HDF5 file handles."""
        # Close existing handles if any
        if self._h5_seq is not None:
            try:
                self._h5_seq.close()
            except Exception:
                pass
        if self._h5_tgt is not None and self._h5_tgt is not self._h5_seq:
            try:
                self._h5_tgt.close()
            except Exception:
                pass
        
        self._h5_seq = h5py.File(self.data_file, 'r')
        if self.targets_file == self.data_file:
            self._h5_tgt = self._h5_seq
        else:
            self._h5_tgt = h5py.File(self.targets_file, 'r')

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Lazy open: if file handles are not yet open, open them.
        # This also covers the main-process path when num_workers=0.
        if self._h5_seq is None:
            self._open_files()
        
        # Load Sequence
        seq_data = self._h5_seq[self.seq_key][idx]
        
        # Convert One-Hot to Indices (if one-hot)
        if seq_data.ndim == 2 and seq_data.shape[-1] == 4:
            seq = np.argmax(seq_data, axis=-1).astype(np.int64) + 7  # 0→7(A), 1→8(C), 2→9(G), 3→10(T)
        else:
            seq = seq_data.astype(np.int64)

        # Load Target
        target_data = self._h5_tgt[self.tgt_key][idx]
        target = target_data.astype(np.float32)
        
        # --- Reverse Complement Augmentation ---
        # Reverse the sequence order AND swap complement bases.
        # For targets: reverse the bin order (spatial flip along the genome)
        if self.rc_augment and np.random.random() < 0.5:
            seq = _RC_MAP[seq[::-1] - 7].copy()  # Subtract 7 to index 0-3 into RC_MAP
            target = target[::-1].copy()

        seq = torch.from_numpy(seq)
        target = torch.from_numpy(target)

        return seq, target

    def __del__(self):
        """Clean up file handles on deletion."""
        if self._h5_seq is not None:
            try:
                self._h5_seq.close()
            except Exception:
                pass
        if self._h5_tgt is not None and self._h5_tgt is not self._h5_seq:
            try:
                self._h5_tgt.close()
            except Exception:
                pass
