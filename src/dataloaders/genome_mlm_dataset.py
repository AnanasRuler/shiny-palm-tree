"""
Reference Genome MLM Dataset for encoder pre-training.

This module provides a dataset that loads reference genome data (FASTA)
and samples windows using a BED file for defining genomic intervals,
following the Caduceus hg38_dataset.py approach.

Key differences from the original implementation:
    1. Uses BED file for defining genomic intervals (not random sampling)
    2. Implements shift-based sampling like Caduceus:
       - MAX_ALLOWED_LENGTH = 2^20 = 1,048,576
       - shifts = MAX_ALLOWED_LENGTH // max_length
       - Each BED row generates `shifts` samples
    3. Uses CaduceusTokenizer for tokenization
    4. Uses pyfaidx for indexed FASTA access
    5. No RC augmentation at dataset level (handle in model if needed)
    6. No H5 mode support (FASTA mode only)

Token ID Mapping (CaduceusTokenizer):
    [CLS]: 0, [SEP]: 1, [BOS]: 2, [MASK]: 3, [PAD]: 4, [RESERVED]: 5, [UNK]: 6
    A: 7, C: 8, G: 9, T: 10, N: 11
"""

import logging
import math
from pathlib import Path

import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)

# Caduceus standard: 2^20 = 1,048,576
MAX_ALLOWED_LENGTH = 2 ** 20

# Token IDs from CaduceusTokenizer
_PAD_TOKEN_ID = 4
_MASK_TOKEN_ID = 3
_IGNORE_INDEX = -100  # Label value for non-masked positions (ignored in loss)


def mlm_worker_init_fn(worker_id):
    """Re-open file handles in each DataLoader worker process.

    FASTA index handles are NOT fork-safe. Each worker must
    open its own independent handles.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, ReferenceGenomeMLMDataset):
            dataset._open_data_source()


class FastaInterval:
    """Retrieves sequences from a fasta file given a chromosome and start/end indices."""

    def __init__(
        self,
        fasta_file: str,
        rc_aug: bool = False,
    ):
        fasta_path = Path(fasta_file)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        self.seqs = Fasta(str(fasta_path))
        self.rc_aug = rc_aug

        # Cache chromosome lengths
        self.chr_lens = {}
        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])

    @staticmethod
    def _compute_interval(start: int, end: int, max_length: int, i_shift: int):
        """Compute the actual interval based on shift index.

        For shift-based sampling, each BED row of length MAX_ALLOWED_LENGTH
        is divided into `shifts` samples of length `max_length`.

        Args:
            start: Start position from BED file
            end: End position from BED file
            max_length: Target sequence length
            i_shift: Shift index (0 to shifts-1)

        Returns:
            Tuple of (new_start, new_end) for the specific shift
        """
        if max_length == MAX_ALLOWED_LENGTH:
            return start, end
        if max_length < MAX_ALLOWED_LENGTH:
            assert MAX_ALLOWED_LENGTH % max_length == 0, \
                f"max_length ({max_length}) must be a power of 2 dividing {MAX_ALLOWED_LENGTH}"
            return start + i_shift * max_length, start + (i_shift + 1) * max_length
        else:
            raise ValueError(
                f"`max_length` {max_length} (> 2^{int(math.log2(MAX_ALLOWED_LENGTH))}) is too large!"
            )

    def __call__(
        self,
        chr_name: str,
        start: int,
        end: int,
        max_length: int,
        i_shift: int,
        return_augs: bool = False,
    ) -> str:
        """Retrieve a sequence from the FASTA file.

        Args:
            chr_name: Chromosome name
            start: Start position (0-based)
            end: End position
            max_length: Target sequence length
            i_shift: Shift index for sub-sampling
            return_augs: Whether to return augmentation info (not used)

        Returns:
            DNA sequence string
        """
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        # Compute actual interval based on shift
        start, end = self._compute_interval(start, end, max_length, i_shift)

        # Handle boundary conditions
        if end > chromosome_length:
            # Shift interval down
            start = start - (end - chromosome_length)
            end = chromosome_length
            assert start == chromosome_length - max_length

        if start < 0:
            # Shift interval up
            end = end - start
            start = 0
            assert end == max_length

        if end > chromosome_length:
            # This may occur if start + MAX_ALLOWED_LENGTH extends beyond chromosome end
            start = chromosome_length - max_length
            end = chromosome_length

        seq = str(chromosome[start:end])

        # Note: RC augmentation is disabled by default for MLM pre-training
        # Handle in model if needed

        return seq


class ReferenceGenomeMLMDataset(Dataset):
    """Dataset for MLM pre-training using reference genome data.

    Follows the Caduceus HG38Dataset approach:
    - Loads genomic intervals from a BED file
    - Uses shift-based sampling: each BED row generates `shifts` samples
    - MAX_ALLOWED_LENGTH = 2^20 = 1,048,576
    - shifts = MAX_ALLOWED_LENGTH // max_length
    - Uses CaduceusTokenizer for tokenization
    - Applies BERT-style MLM masking

    Args:
        bed_file: Path to BED file defining genomic intervals
        fasta_file: Path to reference genome FASTA file (.fa/.fasta)
        split: Split name to filter from BED file (e.g., 'train', 'val', 'test')
        max_length: Length of each sequence (must be power of 2, <= 2^20)
        mlm: Enable MLM masking (default: True)
        mlm_probability: Probability of masking each token (default: 0.15)
        tokenizer: CaduceusTokenizer instance for tokenization
        rc_augment: Apply reverse complement augmentation (default: False)
    """

    def __init__(
        self,
        bed_file: str,
        fasta_file: str,
        split: str = "train",
        max_length: int = 131072,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        tokenizer: PreTrainedTokenizer = None,
        rc_augment: bool = False,
    ):
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        if self.mlm and self.mlm_probability <= 0.0:
            raise ValueError(f"`mlm_probability` must be > 0.0, got {self.mlm_probability}")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.rc_augment = rc_augment

        # Validate max_length
        if max_length <= MAX_ALLOWED_LENGTH:
            assert MAX_ALLOWED_LENGTH % max_length == 0, \
                f"`max_length` must be a power of 2 dividing {MAX_ALLOWED_LENGTH}"
            self.shifts = MAX_ALLOWED_LENGTH // max_length
        else:
            raise ValueError(
                f"`max_length` {max_length} (> 2^{int(math.log2(MAX_ALLOWED_LENGTH))}) is too large!"
            )

        # Load BED file
        bed_path = Path(bed_file)
        if not bed_path.exists():
            raise FileNotFoundError(f"BED file not found: {bed_file}")

        # Read BED file (format: chr_name, start, end, split)
        df_raw = pd.read_csv(str(bed_path), sep="\t", names=["chr_name", "start", "end", "split"])

        # Filter by split
        self.df = df_raw[df_raw["split"] == split].copy()

        if len(self.df) == 0:
            raise ValueError(f"No entries found for split '{split}' in BED file")

        # Update end points so sequences are MAX_ALLOWED_LENGTH
        self.df.loc[:, "end"] = self.df["start"] + MAX_ALLOWED_LENGTH

        # Initialize FASTA handler
        self.fasta = FastaInterval(
            fasta_file=fasta_file,
            rc_aug=self.rc_augment,
        )

        # Lazy-open data handle (will be opened in worker)
        self._data_handle = None

        log.info(
            f"MLM Dataset initialized: split={split}, "
            f"max_length={max_length}, shifts={self.shifts}, "
            f"bed_entries={len(self.df)}, total_samples={len(self.df) * self.shifts}, "
            f"mlm={mlm}, mlm_prob={mlm_probability}"
        )

    def _open_data_source(self):
        """Open (or re-open) the FASTA handle. Called per-worker."""
        if self._data_handle is not None:
            try:
                self._data_handle.close()
            except Exception:
                pass

        self._data_handle = Fasta(self.fasta.seqs.filename)

    def __len__(self) -> int:
        """Total number of samples = BED rows * shifts per row."""
        return len(self.df) * self.shifts

    def __getitem__(self, idx):
        """Return (masked_input_ids, mlm_labels) for one sample.

        Args:
            idx: Sample index

        Returns:
            data: Tensor of shape (max_length,) with MLM masking applied
            target: Tensor of shape (max_length,) with -100 for non-masked positions
        """
        # Lazy-open data source
        if self._data_handle is None:
            self._open_data_source()

        # Compute row and shift indices
        row_idx = idx // self.shifts
        shift_idx = idx % self.shifts

        # Get BED row
        row = self.df.iloc[row_idx]
        chr_name = row["chr_name"]
        start = row["start"]
        end = row["end"]

        # Retrieve sequence
        seq_str = self.fasta(
            chr_name=chr_name,
            start=start,
            end=end,
            max_length=self.max_length,
            i_shift=shift_idx,
        )

        # Tokenize using CaduceusTokenizer
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        # Tokenize the sequence
        encoding = self.tokenizer(
            seq_str,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = encoding["input_ids"]
        seq = torch.LongTensor(input_ids)

        # Replace N token with PAD token (ignore in loss)
        n_token_id = self.tokenizer._vocab_str_to_int.get("N", 6)  # Default to UNK if N not found
        seq = torch.where(seq == n_token_id, self.tokenizer.pad_token_id, seq)

        # Apply MLM masking
        if self.mlm:
            data, target = self._apply_mlm_masking(seq)
        else:
            # Without MLM, return shifted sequences (standard language modeling)
            data = seq[:-1].clone()
            target = seq[1:].clone()

        return data, target

    def _apply_mlm_masking(self, seq: torch.Tensor):
        """Apply BERT-style MLM masking.

        Args:
            seq: Tensor of shape (seq_length,) with token IDs

        Returns:
            data: Tensor with MLM masking applied
            target: Tensor with pad_token_id for non-masked positions (ignored in loss)
        """
        data = seq.clone()
        target = seq.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(data.shape, self.mlm_probability)

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set target to PAD for non-masked positions (ignore in loss)
        target[~masked_indices] = self.tokenizer.pad_token_id

        # 80% of masked tokens -> [MASK] token
        indices_replaced = torch.bernoulli(torch.full(data.shape, 0.8)).bool() & masked_indices
        data[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of masked tokens -> random token
        indices_random = torch.bernoulli(torch.full(data.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), size=data.shape, dtype=torch.long)
        data[indices_random] = random_words[indices_random]

        # Remaining 10% of masked tokens -> keep original (already copied)

        return data, target
