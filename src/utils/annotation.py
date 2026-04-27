"""
Annotation Utility for Genomic Sequence Retrieval

This module provides utilities for extracting DNA sequences from a reference genome
using BED file annotations. It's adapted from the BEND project's retrieve_from_bed.py.

Example Usage:
    annotation = Annotation('path/to/variants.bed', reference_genome='/path/to/genome.fa')
    seq = annotation.get_dna_segment(index=0)
"""

from Bio import SeqIO
import pandas as pd
from typing import Optional, Union
import os


class Annotation:
    """
    An annotation object that can be used to retrieve DNA segments from a reference genome.
    
    Adapted from BEND project (https://github.com/frederikkemarin/BEND)
    """
    
    def __init__(
        self, 
        annotation: Optional[Union[str, pd.DataFrame]] = None,
        reference_genome: Optional[str] = None
    ):
        """
        Initialize Annotation object for retrieving sequences from a reference genome.

        Parameters
        ----------
        annotation : str or pd.DataFrame, optional
            Path to a BED file containing genomic coordinates, or a DataFrame.
            The default is None.
        reference_genome : str, optional
            Path to a reference genome FASTA file.
            The default is None.
        """
        self.annotation = None
        self.genome_dict = None
        
        if annotation is not None:
            if isinstance(annotation, str):
                if not os.path.exists(annotation):
                    raise FileNotFoundError(f"BED file not found: {annotation}")
                self.annotation = pd.read_csv(annotation, sep='\t')
                print(f"Loaded annotation with {len(self.annotation)} entries from {annotation}")
            elif isinstance(annotation, pd.DataFrame):
                self.annotation = annotation
            else:
                raise ValueError(f"annotation must be a file path or DataFrame, got {type(annotation)}")
                
        if reference_genome is not None:
            if not os.path.exists(reference_genome):
                raise FileNotFoundError(f"Reference genome not found: {reference_genome}")
            print(f"Loading reference genome from {reference_genome}...")
            self.genome_dict = SeqIO.to_dict(SeqIO.parse(reference_genome, "fasta"))
            print(f"Loaded {len(self.genome_dict)} chromosomes/contigs")
    
    def extend_segments(
        self, 
        extra_context_left: Optional[int] = None, 
        extra_context_right: Optional[int] = None, 
        extra_context: Optional[int] = None
    ) -> None:
        """
        Add extra context to the coordinates in the annotation file.
        Each sample in the annotation file will be extended by extra_context_left 
        and extra_context_right.

        Parameters
        ----------
        extra_context_left : int, optional
            Number of nucleotides to add to the left of each segment.
            The default is None.
        extra_context_right : int, optional
            Number of nucleotides to add to the right of each segment.
            The default is None.
        extra_context : int, optional
            Number of nucleotides to add to both sides of each segment.
            Use this instead of extra_context_left and extra_context_right.
            The default is None.

        Raises
        ------
        ValueError
            If extra_context is used simultaneously with extra_context_left or extra_context_right.
        """
        if self.annotation is None:
            raise ValueError("No annotation loaded")
            
        if extra_context is not None:
            if extra_context_right is not None or extra_context_left is not None:
                raise ValueError('extra_context cannot be used with extra_context_left or extra_context_right')
            extra_context_left = extra_context
            extra_context_right = extra_context

        if extra_context_left is not None:
            self.annotation.loc[:, 'start'] = self.annotation.loc[:, 'start'] - extra_context_left
        if extra_context_right is not None:
            self.annotation.loc[:, 'end'] = self.annotation.loc[:, 'end'] + extra_context_right
    
    def get_item(self, index: int) -> pd.Series:
        """
        Get a row from the annotation file.

        Parameters
        ----------
        index : int
            Index of the row to return.

        Returns
        -------
        row : pandas.Series
            Row of the annotation file.
        """
        if self.annotation is None:
            raise ValueError("No annotation loaded")
        return self.annotation.iloc[index]

    def get_dna_segment(self, index: int) -> str:
        """
        Get a DNA sequence from the reference genome for a segment.

        Parameters
        ----------
        index : int
            Index of the row in the bed file for which to return the DNA sequence.
        
        Returns
        -------
        dna_segment : str
            The genomic DNA sequence of the segment.
        """
        if self.genome_dict is None:
            raise ValueError("No reference genome loaded")
            
        item = self.get_item(index)
        
        chrom = item.chromosome
        start = int(item.start)
        end = int(item.end)
        
        # Handle chromosome naming (with or without 'chr' prefix)
        if chrom not in self.genome_dict:
            if chrom.startswith('chr'):
                alt_chrom = chrom[3:]
            else:
                alt_chrom = 'chr' + chrom
            
            if alt_chrom in self.genome_dict:
                chrom = alt_chrom
            else:
                raise KeyError(f"Chromosome {item.chromosome} not found in reference genome")
        
        # Ensure valid coordinates
        seq_len = len(self.genome_dict[chrom].seq)
        start = max(0, start)
        end = min(end, seq_len)
        
        dna_segment = str(self.genome_dict[chrom].seq[start:end])
        
        return dna_segment.upper()
    
    def get_variant_sequences(
        self, 
        index: int, 
        extra_context_left: int = 0, 
        extra_context_right: int = 0
    ) -> tuple:
        """
        Get both reference and alternative sequences for a variant.
        
        Parameters
        ----------
        index : int
            Index of the variant in the annotation file.
        extra_context_left : int
            Extra context on the left side (5').
        extra_context_right : int
            Extra context on the right side (3').
            
        Returns
        -------
        tuple
            (ref_seq, alt_seq, variant_position_in_seq)
        """
        if self.genome_dict is None:
            raise ValueError("No reference genome loaded")
        
        item = self.get_item(index)
        
        chrom = item.chromosome
        var_pos = int(item.start)  # BED uses 0-based coordinates
        ref_allele = item.ref
        alt_allele = item.alt
        
        # Handle chromosome naming
        if chrom not in self.genome_dict:
            if chrom.startswith('chr'):
                alt_chrom = chrom[3:]
            else:
                alt_chrom = 'chr' + chrom
            
            if alt_chrom in self.genome_dict:
                chrom = alt_chrom
            else:
                raise KeyError(f"Chromosome {item.chromosome} not found in reference genome")
        
        # Calculate sequence boundaries
        seq_start = max(0, var_pos - extra_context_left)
        seq_end = var_pos + len(ref_allele) + extra_context_right
        seq_end = min(seq_end, len(self.genome_dict[chrom].seq))
        
        # Get reference sequence
        ref_seq = str(self.genome_dict[chrom].seq[seq_start:seq_end]).upper()
        
        # Position of variant in the extracted sequence
        var_pos_in_seq = var_pos - seq_start
        
        # Create alternative sequence
        alt_seq = list(ref_seq)
        
        # Replace reference allele with alternative allele
        # Handle SNPs (single nucleotide polymorphisms)
        if len(ref_allele) == 1 and len(alt_allele) == 1:
            alt_seq[var_pos_in_seq] = alt_allele
        else:
            # Handle indels (more complex)
            alt_seq = ref_seq[:var_pos_in_seq] + alt_allele + ref_seq[var_pos_in_seq + len(ref_allele):]
            alt_seq = list(alt_seq)
        
        return ''.join(ref_seq), ''.join(alt_seq), var_pos_in_seq
    
    def __len__(self) -> int:
        """Return the number of entries in the annotation."""
        if self.annotation is None:
            return 0
        return len(self.annotation)
    
    def filter_by_split(self, split: str) -> 'Annotation':
        """
        Filter annotation by split (train/val/test).
        
        Parameters
        ----------
        split : str
            The split to filter by ('train', 'val', 'test').
            
        Returns
        -------
        Annotation
            A new Annotation object with filtered data.
        """
        if self.annotation is None:
            raise ValueError("No annotation loaded")
        
        if 'split' not in self.annotation.columns:
            print("Warning: Annotation does not have 'split' column, returning all data")
            filtered = Annotation()
            filtered.annotation = self.annotation.copy().reset_index(drop=True)
            filtered.genome_dict = self.genome_dict
            return filtered
        
        # Check available splits
        available_splits = self.annotation['split'].unique().tolist()
        print(f"Available splits in data: {available_splits}")
        
        if split not in available_splits:
            print(f"Warning: Split '{split}' not found in data. Using all data instead.")
            filtered = Annotation()
            filtered.annotation = self.annotation.copy().reset_index(drop=True)
            filtered.genome_dict = self.genome_dict
            return filtered
        
        filtered = Annotation()
        filtered.annotation = self.annotation[self.annotation['split'] == split].reset_index(drop=True)
        filtered.genome_dict = self.genome_dict
        
        return filtered
    
    def get_label(self, index: int) -> int:
        """
        Get the label for a variant.
        
        Parameters
        ----------
        index : int
            Index of the variant.
            
        Returns
        -------
        int
            The label (0 or 1).
        """
        item = self.get_item(index)
        return int(item.label)
