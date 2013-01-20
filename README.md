SWalign
============

SWalign is a CUDA parallelized sequence alignment software implementing Smith-Waterman's local alignment algorithm.

Authors:

Marija Mikulic, marija.mikulic@fer.hr

Luka Šterbić, luka.sterbic@fer.hr

Contributors:

Matija Korpar, matija.korpar@fer.hr

Index:
---------------------

1. Dependencies

2. Installation

3. Usage

4. Tests


1) Dependencies
---------------------

To properly run SWalign the following software is needed:

1. bash shell
2. nvcc 4.0+ (and compatible gcc version)
3. make
    
A CUDA capable GPU with computing capability 2.0 or greater is also needed.

This software was developed for the Fermi architecture and was tested under Fermi and Kepler.


2) Installation
---------------------

To instal SWalign run make in its root folder. This will build an executable named SWalign.


3) Usage
---------------------

`./SWalign <first_sequence> <second_sequence> <match> <mismatch> <gap_open> <gap_extend>`

SWalign expects 6 command line arguments. The first two are paths to sequences of interest in FASTA format.

The remaining arguments are integer values describing the scoring system.

Match should be positive, while mismatch, gap_open and gap_extend should be negative.


4) Tests
---------------------

To run a predefined test enter the following command while being placed in the SWalign root folder:

`./Test/test<X>.sh`

`<X>` = {1, 2, 3, 4, 5, 6}

The tests use sequences placed in <SWalign_root>/Sequences/
