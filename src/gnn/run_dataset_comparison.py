"""
run_dataset_comparison.py

CLI entry point for the biax_longer vs DeformingPlate comparison pipeline.

Usage (from repo root, with fenicsproject conda environment active):

python -m scripts.run_dataset_comparison \
    --biax-dir dataset/raw/raw/biax_longer/small/test/graphs \
    --dp-tfrecord deforming_plate/test.tfrecord \
    --n-timesteps 200 \
    --output-dir comparison_artifacts
"""

from src.gnn.comparison_defplate.runner import build_arg_parser, main

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
