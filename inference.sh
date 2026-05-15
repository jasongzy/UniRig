#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

INPUT="examples/giraffe.glb"
OUTPUT_PREFIX="results/giraffe"

bash launch/inference/generate_skeleton.sh --input "$INPUT" --output "${OUTPUT_PREFIX}_skel.fbx"

bash launch/inference/generate_skin.sh --input "${OUTPUT_PREFIX}_skel.fbx" --output "${OUTPUT_PREFIX}_skin.fbx"

bash launch/inference/merge.sh --source "${OUTPUT_PREFIX}_skin.fbx" --target "$INPUT" --output "${OUTPUT_PREFIX}_rigged.glb"
