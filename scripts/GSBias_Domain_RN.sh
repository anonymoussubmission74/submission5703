#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_gsbias.py     --config configs \
                                                --wandb-log \
                                                --datasets I/A/V/R/S \
                                                --backbone RN50 \
                                                --bs 64 \
                                                --selection_p 0.1