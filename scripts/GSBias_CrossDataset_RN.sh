#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_gsbias.py     --config configs \
                                                --wandb-log \
                                                --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
                                                --backbone RN50 \
                                                --bs 8 \
                                                --selection_p 0.5