import argparse
import functools
import glob
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf

from model import ZeoGAN
from config import (ArgumentParser,
                    make_esgan_arg_parser,
                    write_config_log,
                    make_args_from_config,
                    find_config_from_checkpoint)
from dataset import EnergyShapeDataset

def main():
    # Custom argparser.
    gen_parser = ArgumentParser()
    gen_parser.add_argument("--checkpoint", type=str)
    gen_parser.add_argument("--n_samples", type=int)
    gen_parser.add_argument("--savedir", type=str)
    gen_parser.add_argument("--device", type=str)
    gen_parser.add_argument("--batch_size", type=int)
    gen_parser.add_argument("--type", choices=["normal", "interp", "step"])

    # Parse args for gen.py
    gen_args = gen_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gen_args.device

    if gen_args.batch_size:
        batch_size = gen_args.batch_size
    else:
        # Assign default values.
        batch_size = 1 if (gen_args.type == "step") else 50

    parser = make_esgan_arg_parser()

    config = find_config_from_checkpoint(gen_args.checkpoint)
    # Parse original configs
    args = make_args_from_config(config)
    args = parser.parse_args(args)

    dataset = EnergyShapeDataset(
        path=args.dataset_path,
        rotate=args.rotate,
        shape=args.voxel_size,
        move=args.move,
        prefetch_size=256,
        shuffle_size=10000,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        cell_length_scale=args.cell_length_scale,
        invert=args.invert,
    )

    esgan = ZeoGAN(
        dataset=dataset,
        logdir=args.logdir,
        save_every=args.save_every,
        batch_size=batch_size,
        z_size=args.z_size,
        voxel_size=args.voxel_size,
        bottom_size=args.bottom_size,
        bottom_filters=args.bottom_filters,
        rate=args.rate,
        top_size=args.top_size,
        filter_unit=args.filter_unit,
        g_learning_rate=args.g_learning_rate,
        d_learning_rate=args.d_learning_rate,
        minibatch=args.minibatch,
        minibatch_kernel_size=args.minibatch_kernel_size,
        minibatch_dim_per_kernel=args.minibatch_dim_per_kernel,
        l2_loss=args.l2_loss,
        train_gen_per_disc=args.train_gen_per_disc,
        in_temper=args.in_temper,
        feature_matching=args.feature_matching,
        n_critics=args.n_critics,
        gp_lambda=args.gp_lambda,
        user_desired=args.user_desired,
        user_range=args.user_range,
    )

    if gen_args.type == "interp":
        esgan.interpolate_samples(
            sample_dir=gen_args.savedir,
            checkpoint=gen_args.checkpoint,
            n_samples=gen_args.n_samples,
        )
    elif gen_args.type == "normal":
        esgan.generate_samples(
            sample_dir=gen_args.savedir,
            checkpoint=gen_args.checkpoint,
            n_samples=gen_args.n_samples
        )
    elif gen_args.type == "step":
        z = np.random.normal(loc=0, scale=1, size=args.z_size)

        # Get all checkpoints from checkpoint data.
        expression = "-".join(gen_args.checkpoint.split("-")[:-1])
        # 27 samples.
        indices = list(range(0, 10000, 1000))
        indices += list(range(10000, 100000, 10000))
        indices += list(range(100000, 1000000, 100000))

        for index in indices:
            ckpt = "{}-{}".format(expression, index)
            print("Making:", ckpt)
            esgan.generate_sample_from_fixed_z(
                z=z,
                sample_dir=gen_args.savedir,
                checkpoint=ckpt,
            )


if __name__ == "__main__":
    main()
