import os
import pathlib

import numpy as np
import tensorflow as tf

from model import ZeoGAN
from config import (make_esgan_arg_parser,
                    write_config_log)
from dataset import EnergyShapeDataset

def main():
    parser = make_esgan_arg_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    energy_scale = args.energy_scale

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
        batch_size=args.batch_size,
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

    write_config_log(args, esgan.date)

    if args.restore_ckpt:
        # Extract steps
        name = pathlib.Path(args.restore_ckpt).name
        step = int(name.split("-")[-1])
    else:
        step = 0

    esgan.train(checkpoint=args.restore_ckpt, start_step=step)


if __name__ == "__main__":
    main()
