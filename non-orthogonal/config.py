import os
import sys
import glob
import pathlib
import shutil
import argparse

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise Exception("Invalid str for bool cast: {}".format(v))


# Custom argument parser for file reading.
class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []

        # Take args only before # character
        arg_line = arg_line.split("#")[0]

        return arg_line.split()


def make_esgan_arg_parser():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--z_size", type=int, required=True)
    parser.add_argument("--voxel_size", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--move", type=str2bool, required=True)
    parser.add_argument("--rotate", type=str2bool, required=True)
    parser.add_argument("--invert", type=str2bool, required=True)
    parser.add_argument("--energy_limit", type=float, nargs=2, required=True)
    parser.add_argument("--energy_scale", type=float, nargs=2, required=True)
    parser.add_argument("--cell_length_scale",
                            type=float, nargs=2, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--bottom_size", type=int, required=True)
    parser.add_argument("--bottom_filters", type=int, required=True)
    parser.add_argument("--top_size", type=int, required=True)
    parser.add_argument("--filter_unit", type=int, required=True)
    parser.add_argument("--g_learning_rate", type=float, required=True)
    parser.add_argument("--d_learning_rate", type=float, required=True)
    parser.add_argument("--minibatch", type=str2bool, required=True)
    parser.add_argument("--minibatch_kernel_size", type=int, required=True)
    parser.add_argument("--minibatch_dim_per_kernel", type=int, required=True)
    parser.add_argument("--l2_loss", type=str2bool, required=True)
    parser.add_argument("--train_gen_per_disc", type=int, required=True)
    parser.add_argument("--n_critics", type=int, required=True)
    parser.add_argument("--gp_lambda", type=float, required=True)
    parser.add_argument("--user_desired", type=str2bool, required=True)
    parser.add_argument("--user_range", type=float, nargs=2, required=True)
    parser.add_argument("--device", type=str, required=True)

    # Optional arguments
    parser.add_argument("--in_temper", type=float, default=300.0)
    parser.add_argument("--feature_matching", type=str2bool, default=True)
    parser.add_argument("--restore_ckpt", type=str)

    return parser


def write_config_log(args, date):
    logdir = args.logdir

    # Try to make parent folder.
    try:
        os.makedirs(logdir)
    except Exception as e:
        print("Error:", e, "but keep going.")

    items = sorted(list(args.__dict__.items()))
    with open("{}/config-{}".format(logdir, date), "w") as f:
        for key, val in items:
            f.write("{} {}\n".format(key, val))


def make_args_from_config(config):
    """Convert config file to args that can be used for argparser."""
    # Get file name (not a path)
    #config_name = config.split("/")[-1]
    # Get parent path
    #config_folder = "/".join(config.split("/")[:-1])

    args = list()
    with open(config, "r") as f:
        for line in f:
            # Neglect restore_ckpt...
            if "restore" in line:
                continue
            # Remove trash chars
            line = line.replace("[", "")
            line = line.replace(",", "")
            line = line.replace("]", "")

            line = "--" + line

            args += line.split()

    return args


def cache_ckpt_from_config(*, cache_folder, config):
    # Get file name (not a path)
    config_name = config.split("/")[-1]
    # Get parent path
    config_folder = "/".join(config.split("/")[:-1])
    # Extract path
    date = "-".join(config_name.split("-")[1:])

    expression = "{}/save-{}-*".format(config_folder, date)
    ckpts = glob.glob(expression)

    try:
        for f in ckpts:
            shutil.copy2(f, cache_folder)
    except Exception as e:
        raise Exception(str(e) + ", Terminate program")

    # ckpt path example
    # /path/to/ckpt/save-2018-02-01T20:03:39.639994-35900
    ckpt = ".".join(ckpts[0].split(".")[:-1])
    ckpt = ckpt.split("/")[-1] # save-2018-02-01T20:03:39.639994-35900
    final_step = int(ckpt.split("-")[-1]) # 35900 as int
    ckpt = "{}/{}".format(cache_folder, ckpt)

    return ckpt, final_step


def find_config_from_checkpoint(checkpoint):
    # strict=True will raise an exception if file does not exist.
    path = pathlib.Path(checkpoint + ".data-00000-of-00001").resolve()

    # Checkpoint format: /path/to/ckpt/save-2018-02-01T20:03:39.639994-35900
    # stem not a name. because ".data-00000-of-00001" is added to ckpt.
    name = path.stem
    parent = path.parent

    # Extract time, remove "save" and "steps" at front and tail of the name.
    time = "-".join(name.split("-")[1:-1])
    # Generate config path.
    config = "{}/config-{}".format(parent, time)

    return config


def _test():
    """
    parser = make_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    print(args.__dict__)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt = cache_ckpt_from_config(cache_folder=temp_dir, config=sys.argv[1])
        print(ckpt)
    """

    print(find_config_from_checkpoint(checkpoint="/home/lsw/Workspace/EGRID_GAN/EGGAN/tensorboard/eggan/zeo-100/save-2018-03-21T15:48:06.635430-1565000"))


if __name__ == "__main__":
    _test()
