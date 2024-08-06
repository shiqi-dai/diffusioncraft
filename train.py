"""
Train a diffusion model on a MineCraft world.
"""

import argparse
import os
import math
import random
from PIL import Image
import torchvision as tv
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    adjust_scales2image,
    adjust_scales2mc
)
from guided_diffusion.train_util import TrainLoop, parse_resume_step_from_filename

from worldgan.minecraft.level_utils import read_level as mc_read_level
from worldgan.utils import set_seed, load_pkl
import wandb

def main():
    args = create_argparser("./configs/train.yaml").parse_args()

    
    dist_util.setup_dist()
    logger.configure()

    real = mc_read_level(args) # (B, C, X, Y, Z)
    args.level_shape = real.shape[2:]
    adjust_scales2mc(real, args, args) # Complete the args parameters: num_scales, scale1, scale_factor, stop_scale

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        opt=args,
        mc_level=real,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        # scale_init=args.scale1,
        # scale_factor=args.scale_factor,
        stop_scale=args.stop_scale,
        current_scale=args.stop_scale
    ) 
    
    wandb_dir = "WANDB"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    tags = [args.input_area_name+args.input_area_size, str(args.predict_xstart), str(args.lr_anneal_steps)]
    run = wandb.init(project="DiffusionCraft_opensource", tags=tags, config=args, dir=wandb_dir)

    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        wg_opt=args,
        crop=args.crop
    ).run_loop()


def create_argparser(args):
    import yaml
    with open(args, 'r') as f:
        cfg = yaml.safe_load(f)
    
    """
    Minecraft Setting, copy from WolrdGAN.Config
    """
    if cfg["manualSeed"] is None:
        cfg["manualSeed"] = random.randint(1, 10000)
        set_seed(cfg["manualSeed"])

        cfg["num_scales"] = len(cfg["scales"]) # number of scales is implicitly defined
        cfg["noise_amp"] = 1.0  # noise amp for lowest scale always starts at 1
        cfg["seed_road"] = None  # for mario kart seed roads after training
        cfg["stop_scale"] = cfg["num_scales"] + 1 # which scale to stop on - usually always last scale defined
        cfg["sub_coords"] = cfg["sub_coord_dict"][cfg["input_area_name"]+cfg["input_area_size"]] # support multi biome


    if not cfg["server_train"]:
        coord_dict = load_pkl('primordial_coords_dict', './worldgan/minecraft/')
        tmp_coords = coord_dict[cfg["input_area_name"]]
        sub_coords = [(cfg["sub_coords"][0], cfg["sub_coords"][1]),
                        (cfg["sub_coords"][2], cfg["sub_coords"][3]),
                        (cfg["sub_coords"][4], cfg["sub_coords"][5])]
        coords = []
        for i, (start, end) in enumerate(sub_coords):
            curr_len = tmp_coords[i][1] - tmp_coords[i][0]
            if isinstance(start, float):
                tmp_start = curr_len * start + tmp_coords[i][0]
                tmp_end = curr_len * end + tmp_coords[i][0]
            elif isinstance(start, int):
                tmp_start = tmp_coords[i][0] + start
                tmp_end = tmp_coords[i][0] + end
            else:
                AttributeError("Unexpected type for sub_coords")
                tmp_start = tmp_coords[i][0]
                tmp_end = tmp_coords[i][1]
            coords.append((int(tmp_start), int(tmp_end)))
        cfg["coords"] = coords
    else:
        coords = []
        for i in range(len(cfg["server_start_pos"])):
            if cfg["server_start_pos"][i] < cfg["server_end_pos"][i]:
                coords.append((cfg["server_start_pos"][i], cfg["server_end_pos"][i]))
            else:
                coords.append((cfg["server_end_pos"][i], cfg["server_start_pos"][i]))
        cfg["coords"] = coords

        if cfg["repr_type"]:
            raise NotImplementedError("Server generation is not implemented with block2vec or bert yet.")

    if not cfg["repr_type"]:
        cfg["block2repr"] = None
    elif cfg["repr_type"] == "block2vec":
        # village分large和small两个
        if cfg["input_area_name"] == "vanilla_village":
            path = cfg["input_area_name"]+cfg["input_area_size"]
        else:
            path = cfg["input_area_name"]
        repr_fn = "{}/{}/".format(cfg["repr_fn"], path)
        cfg["block2repr"] = load_pkl("representations", repr_fn)
    else:
        AttributeError("unexpected repr_type, use "
                        "[None, block2vec, bert, one-hot-neighbors, neighbert, autoencoder]")

    """
    Guidied Diffusion Setting
    """
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=50000,
        num_channels_init=128,
        num_res_blocks_init=6,
        scale_factor_init=0.75,
        min_size=25,
        max_size=250,
        nc_im=3,
        batch_size=1,
        microbatch=-1,  
        ema_rate="0.9999",  
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        crop=0.8,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
