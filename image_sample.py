"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path

from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def get_sampling_fn(diffusion, sampling_method):
    if sampling_method=='ddim':
        return diffusion.ddim_sample_loop
    elif sampling_method=='pc-ode':
        return diffusion.pc_sample_loop
    else:
        return diffusion.p_sample_loop

def main():
    args = create_argparser().parse_args()
    print(args.index2time_dir)

    #dist_util.setup_dist()
    #logger.configure()

    #logger.log("creating model and diffusion...")
    print("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    #model.load_state_dict(
    #    dist_util.load_state_dict(args.model_path, map_location="cpu")
    #)
    device = 'cuda'
    model.load_state_dict(th.load(args.model_path))
    model.to(device)

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    #logger.log("sampling...")
    print("sampling...")

    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = get_sampling_fn(diffusion, args.sampling_method)
        
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs, progress=True
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        #gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        #dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        #all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images.extend(sample.cpu().numpy())

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        #logger.log(f"created {len(all_images) * args.batch_size} samples")
        print(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    #if dist.get_rank() == 0:

    Path(args.write_dir).mkdir(parents=True, exist_ok=True)

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(args.write_dir, f"samples_{shape_str}.npz")
    #logger.log(f"saving to {out_path}")
    print(f"saving to {out_path}")

    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    #dist.barrier()
    #logger.log("sampling complete")
    print("sampling complete")


def create_argparser():
    defaults = dict(clip_denoised=True,
                    num_samples=10000,
                    batch_size=16,
                    use_ddim=False,
                    model_path="",
                    write_dir="",
                    sampling_method='sde'
                )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
