import torch
import numpy as np
import random
import pickle
import os
from torch.nn.functional import interpolate, grid_sample


def generate_spatial_noise(size, device, embeddings=None, *args, **kwargs):
    """ Generates a noise tensor. Currently uses torch.randn. """
    # noise = generate_noise([size[0], *size[2:]], *args, **kwargs)
    # return noise.expand(size)
    if embeddings is not None:
        directions = torch.stack(embeddings.values())
        rand_directions = torch.randint(0, len(directions), size[0], *size[2:])
        
        # TODO(frederik): return randn amount in direction from current point to chosen embedding
        pass
    else:
        return torch.randn(size, device=device)


def interpolate3D(data, shape, mode='bilinear', align_corners=False):
    d_1 = torch.linspace(-1, 1, shape[-3]) 
    d_2 = torch.linspace(-1, 1, shape[-2])
    d_3 = torch.linspace(-1, 1, shape[-1])
    meshz, meshy, meshx = torch.meshgrid((d_1, d_2, d_3))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0).to(data.dtype).to(data.device)  
    # Specifies the batch size
    N = data.shape[0]
    new_grid = torch.ones(N, grid.shape[1], grid.shape[2], grid.shape[3], 3).to(data.dtype).to(data.device)  
    new_grid[:N] = grid[0,:]
    scaled = grid_sample(data, new_grid, mode=mode, align_corners=align_corners)

    return scaled

def format_and_use_generator(curr_img, G_z, count, mode, Z_opt, pad_noise, pad_image, noise_amp, G, opt):
    """ Correctly formats input for generator and runs it through. """
  
    G_z = interpolate3D(G_z, curr_img.shape[-3:], mode='bilinear', align_corners=True)
    if mode == "rand":
        curr_img = pad_noise(curr_img)  # Curr image is z in this case
        z_add = curr_img
    else:
        z_add = Z_opt
    G_z = pad_image(G_z)
    z_in = noise_amp * z_add + G_z 
    G_z = G(z_in.detach(), G_z)
    return G_z


def draw_concat(generators, noise_maps, reals, noise_amplitudes, in_s, mode, pad_noise, pad_image, opt):
    """ Draw and concatenate output of the previous scale and a new noise map. """
    G_z = in_s
    if len(generators) > 0:
        if mode == "rand":
            noise_padding = 1 * opt.num_layer
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                    zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):
                if count < opt.stop_scale:  # - 1):
                    z = generate_spatial_noise([1,
                                                real_curr.shape[1],
                                                Z_opt.shape[2] - 2 * noise_padding,
                                                Z_opt.shape[3] - 2 * noise_padding,
                                                Z_opt.shape[4] - 2 * noise_padding],
                                                device=opt.device)

                G_z = format_and_use_generator(z, G_z, count, "rand", Z_opt,
                                               pad_noise, pad_image, noise_amp, G, opt)

        if mode == "rec":
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                    zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):
                G_z = format_and_use_generator(real_curr, G_z, count, "rec", Z_opt,
                                               pad_noise, pad_image, noise_amp, G, opt)

    return G_z

def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)

    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
    np.random.seed(seed)
    random.seed(seed)

def save_pkl(obj, name, prepath='default_path/'):
    with open(prepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name, prepath='default_path/'):
    with open(prepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)


