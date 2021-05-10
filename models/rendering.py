import torch


def get_rays_shapenet(hwf, poses):
    """
    shapenet camera intrinsics are defined by H, W and focal.
    this function can handle multiple camera poses at a time.
    Args:
        hwf (3,): H, W, focal
        poses (N, 4, 4): pose for N number of images
        
    Returns:
        rays_o (N, H, W, 3): ray origins
        rays_d (N, H, W, 3): ray directions
    """
    if poses.ndim == 2:
        poses = poses.unsqueeze(dim=0)  # if poses has shape (4, 4)
                                        # make it (1, 4, 4)

    H, W, focal = hwf
    yy, xx = torch.meshgrid(torch.arange(0., H, device=focal.device),
                            torch.arange(0., W, device=focal.device))
    direction = torch.stack([(xx-0.5*W)/focal, -(yy-0.5*H)/focal, -torch.ones_like(xx)], dim=-1) # (H, W, 3)
                                        
    rays_d = torch.einsum("hwc, nrc -> nhwr", direction, poses[:, :3, :3]) # (N, H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rays_o = poses[:, :3, -1] # (N, 3)
    rays_o = rays_o[:, None, None, :].expand_as(rays_d) # (N, H, W, 3)

    return rays_o, rays_d


def get_rays_tourism(H, W, kinv, pose):
    """
    phototourism camera intrinsics are defined by H, W and kinv.
    Args:
        H: image height
        W: image width
        kinv (3, 3): inverse of camera intrinsic
        pose (4, 4): camera extrinsic
    Returns:
        rays_o (H, W, 3): ray origins
        rays_d (H, W, 3): ray directions
    """
    yy, xx = torch.meshgrid(torch.arange(0., H, device=kinv.device),
                            torch.arange(0., W, device=kinv.device))
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
 
    directions = torch.matmul(pixco, kinv.T) # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # (H, W, 3)
    
    rays_o = pose[:3, -1].expand_as(rays_d) # (H, W, 3)

    return rays_o, rays_d
    

def sample_points(rays_o, rays_d, near, far, num_samples, perturb=False):
    """
    Sample points along the ray
    Args:
        rays_o (num_rays, 3): ray origins
        rays_d (num_rays, 3): ray directions
        near (float): near plane
        far (float): far plane
        num_samples (int): number of points to sample along each ray
        perturb (bool): if True, use randomized stratified sampling
    Returns:
        t_vals (num_rays, num_samples): sampled t values
        coords (num_rays, num_samples, 3): coordinate of the sampled points
    """
    num_rays = rays_o.shape[0]

    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    t_vals = t_vals.expand(num_rays, num_samples)   # t_vals has shape (num_samples)
                                                    # we must broadcast it to (num_rays, num_samples)
    if perturb:
        rand = torch.rand_like(t_vals) * (far-near)/num_samples
        t_vals = t_vals + rand

    coords = rays_o.unsqueeze(dim=-2) + t_vals.unsqueeze(dim=-1) * rays_d.unsqueeze(dim=-2)

    return t_vals, coords


def volume_render(rgbs, sigmas, t_vals, white_bkgd=False):
    """
    Volume rendering function.
    Args:
        rgbs (num_rays, num_samples, 3): colors
        sigmas (num_rays, num_samples): densities
        t_vals (num_rays, num_samples): sampled t values
        white_bkgd (bool): if True, assume white background
    Returns:
        color (num_rays, 3): color of the ray
    """
    # for phototourism, final delta is infinity to capture background
    # https://github.com/tancik/learnit/issues/4
    
    if white_bkgd:
        bkgd = 1e-3
    else:
        bkgd = 1e10

    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    delta_final = bkgd * torch.ones_like(deltas[:, -1:])
    deltas = torch.cat([deltas, delta_final], dim=-1) # (num_rays, num_samples)

    alphas = 1 - torch.exp(-deltas*sigmas)
    transparencies = torch.cat([
                                torch.ones_like(alphas[:, :1]),
                                torch.cumprod(1 - alphas[:, :-1] + 1e-10, dim=-1)
                                ], dim=-1)
    weights = alphas * transparencies # (num_rays, num_samples)
    
    # color = torch.sum(rgbs*weights.unsqueeze(-1), dim=-2)
    color = torch.einsum("rsc, rs -> rc", rgbs, weights)
    
    if white_bkgd:
        # composite the image to a white background
        color = color + 1 - weights.sum(dim=-1, keepdim=True)

    return color