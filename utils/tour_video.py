import copy
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from models.rendering import get_rays_tourism, sample_points, volume_render


def test_time_optimize(args, model, meta_state_dict, tto_view):
    """
    quicky optimize the meta trained model to a target appearance
    and return the corresponding network weights
    """
    
    model.load_state_dict(meta_state_dict)
    optim = torch.optim.SGD(model.parameters(), args.tto_lr)
    
    pixels = tto_view['img'].reshape(-1, 3)
    rays_o, rays_d = get_rays_tourism(tto_view['H'], tto_view['W'],
                                      tto_view['kinv'], tto_view['pose'])
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    num_rays = rays_d.shape[0]
    for step in range(args.tto_steps):
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, 
                                    tto_view['bound'][0], tto_view['bound'][1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()

    state_dict = copy.deepcopy(model.state_dict())
    return state_dict

        
def synthesize_view(args, model, H, W, kinv, pose, bound):
    """
    given camera intrinsics and camera pose, synthesize a novel view
    """
    rays_o, rays_d = get_rays_tourism(H, W, kinv, pose)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                args.num_samples, perturb=False)
    
    synth = []
    num_rays = rays_d.shape[0]
    with torch.no_grad():
        for i in range(0, num_rays, args.test_batchsize):
            rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
            color_batch = volume_render(rgbs_batch, sigmas_batch, t_vals[i:i+args.test_batchsize])
            synth.append(color_batch)
        synth = torch.cat(synth, dim=0).reshape(H, W, 3)
    return synth


def interpolate_views(args, model, prev, next):
    """
    generate new views by interpolating between aspect ratio, 
    focal length, camera pose and scene appearance of two views
    """
    synth_views = []
    for t in np.linspace(0, 1, 60):
        H = int(prev['H']*(1-t) + next['H']*t)
        W = int(prev['W']*(1-t) + next['W']*t)
        focal = prev['focal']*(1-t) + next['focal']*t
        kinv = torch.as_tensor([[1/focal, 0, -0.5*W/focal],
                                [0, -1/focal, 0.5*H/focal],
                                [0, 0, -1]], device=focal.device)
        pose = prev['pose']*(1-t) + next['pose']*t
        bound = prev['bound']*(1-t) + next['bound']*t
        state_dict = {name:
                    prev['state_dict'][name]*(1-t) + next['state_dict'][name]*t
                    for name in next['state_dict']
        }

        model.load_state_dict(state_dict)
        view = synthesize_view(args, model, H, W, kinv, pose, bound)
        synth_views.append(view.cpu().numpy())

    return synth_views


def resize_video(video_frames, max_H, max_W):
    """
    downsample the frames by two and pad the frame boundaries 
    to have the same shape
    Note: downsampling is done without anti-aliasing
    """
    max_H = max_H//2 + 5 # add a five pixel margin
    max_W = max_W//2 + 5

    padded_frames = []
    for frame in video_frames:
        frame = frame[::2, ::2]
        H, W, C = frame.shape

        # create a white canvas
        canvas = np.ones([max_H, max_W, C])
        # calculate center offset
        off_H, off_W = (max_H - H)//2, (max_W - W)//2
        # paste the frame in the center
        canvas[off_H:off_H+H, off_W:off_W+W] = frame

        padded_frames.append(canvas)    
    return padded_frames


def create_interpolation_video(args, model, meta_state_dict, test_set, device):
    """
    create interpolation video like the original project demo
    https://www.matthewtancik.com/learnit
    """
    heights = []
    widths = []

    view_idx = args.interpolate[0]
    img, pose, kinv, bound = test_set[view_idx]
    
    prev = {}
    prev['view_idx'] = view_idx
    prev['H'], prev['W'] = img.shape[:2]
    prev['img'], prev['kinv'] = img.to(device), kinv.to(device)
    prev['pose'], prev['bound'] = pose.to(device), bound.to(device)
    prev['focal'] = 1/prev['kinv'][0][0]
    prev['state_dict'] = test_time_optimize(args, model, meta_state_dict, prev)
                         
    heights.append(prev['H'])
    widths.append(prev['W'])

    video_frames = []
    for view_idx in args.interpolate[1:]:
        img, pose, kinv, bound = test_set[view_idx]

        next = {}
        next['view_idx'] = view_idx
        next['H'], next['W'] = img.shape[:2]
        next['img'], next['kinv'] = img.to(device), kinv.to(device)
        next['pose'], next['bound'] = pose.to(device), bound.to(device)
        next['focal'] = 1/next['kinv'][0][0]
        next['state_dict'] = test_time_optimize(args, model, meta_state_dict, next)

        synth_views = interpolate_views(args, model, prev, next)
        video_frames.extend(synth_views)
        
        print(f"test view {prev['view_idx']} and {next['view_idx']} interpolated")

        prev = next
        heights.append(prev['H'])
        widths.append(prev['W'])

    # resize frames
    max_H = max(heights)
    max_W = max(widths)
    video_frames = resize_video(video_frames, max_H, max_W)

    video_frames = np.stack(video_frames, axis=0)
    video_frames = (video_frames*255).astype(np.uint8)
    imageio.mimwrite("interpolation.mp4", video_frames, fps=30)

    return None
