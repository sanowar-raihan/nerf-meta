import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.phototourism import build_tourism
from models.nerf import build_nerf
from models.rendering import get_rays_tourism, sample_points, volume_render
from utils.tour_video import create_interpolation_video


def inner_loop(model, optim, img, rays_o, rays_d, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    pixels = img.reshape(-1, 3)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()


def report_result(model, img, rays_o, rays_d, bound, num_samples, raybatch_size):
    """
    report synthesis result on heldout view
    """
    pixels = img.reshape(-1, 3)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                num_samples, perturb=False)
    
    synth = []
    num_rays = rays_d.shape[0]
    with torch.no_grad():
        for i in range(0, num_rays, raybatch_size):
            rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
            color_batch = volume_render(rgbs_batch, sigmas_batch, t_vals[i:i+raybatch_size])
            synth.append(color_batch)
        synth = torch.cat(synth, dim=0)
        error = F.mse_loss(synth, pixels)
        psnr = -10*torch.log10(error)
    
    return psnr


def test():
    parser = argparse.ArgumentParser(description='phototourism with meta-learning')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the scene')
    parser.add_argument('--weight-path', type=str, required=True,
                        help='path to the meta-trained weight file')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_tourism(image_set="test", args=args)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    checkpoint = torch.load(args.weight_path, map_location=device)
    meta_state_dict = checkpoint['meta_model_state_dict']
    
    test_psnrs = []
    for idx, (img, pose, kinv, bound) in enumerate(test_loader):
        img, pose, kinv, bound = img.to(device), pose.to(device), kinv.to(device), bound.to(device)
        img, pose, kinv, bound = img.squeeze(), pose.squeeze(), kinv.squeeze(), bound.squeeze()
        rays_o, rays_d = get_rays_tourism(img.shape[0], img.shape[1], kinv, pose)
        
        # optimize on the left half, test on the right half
        left_width = img.shape[1]//2
        right_width = img.shape[1] - left_width
        tto_img, test_img = torch.split(img, [left_width, right_width], dim=1)
        tto_rays_o, test_rays_o = torch.split(rays_o, [left_width, right_width], dim=1)
        tto_rays_d, test_rays_d = torch.split(rays_d, [left_width, right_width], dim=1)

        model.load_state_dict(meta_state_dict)
        optim = torch.optim.SGD(model.parameters(), args.tto_lr)

        inner_loop(model, optim, tto_img, tto_rays_o, tto_rays_d,
                    bound, args.num_samples, args.tto_batchsize, args.tto_steps)
        
        psnr = report_result(model, test_img, test_rays_o, test_rays_d, bound, 
                            args.num_samples, args.test_batchsize)
        
        print(f"test view {idx+1}, psnr:{psnr:.3f}")
        test_psnrs.append(psnr)

    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")

    print("\ncreating interpolation video ...\n")
    create_interpolation_video(args, model, meta_state_dict, test_set, device)
    print("\ninterpolation video created!")


if __name__ == '__main__':
    test()