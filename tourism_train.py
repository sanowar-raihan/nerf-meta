import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.phototourism import build_tourism
from models.nerf import build_nerf
from models.rendering import get_rays_tourism, sample_points, volume_render


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


def train_meta(args, meta_model, meta_optim, data_loader, device):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """
    for img, pose, kinv, bound in data_loader:
        img, pose, kinv, bound = img.to(device), pose.to(device), kinv.to(device), bound.to(device)
        img, pose, kinv, bound = img.squeeze(), pose.squeeze(), kinv.squeeze(), bound.squeeze()
        rays_o, rays_d = get_rays_tourism(img.shape[0], img.shape[1], kinv, pose)

        meta_optim.zero_grad()

        inner_model = copy.deepcopy(meta_model)
        inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)

        inner_loop(inner_model, inner_optim, img, rays_o, rays_d, bound,
                    args.num_samples, args.train_batchsize, args.inner_steps)
        
        with torch.no_grad():
            for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                meta_param.grad = meta_param - inner_param
        
        meta_optim.step()


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


def val_meta(args, model, val_loader, device):
    """
    validate the meta trained model for phototourism
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    
    val_psnrs = []
    for img, pose, kinv, bound in val_loader:
        img, pose, kinv, bound = img.to(device), pose.to(device), kinv.to(device), bound.to(device)
        img, pose, kinv, bound = img.squeeze(), pose.squeeze(), kinv.squeeze(), bound.squeeze()
        rays_o, rays_d = get_rays_tourism(img.shape[0], img.shape[1], kinv, pose)
        
        # optimize on the left half, test on the right half
        left_width = img.shape[1]//2
        right_width = img.shape[1] - left_width
        tto_img, test_img = torch.split(img, [left_width, right_width], dim=1)
        tto_rays_o, test_rays_o = torch.split(rays_o, [left_width, right_width], dim=1)
        tto_rays_d, test_rays_d = torch.split(rays_d, [left_width, right_width], dim=1)

        val_model.load_state_dict(meta_trained_state)
        val_optim = torch.optim.SGD(val_model.parameters(), args.inner_lr)

        inner_loop(val_model, val_optim, tto_img, tto_rays_o, tto_rays_d,
                    bound, args.num_samples, args.train_batchsize, args.inner_steps)
        
        psnr = report_result(val_model, test_img, test_rays_o, test_rays_d, bound, 
                                args.num_samples, args.test_batchsize)
        val_psnrs.append(psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def main():
    parser = argparse.ArgumentParser(description='phototourism with meta-learning')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the scene')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_tourism(image_set="train", args=args)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_tourism(image_set="val", args=args)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    meta_model = build_nerf(args)
    meta_model.to(device)

    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    
    for epoch in range(1, args.meta_epochs+1):
        train_meta(args, meta_model, meta_optim, train_loader, device)
        val_psnr = val_meta(args, meta_model, val_loader, device)
        print(f"Epoch: {epoch}, val psnr: {val_psnr:0.3f}")

        torch.save({
            'epoch': epoch,
            'meta_model_state_dict': meta_model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, f'meta_epoch{epoch}.pth')


if __name__ == '__main__':
    main()