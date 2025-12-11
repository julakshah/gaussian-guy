from gsplat.rendering import rasterization
import numpy as np
import torch
import time 
import matplotlib.pyplot as plt
import os

def quat_to_rotation(w,x,y,z):
    """
    Convert quaternion to rotation matrix:
    """
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ],dtype=np.float32)
    return R

def read_camera_txt(path):
    """
    Gets the camera intrinsics from COLMAP's representation of them
    Assumes only one camera is used
    """
    out = {}
    with (open(path,'r')) as f:
        for line_raw in f:
            line = line_raw.strip()
            if line[0] != '#':
                strs = line.split(' ')
                out["id"] = int(strs[0])
                out["model"] = strs[1]
                out["width"] = int(strs[2])
                out["height"] = int(strs[3])
                out["params"] = [float(x) for x in strs[4:]]
                print(f"Got camera params: {out}")
                return out

def read_frames(path):
    """
    Gets the image poses from COLMAP's representation of them
    """
    out = {}
    extensions = [".jpg",".png",".jpeg"]
    with (open(path,'r')) as f:
        for line_raw in f:
            line = line_raw.strip()
            if line[0] != '#':
                strs = line.split(' ')
                # The lines with the image pose have the image name at the end
                # so we see if they end in a file extension
                truths = [ext in strs[-1].lower() for ext in extensions]
                if sum(truths):
                    out[strs[-1]] = [float(x) for x in strs[:-2]] # ignore last two fields (cam ID and image name)
    print(f"Frames read: {out}")
    return out

fig,ax = plt.subplots()

src = "results/estop_2/ckpts/ckpt_6999_rank0.pt"
checkpoint_data = torch.load(src, map_location='cpu')
print(f"checkpt data: {checkpoint_data}")
splats = checkpoint_data["splats"]

cam_src = "images_estop_2_txt/cameras.txt"
images_src = "images_estop_2_txt/images.txt"
cam_params = read_camera_txt(cam_src)
im_poses = read_frames(images_src)

# for simple radial camera:
fx = cam_params['params'][0]
fy = fx
cx = cam_params['params'][1]
cy = cam_params['params'][2]
width = cam_params['width']
height = cam_params['height']

# frame pose 1
keys = list(im_poses.keys())
count = 2
w,x,y,z = im_poses[keys[count]][1:5]
tx,ty,tz = im_poses[keys[count]][5:8]
pose = np.eye(4,dtype=np.float32)
pose[:3,:3] = quat_to_rotation(w=w,x=x,y=y,z=z)
pose[:3,3] = [tx,ty,tz]

# define Gaussians
device = 0
#means = torch.randn((100, 3), device=device)
#quats = torch.randn((100, 4), device=device)
#scales = torch.rand((100, 3), device=device) * 0.1
#colors = torch.rand((100, 3), device=device)
#opacities = torch.rand((100,), device=device) 
means = splats["means"].cuda()
means.requires_grad = True
opacities = splats["opacities"].cuda()
opacities.requires_grad = True
colors = splats["sh0"][:,0,:].cuda()
colors.requires_grad = True
quats = splats["quats"].cuda()
quats.requires_grad = True
scales = splats["scales"].cuda()
scales.requires_grad = True
print(means.shape)
print(opacities.shape)
print(colors.shape)
print(quats.shape)
print(scales.shape)

print("input colors:", colors.min().item(), colors.max().item())

# define cameras
#viewmats = torch.eye(4, device=device)[None, :]
#print(viewmats.shape)
Ks = torch.tensor([
   [fx, 0., cx], [0., fy, cy], [0., 0., 1.]], device=device)[None, :, :]
viewmats = torch.tensor(pose, device=device)[None, :]
print(viewmats.shape)
print(Ks.shape)
# render
render_colors, render_alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height, render_mode='RGB+D'
)

C = render_colors.shape[0]
assert render_colors.shape == (C, height, width, 4)
assert render_alphas.shape == (C, height, width, 1)
render_colors.sum().backward()

render_rgbs = render_colors[..., 0:3]
render_depths = render_colors[..., 3:4]
render_depths = render_depths / render_depths.max()

canvas = (
    torch.cat(
        [
            render_rgbs.reshape(C * height, width, 3),
            #render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
            #render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
        ],
        dim=1
    )
    .detach()
    .cpu()
    .numpy()
)
im = (canvas*255).astype(np.uint8)
print(canvas.shape)
#tensor_cpu = render_rgbs.cpu()
#im = tensor_cpu.detach().numpy()
drawn = ax.imshow(im)
print(im)
plt.show()