from gsplat.rendering import rasterization
from gsplat_colmap_interface import Parser, Dataset
from gsplat.cuda._wrapper import world_to_cam
#from gsplat.examples.gsplat_viewer import GsplatRenderTabState
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

width = 800
height = 600

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
require_grad = False
means = splats["means"].cuda()
means.requires_grad = require_grad
opacities = splats["opacities"].cuda()
opacities.requires_grad = require_grad

colors0 = splats["sh0"].cuda()
colorsN = splats["shN"].cuda()
colors = torch.cat([colors0,colorsN],dim=1)

quats = splats["quats"].cuda()
quats.requires_grad = require_grad
scales = splats["scales"].cuda()
scales.requires_grad = require_grad
print(means.shape)
print(opacities.shape)
print(colors.shape)
print(quats.shape)
print(scales.shape)

print("input colors:", colors.min().item(), colors.max().item())

## We need to first get the scene normalization for our train images
data_dir = "images_estop_2/"
parser = Parser(data_dir=data_dir,normalize=True)
valset = Dataset(parser=parser,split="val")
colmap_to_normal = parser.transform
camtoworlds = parser.camtoworlds
valloader = torch.utils.data.DataLoader(
    valset,batch_size=1,shuffle=False,num_workers=1
)

for i, data in enumerate(valloader):
    camtoworlds = data["camtoworld"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device) / 255.0
    masks = data["mask"].to(device) if "mask" in data else None
    height, width = pixels.shape[1:3]
    viewmats = torch.linalg.inv(camtoworlds) # world to camrender

    torch.cuda.synchronize()
    print(f"Ks: {Ks}")


#for key in parser.Ks_dict.keys():
#    Ks_dict = parser.Ks_dict[key]
#Ks_dict = torch.tensor(Ks_dict,device=device)

# define cameras
#viewmats = torch.eye(4, device=device)[None, :]
#print(viewmats.shape)
#Ks = torch.tensor([
#   [fx/100, 0., cx], [0., fy/100, cy], [0., 0., 1.]], device=device)[None, :, :]

#viewmats = torch.tensor(pose, device=device)[None, :]
#cam_id = parser.camera_ids[count]           # assuming count indexes same ordering
#K = parser.Ks_dict[cam_id].astype(np.float32)
#w2c_np = np.linalg.inv(parser.camtoworlds[cam_id]).astype(np.float32)
#viewmats = torch.from_numpy(w2c_np).to(device=device).unsqueeze(0)

#Ks = torch.from_numpy(K).to(device=device)      # (1,3,3)
# width, height = parser.imsize_dict[cam_id]

# width_full  = cam_params["width"]
# height_full = cam_params["height"]

# W = 68
# H = 40
# sx = W / width_full
# sy = H / height_full

# K = np.array([[fx*sx, 0, cx*sx],
#               [0, fy*sy, cy*sy],
#               [0, 0, 1]], dtype=np.float32)

# Ks = torch.from_numpy(K)[None].cuda()
# width, height = W, H

# # Apply normalization from gsplat
# new_viewmats = viewmats
# print(f"Colmap to normal: {colmap_to_normal}")

#for idx in range(viewmats.shape[0]):
#    viewmat = viewmats[idx].cpu().numpy() @ colmap_to_normal
#    new_viewmats[idx] = torch.tensor(viewmat,device=device)

downsample = 2
height = height // downsample
width = width // downsample
print(f"Ks: {Ks} with shape {Ks.shape}")
Ks_ds = Ks.clone()
Ks_ds[:, 0, :] /= downsample
Ks_ds[:, 1, :] /= downsample

assert pixels.ndim == 4 and pixels.shape[-1] == 3, pixels.shape

H, W = pixels.shape[1], pixels.shape[2]
H2, W2 = H // downsample, W // downsample
print(f"H: {H}, W: {W}, H2: {H2}, W2: {W2}")

pixels_nchw = pixels.permute(0, 3, 1, 2).contiguous()          # [1, 3, H, W]
downscaled_nchw = torch.nn.functional.interpolate(pixels_nchw, size=(H2, W2), mode="area")

if "mask" in data:
    downscaled_mask = torch.nn.functional.interpolate(masks,size=(H2,W2),mode='nearest')
    downscaled_mask = downscaled_mask.permute(0,2,3,1).contiguous()

downscaled_pixels = downscaled_nchw.permute(0, 2, 3, 1).contiguous()  # [1, H2, W2, 3]

#print(viewmats.shape)
#print(Ks.shape)

print(f"Ks: {Ks_ds}")
print(f"Viewmat: {viewmats}")

print(f"downscaled pixels shape: {downscaled_pixels.shape}")
if "mask" in data:
    print(f"Masks shape: {masks.shape}")
# render
render_colors, render_alphas, meta = rasterization(
    means, quats, torch.exp(scales), torch.sigmoid(opacities), colors, viewmats, Ks_ds[:], W2, H2, render_mode='RGB',
    sh_degree=3
)
print("alpha min/max/mean:", render_alphas.min().item(), render_alphas.max().item(), render_alphas.mean().item())
if masks is not None:
    render_colors[~masks] = 0
#print(f"Render colors: {render_colors}")
#C = render_colors.shape[0]
print(f"Colors after raster: {render_colors}")
print(f" downscaled pixels shape: {downscaled_pixels.shape}, render color shape: {render_colors.shape}")

error_im = torch.abs(downscaled_pixels - render_colors) * 5
canvas_list = [downscaled_pixels,render_colors,error_im]

print(canvas_list)
canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
print(f"canvas shape: {canvas.shape}")
canvas = (canvas * 255).astype(np.uint8)

# print(f"Height {height} and width {width}")
# canvas = (
#     torch.cat(
#         [
#             render_rgbs.reshape(C * height, width, 3),
#             #render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
#             #render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
#         ],
#         dim=1
#     )
#     #.detach()
#     .cpu()
#     .numpy()
# )
# #im = (canvas*255).astype(np.uint8)
im = canvas
#print(canvas.shape)
#tensor_cpu = render_rgbs.cpu()
#im = tensor_cpu.detach().numpy()
drawn = ax.imshow(im)
#print(im)
plt.show()