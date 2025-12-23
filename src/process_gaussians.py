"""
Python script to load Gaussians from gsplat output
"""
import time
import torch
import numpy as np
import sys
import os
import hdbscan
from matplotlib import colormaps

def logit(x, eps=1e-6):
    """
    Inverse of signmoid function
    We manually need to pass opacity through a sigmoid to adjust it as we would expect,
    so this makes undoing that activation function easier
    
    Args:
        x (pytorch Tensor): Pytorch Tensor to pass through the function
        eps (float): epsilon to avoid div by zero
    """
    # clamp to avoid infs at 0 or 1
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def main(src: str="./results/splatted_new/ckpts/ckpt_6999_rank0.pt",
    dst: str="./modified_gaussians.pt"):
    """
    Main function to load gaussians, mask out outliers and try to isolate the scanned object
    The result is another .pt file that can be viewed with gsplat's simple_viewer

    Works as follows:
        Loads data from source
        Clusters data using HDBSCAN according to means and colors
        Sets opacity of outlier points to zero
        Identifies clusters with x and y standard deviations as 
            greater than THRESHOLD_FACTOR the z std dev as part of the table,
            and masks those out as well
        Saves the resulting splats to a destination file.

    Args:
        src (str): Source directory to load gaussians from
        dst (str): Destination directory to save gaussians into after modifying them
    """

    checkpoint_data = torch.load(src, map_location='cpu')
    print(f"checkpt data: {checkpoint_data}")
    gaussian_model = checkpoint_data['splats']
    print(f"splats: {gaussian_model}")

    cmap = colormaps.get_cmap("gist_rainbow")

    # Convert all relevant tensors to numpy arrays
    numpy_data = {}
    
    if 'means' in gaussian_model:
        # x y z points
        numpy_data['means'] = gaussian_model['means'].numpy() 
    if 'opacities' in gaussian_model:
        # scalar value for each gaussian
        numpy_data['opacities'] = gaussian_model['opacities'].numpy()
    if 'scales' in gaussian_model:
        # x y z scales
        numpy_data['scales'] = gaussian_model['scales'].numpy()
    if 'quats' in gaussian_model:
        # w x y z components
        numpy_data['quats'] = gaussian_model['quats'].numpy()
    if 'sh0' in gaussian_model:
        numpy_data['sh0'] = gaussian_model['sh0'].numpy()
    if 'shN' in gaussian_model:
        numpy_data['shN'] = gaussian_model['shN'].numpy()

    print(f"Numpy data: {numpy_data}")
    print(f"sh0 shape: {numpy_data['sh0'].shape}, shN shape: {numpy_data['shN'].shape}")

    data_in = np.concatenate([numpy_data['means'],numpy_data['sh0'].squeeze(1),numpy_data['sh0'].squeeze(1)],axis=1)
    print(f"Input data shape: {data_in.shape}")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=1000,cluster_selection_method="leaf")
    t0 = time.perf_counter()
    print(f"Fitting data")

    #cluster_amount = 100000
    #clusterer.fit(numpy_data['means'][0:cluster_amount,:])
    clusterer.fit(data_in[:,:])
    print(f"Fit data complete! Took {time.perf_counter() - t0} seconds")
    print(f"Num clusters: {clusterer.labels_.max()}")

    #pad_amount = numpy_data['means'].shape[0] - cluster_amount
    pad_amount = 0
    labels = np.pad(clusterer.labels_,(0,pad_amount),constant_values=-1)
    sh_C0 = 0.28209479177387814

    colors = np.zeros(shape=(clusterer.labels_.max(),3),dtype=np.float64)
    cmap_indices = np.array(range(clusterer.labels_.max()))
    np.random.shuffle(cmap_indices)

    CLUSTER_COLORS = False
    for label in range(clusterer.labels_.max()):
        color = np.array(cmap(cmap_indices[label]/clusterer.labels_.max())[0:3])
        label_indices = (labels == label)
        print(f"There are {sum(label_indices)} pts with label {label}")

        if CLUSTER_COLORS:
            numpy_data['sh0'][label_indices,0,:] = color / sh_C0
    
    # Mask out points that don't get matched
    label_indices = (labels == -1)
    print(f"There are {sum(label_indices)} pts without a cluster (outliers)")
    numpy_data['sh0'][label_indices,0,:] = 0.0

    #print(f"Colors: {colors}")

    opacity_active = torch.sigmoid(gaussian_model['opacities'])
    opacity_active_modified = 1.0 * opacity_active
    opacity_active_modified[label_indices] = 0.0
    opacity_modified = logit(opacity_active_modified)

    #color_red = np.array([1.0,0.0,0.0])/sh_C0
    #numpy_data['sh0'][:,0,:] = color_red
    numpy_data['shN'][:,:,:] = 0.0

    print(f"Opacity Modified: {opacity_modified.numpy()}")

    checkpoint_data['splats']['opacities'] = opacity_modified
    checkpoint_data['splats']['sh0'] = torch.from_numpy(numpy_data['sh0'])
    checkpoint_data['splats']['shN'] = torch.from_numpy(numpy_data['shN'])

    torch.save(checkpoint_data,dst)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # no parameters passed
        src = "./results/splatted_new/ckpts/ckpt_6999_rank0.pt"
        dst = "./modified_gaussians.pt"
    elif len(sys.argv) < 3:
        # source passed
        src = sys.argv[1]
        dst = "./modified_gaussians.pt"
    else:
        # both source and destination passed
        src = sys.argv[1]
        dst = sys.argv[2]
    main(src=src,dst=dst)