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
    # clamp to avoid infs at 0 or 1
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

if __name__ == "__main__":
    #src = "../../gsplat/examples/results/personhall_downsample/ckpts/ckpt_29999_rank0.pt"
    src = "./results/estop_2/ckpts/ckpt_6999_rank0.pt"
    dst = "./modified_gaussians.pt"

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

    data_in = np.concatenate([numpy_data['means'],10*numpy_data['sh0'].squeeze(1),numpy_data['sh0'].squeeze(1)],axis=1)
    print(f"Input data shape: {data_in.shape}")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=1000,min_samples=10,cluster_selection_method="leaf")
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
    valid_labels = []

    for label in range(clusterer.labels_.max()):
        color = np.array(cmap(cmap_indices[label]/clusterer.labels_.max())[0:3])
        label_indices = (labels == label)
        print(f"There are {sum(label_indices)} pts with label {label}")
        #numpy_data['sh0'][label_indices,0,:] = color / sh_C0

        # We want to filter out the ground to isolate our object. 
        # To do this, we look at clusters with more horizontal std dev than vertical
        x_vals = numpy_data['means'][label_indices,0]
        y_vals = numpy_data['means'][label_indices,1]
        z_vals = numpy_data['means'][label_indices,2]
        x_std = np.std(x_vals)
        y_std = np.std(y_vals)
        z_std = np.std(z_vals)

        THRESHOLD_FACTOR = 5
        if not(z_std * THRESHOLD_FACTOR < x_std and z_std * THRESHOLD_FACTOR < y_std):
            valid_labels.append(label)


    # Mask out points that don't get matched
    label_indices = (labels == -1)
    print(f"There are {sum(label_indices)} pts without a cluster (outliers)")

    bad_labels = [i for i in range(clusterer.labels_.max()) if i not in valid_labels]
    for bad_label in bad_labels:
        label_indices = label_indices | (labels == bad_label)

    numpy_data['sh0'][label_indices,0,:] = 0.0

    #print(f"Colors: {colors}")

    opacity_active = torch.sigmoid(gaussian_model['opacities'])
    opacity_active_modified = 1.0 * opacity_active
    opacity_active_modified[label_indices] = 0.0
    opacity_modified = logit(opacity_active_modified)

    #color_red = np.array([1.0,0.0,0.0])/sh_C0
    #numpy_data['sh0'][:,0,:] = color_red
    #numpy_data['shN'][:,:,:] = 0.0

    print(f"Opacity Modified: {opacity_modified.numpy()}")

    checkpoint_data['splats']['opacities'] = opacity_modified
    checkpoint_data['splats']['sh0'] = torch.from_numpy(numpy_data['sh0'])
    checkpoint_data['splats']['shN'] = torch.from_numpy(numpy_data['shN'])

    torch.save(checkpoint_data,'modified_pt.pt')