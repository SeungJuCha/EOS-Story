import torch
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
from skimage import filters
import scipy.ndimage


def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Function to apply Gaussian smoothing on each 2D slice of a 3D tensor.
    """
    kernel = np.fromfunction(
        lambda x, y: 1
        / (2 * np.pi * sigma**2)
        * np.exp(
            -((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)
            / (2 * sigma**2)
        ),
        (kernel_size, kernel_size),
    )
    kernel = (
        torch.Tensor(kernel / kernel.sum())
        .to(input_tensor.dtype)
        .to(input_tensor.device)
    )
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    smoothed_slices = []
    for i in range(input_tensor.size(0)):
        slice_tensor = input_tensor[i, :, :]
        slice_tensor = F.conv2d(
            slice_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2
        )[0, 0]
        smoothed_slices.append(slice_tensor)
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)
    return smoothed_tensor


def largest_component_mask(mask_np):
    N = mask_np.shape[0]
    res = int(np.sqrt(N))
    assert res * res == N, "mask length must be a perfect square"
    mask_2d = mask_np.reshape(res, res)
    (labeled, num) = scipy.ndimage.label(mask_2d.astype(np.int32))
    if num == 0:
        return mask_np
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    max_label = sizes.argmax()
    largest_mask_2d = labeled == max_label
    return largest_mask_2d.ravel()


def min_max_normalize(tensor, dim=-1, eps=1e-06):
    min_val = tensor.min(dim=dim, keepdim=True)[0]
    max_val = tensor.max(dim=dim, keepdim=True)[0]
    return (tensor - min_val) / (max_val - min_val + eps)


def otsu_mask(x: torch.Tensor, scaler=1.0, normalize=True):
    if normalize:
        x = min_max_normalize(x)
    x_np = x.detach().cpu().numpy()
    threshold_value = filters.threshold_otsu(x_np) * scaler
    binary_mask = (x_np > threshold_value).astype(np.uint8)
    binary_mask = largest_component_mask(binary_mask)
    return binary_mask


def cluster_mask_1d(x: torch.Tensor, n_clusters=2, normalize=True) -> torch.Tensor:
    """Cluster each one-dimensional tensor independently and keep the largest connected component of the cluster with the highest centroid."""
    if normalize:
        x = min_max_normalize(x)
    if x.dim() == 1:
        X = x.detach().cpu().view(-1, 1).numpy()
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        labels = km.labels_
        centers = km.cluster_centers_.flatten()
        fg_cluster = int(np.argmax(centers))
        mask = labels == fg_cluster
        mask = largest_component_mask(mask)
        return torch.from_numpy(mask)
    elif x.dim() == 2:
        masks = []
        for xi in x:
            X = xi.detach().cpu().view(-1, 1).numpy()
            km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            labels = km.labels_
            centers = km.cluster_centers_.flatten()
            fg_cluster = int(np.argmax(centers))
            mask = labels == fg_cluster
            mask = largest_component_mask(mask)
            masks.append(torch.from_numpy(mask))
        return torch.stack(masks, dim=0)
    else:
        raise ValueError("Input tensor x must be 1D or 2D.")
