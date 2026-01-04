import numpy as np
import matplotlib.pyplot as plt

def map_loader(map_path):

        print("Loading congestion map from:", map_path)
        congestion_map = np.load(map_path)
        congestion_map = congestion_map.squeeze()
        return congestion_map

def MAx_Pool2d(input_array, pool_size, stride):

    H, W = input_array.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output_array = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size

            output_array[i, j] = np.max(input_array[h_start:h_end, w_start:w_end])

    return output_array, output_array.shape

def visualize_map(initial_data, processed_data):

    diff = processed_data - initial_data

    print("Max abs diff:", np.max(np.abs(diff)))
    print("Mean abs diff:", np.mean(np.abs(diff)))
    print("Non-zero count (>1e-8):", np.sum(np.abs(diff) > 1e-8))


    vmin = initial_data.min()
    vmax = initial_data.max()

    diff = processed_data - initial_data
    diff_abs_max = np.max(np.abs(diff))

    fig, axs = plt.subplots(
        1, 3,
        figsize=(18, 6),
        constrained_layout=True
    )

    im0 = axs[0].imshow(
        initial_data.squeeze(),
        cmap="hot",
        vmin=vmin,
        vmax=vmax
    )
    axs[0].set_title("Original Congestion Map")
    axs[0].axis("off")

    im1 = axs[1].imshow(
        processed_data.squeeze(),
        cmap="hot",
        vmin=vmin,
        vmax=vmax
    )
    axs[1].set_title("Processed Congestion Map")
    axs[1].axis("off")

    im2 = axs[2].imshow(
        diff.squeeze(),
        cmap="bwr",
        vmin=-diff_abs_max,
        vmax=diff_abs_max
    )
    axs[2].set_title("Difference Map")
    axs[2].axis("off")

    cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Congestion Intensity")

    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Congestion Intensity")

    cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Δ Congestion")

    plt.show()

def reduce_congestion(congestion_map,visualize=True,alpha=1):

    alpha = 1
    pool_size = 7
    stride = 3

    max_pool_output, max_pool_shape = MAx_Pool2d(congestion_map, pool_size, stride)
    order = max_pool_shape[0]
    map_copy = congestion_map.copy()
    H, W = congestion_map.shape
 
    neighbour_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),(-1,-1),(-1,1),(1,-1),(1,1)]
    updates = 0
    delta = 0.005


    for z in range(0,order):
        for r in range(0,order):

            over_threshold = np.exp(alpha*max_pool_output[z, r]-1) / (np.e - 1)

            for x in range(stride* z, min(stride* z + pool_size, H)):
                for y in range(stride* r, min(stride* r + pool_size, W)):

                    value = congestion_map[x, y]          
                    excess = value - over_threshold
                    if excess <= 0:
                        continue
                    residue = excess
                    distribution_loc = []

                    for dx, dy in neighbour_offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if (congestion_map.min()) < congestion_map[nx, ny] < over_threshold:
                                distribution_loc.append((nx, ny))

                    if not distribution_loc:
                        continue

                    share = excess / len(distribution_loc)

                    for nx, ny in distribution_loc:
                        capacity = (1+delta)*over_threshold - congestion_map[nx, ny]
                        delta_neighbor = min(share, capacity)

                        if delta_neighbor > 0:
                            congestion_map[nx, ny] += delta_neighbor
                            residue -= delta_neighbor
                            updates += 1

                    congestion_map[x, y] = over_threshold + residue

    if visualize:
        visualize_map(map_copy, congestion_map)

    return congestion_map

def evaluate_congestion(initial, processed):

    metrics = {}

    metrics["peak_before"] = initial.max()
    metrics["peak_after"] = processed.max()
    metrics["peak_reduction_ratio"] = (
        metrics["peak_before"] - metrics["peak_after"]
    ) / metrics["peak_before"]

    overflow_th = 0.9 * metrics["peak_before"]
    metrics["overflow_area_before"] = np.mean(initial > overflow_th)
    metrics["overflow_area_after"] = np.mean(processed > overflow_th)
    metrics["overflow_area_reduction"] = (
        metrics["overflow_area_before"] - metrics["overflow_area_after"]
    )

    metrics["overflow_volume_before"] = np.sum(
        np.maximum(0, initial - overflow_th)
    )
    metrics["overflow_volume_after"] = np.sum(
        np.maximum(0, processed - overflow_th)
    )
    metrics["overflow_volume_reduction"] = (
        metrics["overflow_volume_before"] - metrics["overflow_volume_after"]
    )

    metrics["mean_abs_change"] = np.mean(np.abs(processed - initial))
    metrics["changed_area_ratio"] = np.mean(
        np.abs(processed - initial) > 1e-6
    )

    gx_i, gy_i = np.gradient(initial)
    gx_p, gy_p = np.gradient(processed)
    metrics["grad_energy_before"] = np.mean(gx_i**2 + gy_i**2)
    metrics["grad_energy_after"] = np.mean(gx_p**2 + gy_p**2)
    metrics["grad_energy_reduction"] = (
        metrics["grad_energy_before"] - metrics["grad_energy_after"]
    )

    metrics["mass_error_ratio"] = abs(
        processed.sum() - initial.sum()
    ) / initial.sum()

    return metrics


def process_map(map_path,visualize=True,alpha=1):

    print("\nProcessing congestion map with normalized congestion map")
    congestion_map = map_loader(map_path)
    congestion_map_initial = congestion_map.copy()

    while True:
        prev = congestion_map.copy()
        congestion_map = reduce_congestion(congestion_map,visualize=False, alpha=alpha)

        if np.allclose(prev, congestion_map, atol=1e-8):
            break
    metrics=evaluate_congestion(congestion_map_initial, congestion_map)
        

    if visualize: visualize_map(congestion_map_initial, congestion_map)
    print("processing complete!!")
    return congestion_map,congestion_map_initial,metrics

process_map(r"D:\training_set\congestion\label\19-RISCY-a-1-c2-u0.7-m3-p6-f0.npy",visualize=True,alpha=1)