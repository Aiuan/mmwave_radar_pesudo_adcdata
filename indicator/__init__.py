import numpy as np
def calculate_PCAP(pcd_gt, pcd_detect, confidence, sigma):
    # pointcloud average precision

    n_detect = len(pcd_detect['x'])
    n_gt = len(pcd_gt['x'])

    xyz_gt = np.stack((pcd_gt['x'], pcd_gt['y'], pcd_gt['z'])).T
    xyz = np.stack((pcd_detect['x'], pcd_detect['y'], pcd_detect['z'])).T

    # sort by confidence
    index_sorted = np.argsort(confidence)[::-1]
    xyz_sorted = xyz[index_sorted, :]

    distance = np.sqrt(
        np.sum(
            np.power(
                np.expand_dims(xyz_sorted, axis=1) - np.expand_dims(xyz_gt, axis=0), 2
            ), axis=2
        )
    )

    distance_min = np.min(distance, axis=1)
    index_min = np.argmin(distance, axis=1)
    is_neighbor = (distance_min <= sigma)
    whos_neighbor = [index_min[i] if is_neighbor[i] else None for i in range(n_detect)]

    precision = []
    recall = []
    for i in range(n_detect):
        p = is_neighbor[:i+1].sum() / (i+1)
        precision.append(p)

        unique_gt = set(whos_neighbor[:i+1])
        unique_gt.remove(None)
        r = len(unique_gt) / n_gt
        recall.append(r)
    precision = np.array(precision)
    recall = np.array(recall)

    precision_adjust = precision.copy()
    for i in range(n_detect - 1, 0, -1):
        if precision_adjust[i] > precision_adjust[i-1]:
            precision_adjust[i - 1] = precision_adjust[i]
    precision_adjust = np.hstack((precision_adjust[0], precision_adjust))
    recall_adjust = np.hstack(([0], recall))

    pcap = np.trapz(precision_adjust, recall_adjust)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(recall, precision, 'red')
    plt.plot(recall_adjust, precision_adjust, 'green')
    plt.title('sigma={:.2f}, pcap={:.6f}'.format(sigma, pcap))
    plt.grid('on')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

    return pcap


if __name__ == '__main__':
    pcd_gt = {
        'x': np.array([-4, -8, -5, -12, -7, 0, 12, 9, 11, 21, 13], dtype='float'),
        'y': np.array([3, 6, 12, 16, 24, 30, 35, 40, 60, 72, 84], dtype='float'),
        'z': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float'),
        'velocity': np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], dtype='float'),
        'intensity': np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype='float')
    }

    pcd_detect = {
        'x': np.array([-8, -5, -12, -7, 0, 13, 11, 14, 25, 18], dtype='float'),
        'y': np.array([6, 12, 16, 24, 30, 35, 40, 60, 72, 84], dtype='float'),
        'z': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float'),
        'velocity': np.array([-8, -6, -4, -2, 0, 1, 2, 3, 4, 5], dtype='float'),
        'intensity': np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype='float'),
    }

    confidence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float')

    pcap = calculate_PCAP(pcd_gt, pcd_detect, confidence, sigma=0.1)

    print('done')

