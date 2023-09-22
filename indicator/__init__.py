import numpy as np
import pandas as pd

def calculate_PCAP(all_gt, all_detect):
    n_gt = all_gt.shape[0]
    n_detect = all_detect.shape[0]

    # sort by confidence
    all_detect_sorted = all_detect.sort_values('confidence', ascending=False)

    precision = []
    recall = []
    for i in range(n_detect):
        p = all_detect_sorted['is_TP'].iloc[:i + 1].sum() / (i + 1)
        precision.append(p)

        unique_gt = set(all_detect_sorted['whos_TP'].iloc[:i + 1].tolist())
        if None in unique_gt:
            unique_gt.remove(None)
        r = len(unique_gt) / n_gt
        recall.append(r)
    precision = np.array(precision)
    recall = np.array(recall)

    precision_adjust = precision.copy()
    for i in range(n_detect - 1, 0, -1):
        if precision_adjust[i] > precision_adjust[i - 1]:
            precision_adjust[i - 1] = precision_adjust[i]
    precision_adjust = np.hstack((precision_adjust[0], precision_adjust))
    recall_adjust = np.hstack(([0], recall))

    pcap = np.trapz(precision_adjust, recall_adjust)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(recall, precision, 10, 'red', '*')
    plt.plot(recall_adjust, precision_adjust, 'green')
    plt.title('sigma={:.2f}, pcap={:.6f}'.format(sigma, pcap))
    plt.grid('on')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

    return pcap

def judge_single_frame(frame_id, points_gt, points_detect, sigma):
    n_gt = points_gt.shape[0]
    n_detect = points_detect.shape[0]

    if n_gt > 0 and n_detect > 0:
        xyz_gt = points_gt[:, :3]
        xyz_detect = points_detect[:, :3]

        distance = np.sqrt(
            np.sum(
                np.power(
                    np.expand_dims(xyz_detect, axis=1) - np.expand_dims(xyz_gt, axis=0), 2
                ), axis=2
            )
        )

        distance_min = np.min(distance, axis=1)
        index_min = np.argmin(distance, axis=1)
        is_TP = (distance_min <= sigma)
        whos_TP = ['gt_frame{}_point{}'.format(frame_id, index_min[i]) if is_TP[i] else None for i in range(n_detect)]

        df_gt = pd.DataFrame({
            'name': ['gt_frame{}_point{}'.format(frame_id, i) for i in range(n_gt)],
            'frame_id': np.ones((n_gt,), dtype='int') * frame_id
        })

        df_detect = pd.DataFrame({
            'name': ['detect_frame{}_point{}'.format(frame_id, i) for i in range(n_detect)],
            'frame_id': np.ones((n_detect,), dtype='int') * frame_id,
            'is_TP': is_TP,
            'whos_TP': whos_TP,
            'confidence': points_detect[:, 3]
        })
    elif n_gt == 0 and n_detect > 0:
        df_gt = None
        df_detect = pd.DataFrame({
            'name': ['detect_frame{}_point{}'.format(frame_id, i) for i in range(n_detect)],
            'frame_id': np.ones((n_detect,), dtype='int') * frame_id,
            'is_TP': np.zeros((n_detect,), dtype='bool'),
            'whos_TP': [None for _ in range(n_detect)],
            'confidence': points_detect[:, 3]
        })
    elif n_gt > 0 and n_detect == 0:
        df_gt = pd.DataFrame({
            'name': ['gt_frame{}_point{}'.format(frame_id, i) for i in range(n_gt)],
            'frame_id': np.ones((n_gt,), dtype='int') * frame_id
        })
        df_detect = None
    else:
        df_gt = None
        df_detect = None

    return df_gt, df_detect


if __name__ == '__main__':
    import os

    sigma = 1
    root_data = '../data'

    all_detect = None
    all_gt = None
    frame_folders = os.listdir(root_data)
    frame_folders.sort()
    for i, frame_folder in enumerate(frame_folders):
        root_frame = os.path.join(root_data, frame_folder)

        data = np.load(os.path.join(root_frame, 'data.npz'))
        points_gt = data['points_gt']
        points_detect = data['points_detect']

        df_gt, df_detect = judge_single_frame(frame_id=i, points_gt=points_gt, points_detect=points_detect, sigma=sigma)

        if df_gt is not None:
            if all_gt is None:
                all_gt = df_gt
            else:
                all_gt = pd.concat([all_gt, df_gt])

        if df_detect is not None:
            if all_detect is None:
                all_detect = df_detect
            else:
                all_detect = pd.concat([all_detect, df_detect])

        print('process frame{}'.format(i))

    pcap = calculate_PCAP(all_gt, all_detect)


    print(pcap)

