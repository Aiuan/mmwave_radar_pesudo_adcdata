import numpy as np
import matplotlib.pyplot as plt

from TIRadar import calculate_radar_parameters
from TIRadar.signal_process import NormalModeProcess
from utils import pcd_in_polar_zone, load_json
from pesudoRadarAdcdata import pcd2adcdata

def determine_scope(data, alpha, beta):
    v_max = data.max()
    v_min = data.min()
    v_zone = v_max - v_min

    v_scope_min = v_min + alpha * v_zone
    v_scope_max = v_max - beta * v_zone
    return v_scope_min, v_scope_max

def main():
    n_row = 2
    n_col = 3
    w = 400
    h = w / 2
    w_padding = 100
    h_padding_top = 50
    h_padding_mid = 150
    h_padding_bottom = 100
    h_colorbar = 10
    h_padding_colorbar = 45
    w_all = w * n_col + (n_col + 1) * w_padding
    h_all = h * n_row + h_padding_top + h_padding_mid + h_padding_bottom

    dpi_vedio = 100
    fig = plt.figure(figsize=(w_all / dpi_vedio, h_all / dpi_vedio), facecolor='white')
    plt.tight_layout()

    cmap = 'jet'
    doppler_min, doppler_max = -5, 5
    lidar_intensity_min, lidar_intensity_max = 0, 255
    r_ignore = 5

    # Hypothetical target point
    pcd = {
        'x': np.array([-4, -8, -5, -12, -7, 0, 12, 9, 11, 21, 13], dtype='float'),
        'y': np.array([3, 6, 12, 16, 24, 30, 35, 40, 60, 72, 84], dtype='float'),
        'z': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float'),
        'intensity': np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype='float'),
        'velocity': np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], dtype='float'),
    }
    pcd = pcd_in_polar_zone(pcd, return_type='dict', add_aer=True)

    mode_infos = load_json('./radar_mode_infos.json')
    radar_params = calculate_radar_parameters(mode_infos)

    pesudo_radar_adcdata = pcd2adcdata(pcd, radar_params)

    nmp = NormalModeProcess(
        mode_infos,
        pesudo_radar_adcdata.real, pesudo_radar_adcdata.imag,
        apply_vmax_extend=False
    )

    nmp.run(
        generate_pcd=True, generate_heatmapBEV=True, generate_heatmap4D=False,
        cfar_type='CAOS', calib_on=False, re_order_on=False
    )

    pcd_detect = nmp.get_pcd()
    heatmapBEV = nmp.get_heatmapBEV()

    sig_integrate_dB = 10 * np.log10(nmp.sig_integrate)
    range_bins = nmp.get_range_bins()
    doppler_bins = nmp.get_doppler_bins()
    res_rdm_cfar = nmp.res_rdm_cfar
    rlim = mode_infos['range_max_m']
    v_min_sig_integrate_dB, v_max_sig_integrate_dB = determine_scope(sig_integrate_dB, alpha=0.2, beta=0.2)

    idx_range_start = np.where(range_bins <= r_ignore)[0].max() + 1
    idx_range_end = np.where(range_bins >= mode_infos['range_max_m'] - r_ignore)[0].min()
    heatmapBEV_static_dB = 10 * np.log10(heatmapBEV['heatmapBEV_static'][idx_range_start:idx_range_end, :])
    heatmapBEV_dynamic_dB = 10 * np.log10(heatmapBEV['heatmapBEV_dynamic'][idx_range_start:idx_range_end, :])
    heatmapBEV_x = heatmapBEV['x'][idx_range_start:idx_range_end, :]
    heatmapBEV_y = heatmapBEV['y'][idx_range_start:idx_range_end, :]
    v_min_heatmapBEV_static_dB, v_max_heatmapBEV_static_dB = determine_scope(heatmapBEV_static_dB, alpha=0.2,
                                                                             beta=0.2)
    v_min_heatmapBEV_dynamic_dB, v_max_heatmapBEV_dynamic_dB = determine_scope(heatmapBEV_dynamic_dB, alpha=0.2,
                                                                               beta=0.2)

    # Range-Doppler Curves
    left1 = w_padding / w_all
    bottom1 = (h_padding_mid + h + h_padding_bottom) / h_all
    w1 = w / w_all
    h1 = h / h_all
    ax1 = fig.add_axes([left1, bottom1, w1, h1])
    for idx_doppler in range(sig_integrate_dB.shape[1]):
        if idx_doppler == sig_integrate_dB.shape[1] / 2:
            # v=0
            ax1.plot(np.arange(range_bins.shape[0]), sig_integrate_dB[:, idx_doppler], c='black', linewidth=2)
        else:
            # v!=0
            ax1.plot(np.arange(range_bins.shape[0]), sig_integrate_dB[:, idx_doppler], linewidth=1)
    # mark res_rdm_cfar
    for idx_doppler in range(sig_integrate_dB.shape[1]):
        if idx_doppler in res_rdm_cfar['doppler_index']:
            idxs_range = res_rdm_cfar['range_index'][np.where(idx_doppler == res_rdm_cfar['doppler_index'])[0]]
            ax1.scatter(
                idxs_range, sig_integrate_dB[idxs_range, idx_doppler],
                s=40, marker='o', c='none', edgecolor='red', linewidths=2
            )
    ax1.set_xlim([0, range_bins.shape[0]])
    ax1.set_ylim([np.floor(sig_integrate_dB.min()), np.ceil(sig_integrate_dB.max())])
    ax1.set_xlabel('range_index')
    ax1.set_ylabel('receive power/dB')
    ax1.grid(color='gray', linestyle='-', linewidth=1)
    ax1.set_aspect('auto')
    ax1.set_title('Range-Doppler Curves')

    # Range-Doppler Map
    left2 = left1 + w1 + w_padding / w_all
    bottom2 = (h_padding_mid + h + h_padding_bottom) / h_all
    w2 = w / w_all
    h2 = h / h_all
    ax2 = fig.add_axes([left2, bottom2, w2, h2])
    ax2.imshow(sig_integrate_dB.T, cmap=cmap, vmin=v_min_sig_integrate_dB, vmax=v_max_sig_integrate_dB)
    n_xticks = 7
    idx_xticks = np.round(np.linspace(0, range_bins.shape[0] - 1, n_xticks)).astype('int')
    ax2.set_xticks(idx_xticks)
    ax2.set_xticklabels(['{:.2f}'.format(r) for r in range_bins[idx_xticks]])
    n_yticks = 7
    idx_yticks = np.round(np.linspace(0, doppler_bins.shape[0] - 1, n_yticks)).astype('int')
    ax2.set_yticks(idx_yticks)
    ax2.set_yticklabels(['{:.2f}'.format(v) for v in doppler_bins[idx_yticks]])
    ax2.set_xlabel('range/m')
    ax2.set_ylabel('doppler/m_s')
    ax2.set_aspect('equal')
    ax2.set_title('Range-Doppler Map')
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,
                              norm=plt.Normalize(vmin=v_min_sig_integrate_dB, vmax=v_max_sig_integrate_dB)),
        cax=fig.add_axes(
            [left2, bottom2 - h_padding_colorbar / h_all - h_colorbar / h_all, w2, h_colorbar / h_all]),
        orientation='horizontal', label='power(dB)'
    )

    # radarPcd
    left3 = w_padding / w_all
    bottom3 = h_padding_bottom / h_all
    w3 = w / w_all
    h3 = h / h_all
    ax3 = fig.add_axes([left3, bottom3, w3, h3])
    if pcd_detect is not None:
        ax3.scatter(
            pcd_detect['x'], pcd_detect['y'],
            s=np.round(pcd_detect['snr']),
            c=np.round(pcd_detect['doppler']),
            cmap=cmap, vmin=doppler_min, vmax=doppler_max
        )
        ax3.set_xlim([-rlim, rlim])
        ax3.set_ylim([0, rlim])
        ax3.set_xlabel('x/m')
        ax3.set_ylabel('y/m')
        ax3.grid(color='gray', linestyle='-', linewidth=1)
        ax3.set_aspect('equal')
        ax3.set_title('radarPcd')
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=doppler_min, vmax=doppler_max)),
            cax=fig.add_axes(
                [left3, bottom3 - h_padding_colorbar / h_all - h_colorbar / h_all, w3, h_colorbar / h_all]),
            orientation='horizontal', label='doppler(m/s)'
        )

    # Lidarpcd
    left4 = left3 + w3 + w_padding / w_all
    bottom4 = h_padding_bottom / h_all
    w4 = w / w_all
    h4 = h / h_all
    ax4 = fig.add_axes([left4, bottom4, w4, h4])
    ax4.scatter(
        pcd['x'], pcd['y'],
        s=1,
        c=np.round(pcd['intensity']),
        cmap=cmap, vmin=lidar_intensity_min, vmax=lidar_intensity_max
    )
    ax4.set_xlim([-rlim, rlim])
    ax4.set_ylim([0, rlim])
    ax4.set_xlabel('x/m')
    ax4.set_ylabel('y/m')
    ax4.grid(color='gray', linestyle='-', linewidth=1)
    ax4.set_aspect('equal')
    ax4.set_title('pcd')
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,
                              norm=plt.Normalize(vmin=lidar_intensity_min, vmax=lidar_intensity_max)),
        cax=fig.add_axes(
            [left4, bottom4 - h_padding_colorbar / h_all - h_colorbar / h_all, w4, h_colorbar / h_all]),
        orientation='horizontal', label='intensity'
    )

    # HeatmapBEV Static
    left5 = left2 + w2 + w_padding / w_all
    bottom5 = (h_padding_mid + h + h_padding_bottom) / h_all
    w5 = w / w_all
    h5 = h / h_all
    ax5 = fig.add_axes([left5, bottom5, w5, h5])
    ax5.tripcolor(
        heatmapBEV_x.flatten(),
        heatmapBEV_y.flatten(),
        heatmapBEV_static_dB.flatten(),
        cmap=cmap
    )
    ax5.set_xlim([-rlim, rlim])
    ax5.set_ylim([0, rlim])
    ax5.set_xlabel('x/m')
    ax5.set_ylabel('y/m')
    ax5.grid(color='gray', linestyle='-', linewidth=1)
    ax5.set_aspect('equal')
    ax5.set_title('HeatmapBEV Static')
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min_heatmapBEV_static_dB,
                                                            vmax=v_max_heatmapBEV_static_dB)),
        cax=fig.add_axes(
            [left5, bottom5 - h_padding_colorbar / h_all - h_colorbar / h_all, w5, h_colorbar / h_all]),
        orientation='horizontal', label='power(dB)'
    )

    # HeatmapBEV Dynamic
    left6 = left4 + w4 + w_padding / w_all
    bottom6 = h_padding_bottom / h_all
    w6 = w / w_all
    h6 = h / h_all
    ax6 = fig.add_axes([left6, bottom6, w6, h6])
    ax6.tripcolor(
        heatmapBEV_x.flatten(),
        heatmapBEV_y.flatten(),
        heatmapBEV_dynamic_dB.flatten(),
        cmap=cmap
    )
    ax6.set_xlim([-rlim, rlim])
    ax6.set_ylim([0, rlim])
    ax6.set_xlabel('x/m')
    ax6.set_ylabel('y/m')
    ax6.grid(color='gray', linestyle='-', linewidth=1)
    ax6.set_aspect('equal')
    ax6.set_title('HeatmapBEV Dynamic')
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min_heatmapBEV_dynamic_dB,
                                                            vmax=v_max_heatmapBEV_dynamic_dB)),
        cax=fig.add_axes(
            [left6, bottom6 - h_padding_colorbar / h_all - h_colorbar / h_all, w6, h_colorbar / h_all]),
        orientation='horizontal', label='power(dB)'
    )

    plt.show()



if __name__ == '__main__':
    main()

