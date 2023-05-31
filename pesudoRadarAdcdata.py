import numpy as np

def pcd2adcdata(pcd_dict, radar_params):
    n_obj = pcd_dict['x'].shape[0]
    if not ('velocity' in pcd_dict.keys()):
        pcd_dict['velocity'] = np.zeros((n_obj,), dtype='float')

    num_samples = radar_params['numAdcSamples']
    num_loops = radar_params['numLoops']
    tx_id_transfer_order = radar_params['tx_id_transfer_order']
    num_chirps_in_loop = len(tx_id_transfer_order)
    num_rx = radar_params['num_rx']

    fs = radar_params['digOutSampleRate_ksps'] * 1e3
    Tsample = 1 / fs
    freq_start = radar_params['startFreqConst_GHz'] * 1e9
    adcstart_time = radar_params['adcStartTimeConst_usec'] * 1e-6
    idle_time = radar_params['idleTimeConst_usec'] * 1e-6
    slope = radar_params['freqSlopeConst_MHz_usec'] * 1e6 / 1e-6
    Tchirp = radar_params['Tchirp_usec'] * 1e-6
    Tloop = radar_params['Tloop_usec'] * 1e-6
    lambda_center = radar_params['lambda_center_mm'] * 1e-3
    light_speed_m_sec = radar_params['light_speed_m_sec']
    antenna_d_unit = (radar_params['light_speed_m_sec'] / (radar_params['TI_Cascade_Antenna_DesignFreq_GHz'] * 1e9)) / 2
    virtual_array_azimuth = radar_params['virtual_array_azimuth']
    virtual_array_elevation = radar_params['virtual_array_elevation']

    i_sample = np.arange(num_samples)
    i_loop = np.arange(num_loops)
    i_chirp_in_loop = np.arange(num_chirps_in_loop)
    freq0 = freq_start + slope * adcstart_time

    t = i_sample * Tsample
    St = np.exp(
        1j * 2 * np.pi * (
                freq0 * t + slope / 2 * t * t
        )
    )

    time_offset = idle_time + adcstart_time + Tsample * i_sample.reshape((-1, 1, 1)) + \
                  Tchirp * (
                          i_chirp_in_loop.reshape((1, 1, -1)) + i_loop.reshape((1, -1, 1)) * num_chirps_in_loop
                  )

    tau = 2 * (
            pcd_dict['range'].reshape(-1, 1, 1, 1) + pcd_dict['velocity'].reshape(-1, 1, 1, 1) * np.expand_dims(time_offset, axis=0)
    ) / light_speed_m_sec

    delta_phi_azim = np.sin(pcd_dict['azimuth'] / 180 * np.pi).reshape(-1, 1, 1) * \
                     np.expand_dims(virtual_array_azimuth, axis=0) * antenna_d_unit / lambda_center

    delta_phi_elev = np.sin(pcd_dict['elevation'] / 180 * np.pi).reshape(-1, 1, 1) * \
                     np.expand_dims(virtual_array_elevation, axis=0) * antenna_d_unit / lambda_center

    Ar = pcd_dict['intensity']

    t_tau = t.reshape((1, -1, 1, 1, 1)) - np.expand_dims(tau, 3)
    Sr = Ar.reshape((-1, 1, 1, 1, 1)) * np.exp(
        1j * 2 * np.pi * (
                freq0 * t_tau + slope / 2 * t_tau * t_tau
                - np.expand_dims(np.expand_dims(delta_phi_azim, axis=1), axis=1)
                - np.expand_dims(np.expand_dims(delta_phi_elev, axis=1), axis=1)
        )
    )
    adcdata = (St.reshape((1, -1, 1, 1, 1)) * Sr.conj()).sum(axis=0)

    return adcdata
