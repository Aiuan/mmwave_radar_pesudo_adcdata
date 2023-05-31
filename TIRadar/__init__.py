import numpy as np

def calculate_radar_parameters(infos):
    TI_Cascade_Antenna_DesignFreq_GHz = 76.8

    doa_unitDis = 0.5 * infos['freq_center_GHz'] / TI_Cascade_Antenna_DesignFreq_GHz

    num_rx = infos['numRXPerDevice'] * infos['numDevices']
    num_tx = infos['numTXPerDevice'] * infos['numDevices']

    rx_id = np.arange(num_rx, dtype='int')
    rx_id_onboard = np.array([13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8], dtype='int') - 1
    rx_position_azimuth = np.array([11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3], dtype='int')
    rx_position_elevation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int')

    tx_id_transfer_order = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype='int') - 1
    tx_id = np.arange(num_tx, dtype='int')
    tx_id_onboard = np.array([12, 11, 10, 3, 2, 1, 9, 8, 7, 6, 5, 4], dtype='int') - 1
    tx_position_azimuth = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0], dtype='int')
    tx_position_elevation = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int')

    # calculate virtual antenna array
    virtual_array_azimuth = np.tile(tx_position_azimuth[tx_id_transfer_order], (num_rx, 1)) + \
                            np.tile(rx_position_azimuth[rx_id_onboard], (num_tx, 1)).T
    virtual_array_elevation = np.tile(tx_position_elevation[tx_id_transfer_order], (num_rx, 1)) + \
                              np.tile(rx_position_elevation[rx_id_onboard], (num_tx, 1)).T
    virtual_array_tx_id = np.tile(tx_id_transfer_order.reshape(1, -1), (num_rx, 1))
    virtual_array_rx_id = np.tile(rx_id_onboard.reshape(-1, 1), (1, num_tx))

    # azimuth, elevation, rx_id, tx_id
    virtual_array = np.hstack(
        (
            virtual_array_azimuth.reshape((-1, 1), order='F'),
            virtual_array_elevation.reshape((-1, 1), order='F'),
            virtual_array_rx_id.reshape((-1, 1), order='F'),
            virtual_array_tx_id.reshape((-1, 1), order='F')
        )
    )

    # get antenna_noredundant
    _, virtual_array_index_noredundant = np.unique(
        virtual_array[:, :2], axis=0, return_index=True
    )
    virtual_array_noredundant = virtual_array[virtual_array_index_noredundant, :]

    # get antenna_redundant
    virtual_array_index_redundant = np.setxor1d(
        np.arange(virtual_array.shape[0]),
        virtual_array_index_noredundant
    )
    virtual_array_redundant = virtual_array[virtual_array_index_redundant, :]

    # find and associate overlaped rx_tx pairs
    virtual_array_info_overlaped_associate = []
    for i in range(virtual_array_index_redundant.shape[0]):
        mask = (virtual_array_noredundant == virtual_array_redundant[i])
        mask = np.logical_and(mask[:, 0], mask[:, 1])
        info_associate = virtual_array_noredundant[mask][0]
        info_overlaped = virtual_array_redundant[i]
        virtual_array_info_overlaped_associate.append(
            np.concatenate((info_associate, info_overlaped)).tolist()
        )
    # azimuth, elevation, rx_associated, tx_associated, azimuth, elevation, rx_overlaped, tx_overlaped
    virtual_array_info_overlaped_associate = np.array(virtual_array_info_overlaped_associate)

    diff_tx = abs(virtual_array_info_overlaped_associate[:, 7] - virtual_array_info_overlaped_associate[:, 3])
    virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_associate[diff_tx == 1]

    sorted_index = np.argsort(virtual_array_info_overlaped_diff1tx[:, 0])
    virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_diff1tx[sorted_index, :]
    virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_diff1tx

    # find noredundant row1
    virtual_array_noredundant_row1 = virtual_array_noredundant[virtual_array_noredundant[:, 1] == 0]

    res = {
        'TI_Cascade_Antenna_DesignFreq_GHz': TI_Cascade_Antenna_DesignFreq_GHz,
        'doa_unitDis': doa_unitDis,
        'num_rx': num_rx,
        'num_tx': num_tx,
        'rx_id': rx_id,
        'rx_id_onboard': rx_id_onboard,
        'rx_position_azimuth': rx_position_azimuth,
        'rx_position_elevation': rx_position_elevation,
        'tx_id_transfer_order': tx_id_transfer_order,
        'tx_id': tx_id,
        'tx_id_onboard': tx_id_onboard,
        'tx_position_azimuth': tx_position_azimuth,
        'tx_position_elevation': tx_position_elevation,

        'virtual_array_azimuth': virtual_array_azimuth,
        'virtual_array_elevation': virtual_array_elevation,
        'virtual_array_tx_id': virtual_array_tx_id,
        'virtual_array_rx_id': virtual_array_rx_id,
        'virtual_array': virtual_array,
        'virtual_array_noredundant': virtual_array_noredundant,
        'virtual_array_redundant': virtual_array_redundant,
        'virtual_array_info_overlaped_associate': virtual_array_info_overlaped_associate,
        'virtual_array_info_overlaped_diff1tx': virtual_array_info_overlaped_diff1tx,
        'virtual_array_noredundant_row1': virtual_array_noredundant_row1
    }
    for key, value in infos.items():
        res[key] = value

    return res

