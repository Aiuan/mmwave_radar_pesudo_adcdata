import json
import numpy as np


def pcd_in_polar_zone(pcd_dict, azimlim=[-np.inf, np.inf], elevlim=[-np.inf, np.inf], rlim=[-np.inf, np.inf], return_type='np_array', add_aer=False):
    assert 'x' in pcd_dict.keys()
    assert 'y' in pcd_dict.keys()
    assert 'z' in pcd_dict.keys()
    assert return_type == 'np_array' or return_type == 'dict'

    azim, elev, r = rectangular2polar(pcd_dict['x'], pcd_dict['y'], pcd_dict['z'])

    mask_azim = np.logical_and(azim >= azimlim[0], azim <= azimlim[1])

    mask_elev = np.logical_and(elev >= elevlim[0], elev <= elevlim[1])

    mask_r = np.logical_and(r >= rlim[0], r <= rlim[1])

    mask = np.logical_and(np.logical_and(mask_azim, mask_elev), mask_r)

    if return_type == 'np_array':
        res = np.array(list(pcd_dict.values())).T
        res = res[mask, :]

        if add_aer:
            res = np.hstack((res, np.stack((azim[mask], elev[mask], r[mask]), axis=1)))
    else:
        res = dict()
        for key, value in pcd_dict.items():
            res[key] = value[mask]

        if add_aer:
            res['azimuth'] = azim[mask]
            res['elevation'] = elev[mask]
            res['range'] = r[mask]

    return res

def rectangular2polar(x, y, z):
    '''
        x: right, y: front, z: up
    '''
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

    # elevation (-90, 90)
    # elev = 0, y+
    # elev = -90, z-
    # elev = 90, z+
    sin_elev = z / r
    elev = np.arcsin(sin_elev) / np.pi * 180

    # azimuth (-180, 180]
    # azim = 0, y +
    # azim = 90, x +
    # azim = -90, x -
    # azim = 180, y -
    sin_azim = x / r / np.cos(elev / 180 * np.pi)
    cos_azim = y / r / np.cos(elev / 180 * np.pi)
    azim = np.arcsin(sin_azim) / np.pi * 180
    mask = (cos_azim < 0)
    azim[mask] = azim[mask] / np.abs(azim[mask]) * (180 - np.abs(azim[mask]))

    return azim, elev, r

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

