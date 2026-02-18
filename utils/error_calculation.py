import os
import numpy as np
import pickle
from pathlib import Path

from ase.io import read
from M_models import get_sort_models

def get_latte_force_component_error(val_path, epoch = None, comp = 'x'):
    epochs = get_sort_models(val_path, format = 'dat')
    if epoch:
        epoch = epoch
    else:
        epoch = epochs[-1]

    F_df = read_dat(val_path, f'/epoch_{epoch}_step_{epoch * 1000}_forces.dat')
    E_df = read_dat(val_path, f'/epoch_{epoch}_step_{epoch * 1000}.dat') 
    F_error = F_df[f'f{comp}_ref'] - F_df[f'f{comp}_nn']
    E_error = (E_df['e_ref'] - E_df['e_nn'])/ E_df['n_atoms']
    
    return E_error, F_error 




def get_mace_error(file_ref, file_mace, comp = 'x'):
    ref_data = read(file_ref, index=':')
    mace_data = read(file_mace, index=':')
    E_err = []
    F_err = {k:[] for k in ['#filename', 'atom_id', f'f{comp}_mace', f'f{comp}_ref']}
    for idx, (sys_ref, sys_mace) in enumerate(zip(ref_data, mace_data)):
        try:
            cat = sys_ref.info['category'].replace(' ', '').replace('/', '_')
            file_name = cat.strip() + '_' + str(idx)
        except KeyError:
            file_name = 'config_' + str(idx)
        
        e_ref = sys_ref.get_total_energy()
        e_mace = sys_mace.info['MACE_energy']
        forces_ref = sys_ref.get_forces()
        forces_mace = sys_mace.arrays['MACE_forces']
        E_err.append({
            '#filename': file_name,
            'energy_error_per_atom': (e_ref - e_mace) / len(forces_ref)
        })
        for atom_idx, (f_ref, f_mace) in enumerate(zip(forces_ref, forces_mace)):
            F_err['#filename'].append(file_name)
            F_err['atom_id'].append(atom_idx)
            F_err[f'f{comp}_ref'].append(f_ref[0])
            F_err[f'f{comp}_mace'].append(f_mace[0])

    E_df = pd.DataFrame(E_err)
    F_df = pd.DataFrame(F_err)
    return E_df, F_df

