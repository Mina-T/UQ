import pandas as pd
from ase.io import read


def validate_mace(file_ref, file_mace):
    '''
    Given the reference and validated files, returns a dataframe including ref_F and mace_F.
    '''
    ref_data = read(file_ref, index=':')
    mace_data = read(file_mace, index=':')
    E_err = []
    F_err = {'#filename':[], 'atom_id':[], 'fx_mace':[], 'fy_mace':[], 'fz_mace':[], 'fx_ref':[], 'fy_ref':[], 'fz_ref':[]}
    for idx, (sys_ref, sys_mace) in enumerate(zip(ref_data, mace_data)):
        cat = sys_ref.info['category'].replace(' ', '').replace('/', '_')
        file_name = cat.strip() + '_' + str(idx)
        e_ref = sys_ref.get_total_energy()
        e_mace = sys_mace.info['MACE_energy']
        forces_ref = sys_ref.get_forces()
        forces_mace = sys_mace.arrays['MACE_forces']
        E_err.append({
            '#filename': file_name,
            'energy_error_per_atom': abs(e_ref - e_mace) / len(forces_ref)
        })
        for atom_idx, (f_ref, f_mace) in enumerate(zip(forces_ref, forces_mace)):
            F_err['#filename'].append(file_name)
            F_err['atom_id'].append(atom_idx)
            F_err['fx_ref'].append(f_ref[0])
            F_err['fy_ref'].append(f_ref[1])
            F_err['fz_ref'].append(f_ref[2])
            F_err['fx_mace'].append(f_mace[0])
            F_err['fy_mace'].append(f_mace[1])
            F_err['fz_mace'].append(f_mace[2])

    E_df = pd.DataFrame(E_err)
    F_df = pd.DataFrame(F_err)
    return E_df, F_df

