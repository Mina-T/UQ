import os
import pandas as pd
from Mace_validate import *
from utils.labels import get_drautz_label, drautz_atom_label

ref_file ='/leonardo_scratch/large/userexternal/mtaleblo/Carbon_project/Dataset/Drautz/categories/all_test.xyz'
ensembles_dir = '/leonardo_scratch/large/userexternal/mtaleblo/My_Carbon_Project/MACE/Drautz/categories/all/'
def ensembles_table(ref_file, ensembles_dir):
    model_df = pd.DataFrame([])
    dirs = sorted([d for d in os.listdir() if d.startswith('ensembel_')])
    for dir in dirs:
        val_file = f'{ensembles_dir}{dir}/output_all_test.xyz'
        E_err, F_err = validate_mace(ref_file, val_file)
        if dir == 'ensembel_1':
            model_df['names'] = F_err['#filename'] + '_' + F_err['atom_id'].astype(str) 
        for component in ['x', 'y', 'z']:
            F_err[f'err_{component}'] = (F_err[f'f{component}_ref'] - F_err[f'f{component}_mace'])**2
            F_err[f'ids_{component}'] = F_err['#filename'] + '_' + F_err['atom_id'].astype(str) 
            temp_dict = {name: err for name, err in zip(F_err[f'ids_{component}'], F_err[f'err_{component}'] )} 
            rearranged_dict = {k: temp_dict[k] for k in  model_df['names']}
            model_df[f'{dir}_{component}'] = [v for v in rearranged_dict.values()]
        model_df[f'err_{dir}'] = sum(model_df[f'{dir}_{component}'] for component in ['x', 'y', 'z'])

    splitted = model_df['names'].str.rsplit('_',  n=2, expand=True)
    cat_id = splitted.iloc[:, 0] 
    sys_id = splitted.iloc[:, 1].agg('_'.join, axis=1) 
    atom_id = splitted.iloc[:, -1]
    model_df['names'] = drautz_atom_label(cat_id, sys_id, atom_id)

    _columns = [f'err_{dir}' for dir in dirs]
    model_df['MSE'] = model_df[_columns].mean(axis = 1)
    _columns = [f'{c}_{comp}' for c in dirs for comp in ['x', 'y', 'z']]
    model_df['var_x'] = model_df[[c for c in _columns if c.endswith('_x')]].var(axis=1, ddof=0)
    model_df['var_y'] = model_df[[c for c in _columns if c.endswith('_y')]].var(axis=1, ddof=0)
    model_df['var_z'] = model_df[[c for c in _columns if c.endswith('_z')]].var(axis=1, ddof=0)
    model_df['sigma'] = (model_df['var_x']**2 + model_df['var_y']**2 + model_df['var_z']**2)
    model_df['category'] = model_df['names'].apply(get_drautz_label).str[0]
    model_df.drop(columns = [c for c in _columns if c.endswith('_x')], inplace=True)
    model_df.drop(columns = [c for c in _columns if c.endswith('_y')], inplace=True)
    model_df.drop(columns = [c for c in _columns if c.endswith('_z')], inplace=True)
    return model_df

def get_model_dict(ensembles, model_names)-> dict:
    all_configs_dict = dict()
    for ensemble, name in zip(ensembles, model_names):
        for _, row in ensemble.iterrows():
            config_name = row['names']
            all_configs_dict.setdefault(config_name, {})
            all_configs_dict[config_name][name] = {
            'sigma': row['sigma'],
            'MSE': row['MSE']}
            
    return all_configs_dict    #{"1_1715_0": {"MACE": {"sigma": 0.12, "MSE": 0.03},"LATTE": {"sigma": 0.10, "MSE": 0.02}}}


def ensemble_statistics():
    '''
    statistics of the ensebmle of models:
    min, max, mean, median, std_dev, probability distribution
    '''
    pass
