def get_snapsot_latte_df(p_snap, max_epoch = None):
    latte_force_df, latte_energy_df = pd.DataFrame([]), pd.DataFrame([])
    val_der = p_snap
    epochs = get_sort_models(val_der, format = 'dat')
    if max_epoch:
        epochs = epochs[:max_epoch]
    for epoch in epochs:
        F_df = read_dat(val_der, f'/epoch_{epoch}_step_{epoch * 1000}_forces.dat')
        E_df = read_dat(val_der, f'/epoch_{epoch}_step_{epoch * 1000}.dat') 
    
        if epoch == epochs[0]:
            latte_force_df['names'] = F_df['#filename'] + '_' + F_df['atom_id'].astype(str)
            latte_energy_df['names'] = E_df['#filename']

        latte_energy_df[f'{epoch}'] = (E_df['e_ref'] - E_df['e_nn'])/ E_df['n_atoms']
        F_df['err'] = F_df['fx_ref'] - F_df['fx_nn']
        F_df['ids'] = F_df['#filename'] + '_' + F_df['atom_id'].astype(str) 
        temp_dict = {name: err for name, err in zip(F_df['ids'], F_df['err'] )} 
        rearranged_dict = {k: temp_dict[k] for k in  latte_force_df['names']}
        latte_force_df[f'{epoch}'] = [v for v in rearranged_dict.values()]
        
    splitted = latte_force_df['names'].str.rsplit('_',  n=2, expand=True) 
    part1 = splitted.iloc[:, -2] # sys id
    part2 = splitted.iloc[:, 0:-2].agg('_'.join, axis=1)
    part3 = splitted.iloc[:, -1]# atom id
    latte_force_df['names'] = part1 + '_' + part2 + '_' + part3
    
    latte_columns = [str(e) for e in epochs]
    latte_force_df['Avg_err'] = latte_force_df[latte_columns].mean(axis = 1)
    latte_energy_df['Avg_err'] = latte_energy_df[latte_columns].mean(axis = 1)
    latte_force_df['Resid'] = latte_force_df[latte_columns].sub(latte_force_df['Avg_err'], axis=0).abs().mean(axis=1)
    latte_energy_df['Resid'] = latte_energy_df[latte_columns].sub(latte_energy_df['Avg_err'], axis=0).abs().mean(axis=1)
    latte_force_df['var'] = latte_force_df[latte_columns].var(axis=1, ddof=0)
    latte_force_df['Std_dev'] = latte_force_df[latte_columns].std(axis=1, ddof=0)
    latte_energy_df['Std_dev'] = latte_energy_df[latte_columns].std(axis=1, ddof=0)
    ############################################################################
    latte_force_df = latte_force_df.dropna(subset=['Resid', 'Std_dev'])
    latte_energy_df = latte_energy_df.dropna(subset=['Resid', 'Std_dev'])
    splitted = latte_energy_df['names'].str.rsplit('_', n = 1, expand=True)
    latte_energy_df['names'] = splitted.iloc[:,1] + '_' + splitted.iloc[:, 0]
    return latte_force_df, latte_energy_df
