# -*- coding: utf-8 -*-

import os
import pickle

#%%

def read_data_from_1810_09466():
    # elements in datalist: 
    #    element[0] = R (kpc)
    #    element[1] = vc (km/s)
    #    element[2] = sigma- (km/s)
    #    element[3] = sigma+ (km/s)
    #    element[4] = syst (km/s) # saved later

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_file=dir_path+'/1810_09466.dat'

    with open(data_file, 'r') as f:
        content = f.readlines()
    
    data_lists = []
    for line in content:
        if not line.startswith('#') and line.strip():
            data_lists.append([float(a) for a in line.split()])

    systematics_file = dir_path+'/1810_09466-sys-data.dat'

    with open(systematics_file, 'r') as f:
        content = f.readlines()

    syst_list = []
    for line in content:
        if not line.startswith('#') and line.strip():
            syst_list.append([float(a) for a in line.split()])

    relative_syst_list = [data[1] for data in syst_list]

    data_values_list = [data[1] for data in data_lists]
    syst_list = []
    for i in range(len(relative_syst_list)):
        syst_list.append(relative_syst_list[i]*data_values_list[i])

    counter = 0
    for element in data_lists:
        element.append(syst_list[counter])
        counter += 1

    return data_lists

def pickle_results(Analysis, file_name):

    data_to_pickle = {
                'lnlikelihoods':Analysis.sampler.lnprobability,
                'chains':Analysis.sampler.chain, 
                'acceptance_fractions':Analysis.sampler.acceptance_fraction,
                'variable_names':Analysis.variables_key_list,
    }

    with open(file_name, "wb") as f:
        pickle.dump(data_to_pickle,file=f)
        
def load_pickle_results(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

#%%
if __name__ == '__main__':
    pass
        
        
