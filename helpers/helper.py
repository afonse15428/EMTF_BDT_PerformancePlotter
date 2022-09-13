import numpy as np

def scaleBDTFunction(unbinned_BDT_pT, pt_scaling_A, pt_scaling_B):
    pt_min = 20
    unbinned_BDT_pT_scaled = np.where(unbinned_BDT_pT > pt_min, (pt_scaling_A * unbinned_BDT_pT)/(1 - (pt_scaling_B * pt_min)), (pt_scaling_A * unbinned_BDT_pT)/(1 - (pt_scaling_B * unbinned_BDT_pT)))

    return unbinned_BDT_pT_scaled

def scaleBTDInverseFunction(unbinned_BDT_pT, pt_scaling_A, pt_scaling_B):
    pt_min = 20
    unbinned_BDT_pt_unscale_factor= np.divide(1, (pt_scaling_A + pt_scaling_B * unbinned_BDT_pT))
    unbinned_BDT_pt_unscaled = np.where(unbinned_BDT_pt_unscale_factor > 1/(pt_scaling_A + pt_scaling_B * pt_min),np.divide(unbinned_BDT_pT, (pt_scaling_A + pt_scaling_B * unbinned_BDT_pT)),unbinned_BDT_pT* ((1 - pt_scaling_B * pt_min)/pt_scaling_A))
    return unbinned_BDT_pt_unscaled

def unscaleBDTPtRun2(unbinned_BDT_pT):
    return scaleBTDInverseFunction(unbinned_BDT_pT, 1.2, .015)

def scaleBDTPtRun2(unbinned_BDT_pT):
     return scaleBDTFunction(unbinned_BDT_pT, 1.13, .015)
#    return scaleBDTFunction(unbinned_BDT_pT, 1.15, .009)

def scaleBDTPtRun3(unbinned_BDT_pT):
    return scaleBDTFunction(unbinned_BDT_pT, 1.246, 0.017)

def applyMaskToEVTData(unbinned_EVT_data, keys_to_mask, masked_array, title, verbose = False):
    if(verbose and title):
        print("\n" + title)
        print("Masking " + str(keys_to_mask))

    unbinned_EVT_data_masked = {}
    for k in keys_to_mask:
        unbinned_EVT_data_masked[k] = unbinned_EVT_data[k][masked_array]

    return unbinned_EVT_data_masked

