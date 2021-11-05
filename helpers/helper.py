import numpy as np

def scaleBDTFunction(unbinned_BDT_pT, pt_scaling_A, pt_scaling_B):

     print(pt_scaling_A, pt_scaling_B)
     scaleFactor = pt_scaling_A / (1+pt_scaling_B*unbinned_BDT_pT)
     unbinned_BDT_pT_scaled = np.where(1.715 >= scaleFactor, unbinned_BDT_pT, unbinned_BDT_pT/1.715)
     unbinned_BDT_pT_scaled = np.where(1.715 < scaleFactor, unbinned_BDT_pT_scaled, unbinned_BDT_pT/scaleFactor)

     for i in range(0, 1000):
         print(unbinned_BDT_pT_scaled[i], scaleFactor[i], 1.715, unbinned_BDT_pT[i])

     return unbinned_BDT_pT_scaled
#    return (pt_scaling_A * unbinned_BDT_pT)/(1 - (pt_scaling_B * unbinned_BDT_pT))

def scaleBDTPtRun2(unbinned_BDT_pT):
    return scaleBDTFunction(unbinned_BDT_pT, 1.2, 0.015)

def scaleBDTPtRun3(unbinned_BDT_pT):
    return scaleBDTFunction(unbinned_BDT_pT, 1.3, 0.004)

def applyMaskToEVTData(unbinned_EVT_data, keys_to_mask, masked_array, title, verbose = False):
    if(verbose and title):
        print("\n" + title)
        print("Masking " + str(keys_to_mask))

    unbinned_EVT_data_masked = {}
    for k in keys_to_mask:
        unbinned_EVT_data_masked[k] = unbinned_EVT_data[k][masked_array]

    return unbinned_EVT_data_masked

