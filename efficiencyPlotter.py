#!/usr/bin/env python3

import helpers.fileHelper as fileHelper

import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
import awkward
import sys
import os
import helpers.helper as helper
import helpers.AndrewsHelper as Andrew_Helper
from scipy import optimize

if __name__ == "__main__":
    # This code will run if called from command line directly
    # python3 efficiencyPlotter.py [options] outputDir inputFiles


    # Get input options from command line
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] outputDir outputFileName inputFile")
    parser.add_option("--emtf-mode", dest="emtf_mode", type="int",
                     help="USED FOR LABELS - EMTF Track Mode", default=15)
    parser.add_option("--cmssw-rel", dest="cmssw_rel", type="string",
                     help="USED FOR LABELS - CMSSW Release used in training preperation", default="CMSSW_12_1_0_pre3")
    parser.add_option("--eta-mins", dest="eta_mins", type="string",
                     help="Array of Minimum Eta (Must be same length as --eta-maxs)", default="[1.25]")
    parser.add_option("--eta-maxs", dest="eta_maxs", type="string",
                     help="Array of Maximum Eta (Must be same length as --eta-mins)", default="[2.5]")
    parser.add_option("--pt-cuts", dest="pt_cuts", type="string",
                     help="Array of EMTF pt Cuts (GeV)", default="[22]")
    parser.add_option("-v","--verbose", dest="verbose",
                     help="Print extra debug info", default=False)
    options, args = parser.parse_args()
    
    # Convert string arrays to float values
    options.pt_cuts = [float(pt) for pt in options.pt_cuts[1:-1].split(',')]
    options.eta_mins = [float(eta) for eta in options.eta_mins[1:-1].split(',')]
    options.eta_maxs = [float(eta) for eta in options.eta_maxs[1:-1].split(',')]

    # Check to make sure inputs match and if not print help
    if(len(options.eta_mins) != len(options.eta_maxs)):
        parser.print_help()
        sys.exit(1)
    if(len(args) < 3):
        print("\n######### MUST INCLUDE ARGS: outputDir outputFileName inputFile #########\n")
        parser.print_help()
        sys.exit(1)
    if(len(args[1].split('.')) > 0):
        args[1] = args[1].split('.')[0]

    if(options.verbose):
        print("#######################################################")
        print("#                  EFFICIENCY PLOTTER                 #")
        print("#######################################################")
        print("Loaded Options and Arguments. \n")

    # Get input file using fileHelper
    target_file = fileHelper.getFile(args[2], options.verbose)
    
    if(options.verbose):
        print("\nTarget File Loaded\n")
        print("\nCollecting GEN_pt, GEN_eta, BDTG_AWB_Sq, and TRK_hit_ids\n")

    # Collect branch using fileHelper
    branch_GEN_pt = fileHelper.getBranch(target_file,"f_logPtTarg_invlog2PtWgt/TestTree/GEN_pt", options.verbose)
    branch_EMTF_pt = fileHelper.getBranch(target_file,"f_logPtTarg_invlog2PtWgt/TestTree/EMTF_pt", options.verbose)
    branch_BDTG_AWB_Sq = fileHelper.getBranch(target_file,"f_logPtTarg_invlog2PtWgt/TestTree/BDTG_AWB_Sq", options.verbose)
    branch_GEN_eta = fileHelper.getBranch(target_file,"f_logPtTarg_invlog2PtWgt/TestTree/GEN_eta", options.verbose)
    branch_GEN_phi = fileHelper.getBranch(target_file,"f_logPtTarg_invlog2PtWgt/TestTree/GEN_phi", options.verbose)
    branch_TRK_hit_ids = fileHelper.getBranch(target_file,"f_logPtTarg_invlog2PtWgt/TestTree/TRK_hit_ids", options.verbose)

    # Group branches into dictionary for reference
    unbinned_EVT_data = {}
    unbinned_EVT_data['GEN_pt'] = branch_GEN_pt.arrays()['GEN_pt']
    unbinned_EVT_data['EMTF_pt'] = helper.unscaleBDTPtRun2(branch_EMTF_pt.arrays()['EMTF_pt'])
    unbinned_EVT_data['BDT_pt'] = 2**branch_BDTG_AWB_Sq.arrays()['BDTG_AWB_Sq']
    unbinned_EVT_data['GEN_eta'] = branch_GEN_eta.arrays()['GEN_eta']
    unbinned_EVT_data['GEN_phi'] = branch_GEN_phi.arrays()['GEN_phi']
    unbinned_EVT_data['TRK_hit_ids'] = branch_TRK_hit_ids.arrays()['TRK_hit_ids']

    # Open a matplotlib PDF Pages file
    pp = fileHelper.openPdfPages(args[0], args[1], options.verbose)

    if(options.verbose):
        print("\nCreating ETA Mask and PT_cut MASK\n")
        print("Applying:\n   " + str(options.eta_mins) + " < eta < " + str(options.eta_maxs))
        print("   " + str(options.pt_cuts) + "GeV < pT\n")    

    # Import efficiencyPlotter to access its functions
    import efficiencyPlotter
    pt_cuts = options.pt_cuts
    eta_mins = options.eta_mins
    eta_maxs = options.eta_maxs

    pt_val = [] #[1, 2, 3, 4, 5, 6, 7]
    pt_50 = [] #[0.754, 1.644, 2.889, 4.055, 5.125, 6.18, 7.251]
    c = [] #[0.079, 0.232, 0.181, 0.163, 0.162, 0.167, 0.175]

    # Go through each PT Cut and Eta Cuts and generate figures to save to pdf
    for pt_cut in pt_cuts:
        for i in range(0, len(eta_mins)):
            if(options.verbose):
                print("###################   New Cuts   ###################")

            # Apply ETA Mask
            unbinned_EVT_data_eta_masked = helper.applyMaskToEVTData(
                                            unbinned_EVT_data,
                                            ["GEN_pt", "EMTF_pt", "BDT_pt", "GEN_eta", "GEN_phi", "TRK_hit_ids"], 
                                            ((eta_mins[i] < abs(unbinned_EVT_data["GEN_eta"])) & (eta_maxs[i] > abs(unbinned_EVT_data["GEN_eta"]))),
                                            "ETA CUT: " + str(eta_mins[i]) + " < eta < " + str(eta_maxs[i]), options.verbose)
            
            # Apply PT Cut mask
            unbinned_EVT_data_eta_masked_pt_pass = helper.applyMaskToEVTData(
                                            unbinned_EVT_data_eta_masked,
                                            ["GEN_pt", "GEN_eta", "GEN_phi"],
                                            (pt_cut < unbinned_EVT_data_eta_masked["BDT_pt"]),
                                            "PT CUT: " + str(pt_cut) + " < pT", options.verbose)
            # Apply EMTF PT Cut mask
            unbinned_EVT_data_eta_masked_emtf_pt_pass = helper.applyMaskToEVTData(
                                            unbinned_EVT_data_eta_masked,
                                            ["GEN_pt", "GEN_eta", "GEN_phi"],
                                            (pt_cut < unbinned_EVT_data_eta_masked["EMTF_pt"]),
                                            "EMTF PT CUT: " + str(pt_cut) + " < pT", options.verbose)

            # Apply Plataue Cut
            unbinned_EVT_data_eta_masked_plataue = helper.applyMaskToEVTData(
                                            unbinned_EVT_data_eta_masked,
                                            ["GEN_pt", "BDT_pt", "EMTF_pt", "GEN_eta", "GEN_phi", "TRK_hit_ids"],
                                            (pt_cut+10 < unbinned_EVT_data_eta_masked["GEN_pt"]),
                                            "GEN PT CUT: " + str(pt_cut) + " < pT", options.verbose)

            # Apply PT Cut to Plataue
            unbinned_EVT_data_eta_masked_plataue_pt_pass = helper.applyMaskToEVTData(
                                            unbinned_EVT_data_eta_masked_plataue,
                                            ["GEN_pt", "GEN_eta", "GEN_phi"],
                                            (pt_cut < unbinned_EVT_data_eta_masked_plataue["BDT_pt"]),
                                            "Plataue PT CUT: " + str(pt_cut) + " < pT", options.verbose)

            unbinned_EVT_data_eta_masked_plataue_emtf_pt_pass = helper.applyMaskToEVTData(
                                            unbinned_EVT_data_eta_masked_plataue,
                                            ["GEN_pt", "GEN_eta", "GEN_phi"],
                                            (pt_cut < unbinned_EVT_data_eta_masked_plataue["EMTF_pt"]),
                                            "Plataue EMTF PT CUT: " + str(pt_cut) + " < pT", options.verbose)

            # Generate efficiency vs eta plot
          # eta_fig = efficiencyPlotter.makeEfficiencyVsEtaStackedPlot(unbinned_EVT_data_eta_masked_plataue_pt_pass["GEN_eta"], unbinned_EVT_data_eta_masked_plataue["GEN_eta"],
          #                                    unbinned_EVT_data_eta_masked_plataue_emtf_pt_pass["GEN_eta"], unbinned_EVT_data_eta_masked_plataue["GEN_eta"],
          #                                    "EMTF BDT Efficiency \n Emulation in "  + str(options.cmssw_rel), "Retrained LUT", "Run2 LUT",
          #                                    "mode: " + str(options.emtf_mode)
          #                                   + "\n" + str(eta_mins[i]) + " < $\eta$ < " + str(eta_maxs[i])
          #                                   + "\n $p_T$ > " + str(pt_cut) + "GeV"
          #                                   + "\n" + "$N_{events}$: "+str(len(unbinned_EVT_data_eta_masked_plataue["GEN_eta"])), pt_cut, options.verbose)

            # Generate efficiency vs phi plot
          # phi_fig = efficiencyPlotter.makeEfficiencyVsPhiPlot(unbinned_EVT_data_eta_masked_plataue_pt_pass["GEN_phi"], unbinned_EVT_data_eta_masked_plataue["GEN_phi"],
          #                                     "EMTF BDT Efficiency \n $\epsilon$ vs $\phi$ \n " + str(options.cmssw_rel), "mode: " + str(options.emtf_mode)
          #                                    + "\n" + str(eta_mins[i]) + " < $\eta$ < " + str(eta_maxs[i])
          #                                    + "\n $p_T$ > " + str(pt_cut) + "GeV"
          #                                    + "\n" + "$N_{events}$: "+str(len(unbinned_EVT_data_eta_masked_plataue["GEN_phi"])), pt_cut, options.verbose)

            # Generate efficiency vs pt plot
            pt_fig, pt_50_elem, c_elem = efficiencyPlotter.makeEfficiencyVsPtStackedPlot(unbinned_EVT_data_eta_masked_pt_pass["GEN_pt"], unbinned_EVT_data_eta_masked["GEN_pt"],
                                              unbinned_EVT_data_eta_masked_emtf_pt_pass["GEN_pt"], unbinned_EVT_data_eta_masked["GEN_pt"],
                                              "EMTF BDT Efficiency \n Emulation in " + str(options.cmssw_rel), "Retrained LUT", "Run2 LUT",
                                              "mode: " + str(options.emtf_mode)
                                              + "\n" + str(eta_mins[i]) + " < $\eta$ < " + str(eta_maxs[i])
                                              + "\n $p_T$ > " + str(pt_cut) + "GeV"
                                              + "\n" + "$N_{events}$: "+str(len(unbinned_EVT_data_eta_masked["GEN_pt"])), pt_cut, options.verbose)
            
            pt_50.append(pt_50_elem)
            c.append(c_elem)

            # Increase size to square 6in x 6in on PDF
            pt_fig.set_size_inches(6, 6)
           # phi_fig.set_size_inches(6, 6)
           # eta_fig.set_size_inches(6,6)
            # Save figures to PDF
            pp.savefig(pt_fig)
           # pp.savefig(eta_fig)
           # pp.savefig(phi_fig)

    # Generate fit parameters vs pt plot
    fitParam_fig = efficiencyPlotter.makeFitParametersVsPtPlot(pt_cuts, pt_50, c)

    fitParam_fig.set_size_inches(6, 6)
    pp.savefig(fitParam_fig)


    if(options.verbose):
        print("\nClosing PDF\n")
    #Close PDF
    pp.close()
    if(options.verbose):
        print("\nPDF has been closed\n")

    if(options.verbose):
        print("------------------------------------------------")
        print("DONE.\n")


# This code will run if this file is imported
# import efficiencyPlotter

def getEfficiciencyHist(num_binned, den_binned):
    """
       getEfficiciencyHist creates a binned histogram of the ratio of num_binned and den_binned
       and uses a Clopper-Pearson confidence interval to find uncertainties.

       NOTE: num_binned should be a strict subset of den_binned.

       NOTE: efficiency_binned_err[0] is lower error bar and efficiency_binned_err[1] is upper error bar

       INPUT:
             num_binned - TYPE: numpy array-like
             den_binned - TYPE: numpy array-like
       OUTPUT:
             efficiency_binned - TYPE: numpy array-like
             efficiency_binned_err - TYPE: [numpy array-like, numpy array-like]
       
       
    """
    # Initializing binned data
    efficiency_binned = np.array([])
    efficiency_binned_err = [np.array([]), np.array([])]

    # Iterating through each bin 
    for i in range(0, len(den_binned)):
        # Catching division by 0 error
        if(den_binned[i] == 0):
            efficiency_binned = np.append(efficiency_binned, 0)
            efficiency_binned_err[0] = np.append(efficiency_binned_err[0], [0])
            efficiency_binned_err[1] = np.append(efficiency_binned_err[1], [0])
            continue

        # Filling efficiency bins
        efficiency_binned = np.append(efficiency_binned, [num_binned[i]/den_binned[i]])

        # Calculating Clopper-Pearson confidence interval
        nsuccess = num_binned[i]
        ntrial = den_binned[i]
        conf = 95.0
    
        if nsuccess == 0:
            alpha = 1 - conf / 100
            plo = 0.
            phi = scipy.stats.beta.ppf(1 - alpha, nsuccess + 1, ntrial - nsuccess)
        elif nsuccess == ntrial:
            alpha = 1 - conf / 100
            plo = scipy.stats.beta.ppf(alpha, nsuccess, ntrial - nsuccess + 1)
            phi = 1.
        else:
            alpha = 0.5 * (1 - conf / 100)
            plo = scipy.stats.beta.ppf(alpha, nsuccess + 1, ntrial - nsuccess)
            phi = scipy.stats.beta.ppf(1 - alpha, nsuccess, ntrial - nsuccess)

        # Filling efficiency error bins
        efficiency_binned_err[0] = np.append(efficiency_binned_err[0], [(efficiency_binned[i] - plo)])
        efficiency_binned_err[1] = np.append(efficiency_binned_err[1], [(phi - efficiency_binned[i])])# - efficiency_binned[i]])

    return efficiency_binned, efficiency_binned_err

def makeEfficiencyVsPtPlot(num_unbinned, den_unbinned, title, textStr, xvline, verbose=False):
    """
       makeEfficiencyVsPtPlot creates a binned histogram plot of the ratio of num_unbinned and den_unbinned vs pT
       and calls getEfficiciencyHist.

       NOTE: num_unbinned should be a strict subset of den_unbinned.

       INPUT:
             num_unbinned - TYPE: numpy array-like
             den_unbinned - TYPE: numpy array-like
             title - TYPE: String (Plot Title)
             textStr - TYPE: String (Text Box info)
             xvline - TYPE: Float (x value of vertical line)
       OUTPUT:
             fig - TYPE: MatPlotLib PyPlot Figure containing efficiency vs pt plot
    """

    if(verbose):
        print("\nInitializing Figures and Binning Pt Histograms")

    # Initializing bins and binning histograms from unbinned entries
    # Bins start small and get larger toward larger pT
    bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000]
    den_binned, den_bins = np.histogram(den_unbinned, bins, (0,1000))
    num_binned, num_bins = np.histogram(num_unbinned, bins, (0,1000))

    if(verbose):
        print("Generating Efficiency vs Pt Plot")

    # Calling getEfficiciencyHist to get efficiency with Clopper-Pearson error
    efficiency_binned, efficiency_binned_err = getEfficiciencyHist(num_binned, den_binned)

    # Generating a plot with 2 subplots (One to show turn on region, one to show high pT behavior)
    fig2, ax = plt.subplots(2)
    fig2.suptitle(title)


    # Plotting on first set of axes
    ax[0].errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                    efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5)
    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax[0].set_ylabel("Efficiency")
    ax[0].set_xlabel("$p_T$(GeV)")
    ax[0].axhline(linewidth=.1)        
    ax[0].axvline(linewidth=.1)
    ax[0].grid(color='lightgray', linestyle='--', linewidth=.25)
    ax[0].axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    ax[0].axvline(x=xvline, color='r', linewidth=.5, linestyle='--')
    # Adding a text box to bottom right
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax[0].text(0.95, 0.05, textStr, transform=ax[0].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # Setting axes limits to view turn on region
    ax[0].set_ylim([0,1.2])
    ax[0].set_xlim([0,max(2*xvline,50)])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        item.set_fontsize(8)


    # Plotting on second set of axes
    ax[1].errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                    efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5)
    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax[1].set_ylabel("Efficiency")
    ax[1].set_xlabel("$p_T$(GeV)")
    ax[1].axhline(linewidth=.1)
    ax[1].axvline(linewidth=.1)
    ax[1].grid(color='lightgray', linestyle='--', linewidth=.25)
    ax[1].axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    ax[1].axvline(x=xvline, color='r', linewidth=.5, linestyle='--')
    # Adding a text box to bottom right
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax[1].text(0.95, 0.05, textStr, transform=ax[1].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # Setting y-axis limit but not x-axis limit to see high pT behavior
    ax[1].set_ylim([0,1.2])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        item.set_fontsize(8)


    if(verbose):
        print("Finished Creating Pt Figures\n")
    # Returning the final figure with both plots drawn
    return fig2

def makeEfficiencyVsEtaPlot(num_unbinned, den_unbinned, title, textStr, xvline, verbose=False):
    """
       makeEfficiencyVsEtaPlot creates a binned histogram plot of the ratio of num_unbinned and den_unbinned vs eta
       and calls getEfficiciencyHist.

       NOTE: num_unbinned should be a strict subset of den_unbinned.

       INPUT:
             num_unbinned - TYPE: numpy array-like
             den_unbinned - TYPE: numpy array-like
             title - TYPE: String (Plot Title)
             textStr - TYPE: String (Text Box Info)
             xvline - TYPE: Float (x value of vertical line)
       OUTPUT:
             fig - TYPE: MatPlotLib PyPlot Figure containing efficiency vs eta plot
    """

    if(verbose):
        print("\nInitializing Figures and Binning eta Histograms")

    # Binning unbinned entries with 50 bins from -2.5 to 2.5 (Seeing both endcaps)
    den_binned, den_bins = np.histogram(den_unbinned, 50, (-2.5,2.5))
    num_binned, num_bins = np.histogram(num_unbinned, 50, (-2.5,2.5))

    if(verbose):
        print("Generating Efficiency vs eta Plot")

    # Calling getEfficiciencyHist to get binned efficiency and Clopper-Pearson error
    efficiency_binned, efficiency_binned_err = getEfficiciencyHist(num_binned, den_binned)

    fig2, ax = plt.subplots(1)
    fig2.suptitle(title)


    # Plot the efficiency and errors on the axes
    ax.errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                 efficiency_binned, yerr=efficiency_binned_err, xerr=[(den_bins[i+1] - den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                 linestyle="", marker=".", markersize=3, elinewidth = .5)
    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax.set_ylabel("Efficiency")
    ax.set_xlabel("$\eta$")
    ax.axhline(linewidth=.1)
    ax.axvline(linewidth=.1)
    ax.grid(color='lightgray', linestyle='--', linewidth=.25)
    ax.axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    ax.axvline(x=xvline, color='r', linewidth=.5, linestyle='--')
    # Add text box in the bottom right
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax.text(0.95, 0.05, textStr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # Set x and y limits
    ax.set_ylim([0,1.2])
    ax.set_xlim([-2.5,2.5])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)


    if(verbose):
        print("Finished Creating Eta Figures\n")
    # Returning the final figure with both plots drawn
    return fig2

def makeEfficiencyVsPhiPlot(num_unbinned, den_unbinned, title, textStr, xvline, verbose=False):
    """
       makeEfficiencyVsPhiPlot creates a binned histogram plot of the ratio of num_unbinned and den_unbinned vs phi
       and calls getEfficiciencyHist.

       NOTE: num_unbinned should be a strict subset of den_unbinned.

       INPUT:
             num_unbinned - TYPE: numpy array-like
             den_unbinned - TYPE: numpy array-like
             title - TYPE: String (Plot Title)
             textStr - TYPE: String (Text Box Info)
             xvline - TYPE: Float (x value of vertical line)
       OUTPUT:
             fig - TYPE: MatPlotLib PyPlot Figure containing efficiency vs phi plot
    """

    if(verbose):
        print("\nInitializing Figures and Binning phi Histograms")

    # Binning unbinned entries with 90 bins from -180 to 180
    den_binned, den_bins = np.histogram(den_unbinned, 90, (-180,180))
    num_binned, num_bins = np.histogram(num_unbinned, 90, (-180,180))

    if(verbose):
        print("Generating Efficiency vs phi Plot")

    # Calling getEfficiciencyHist to get binned efficiency and Clopper-Pearson error
    efficiency_binned, efficiency_binned_err = getEfficiciencyHist(num_binned, den_binned)

    fig2, ax = plt.subplots(1)
    fig2.suptitle(title)


    # Plot the efficiency and errors on the axes
    ax.errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                 efficiency_binned, yerr=efficiency_binned_err, xerr=[(den_bins[i+1] - den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                 linestyle="", marker=".", markersize=3, elinewidth = .5)
    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax.set_ylabel("Efficiency")
    ax.set_xlabel("$\phi$")
    ax.axhline(linewidth=.1)
    ax.axvline(linewidth=.1)
    ax.grid(color='lightgray', linestyle='--', linewidth=.25)
    ax.axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    # Add text box in the bottom right
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax.text(0.95, 0.05, textStr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # Set x and y limits
    ax.set_ylim([0,1.2])
    ax.set_xlim([-200,200])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)


    if(verbose):
        print("Finished Creating Phi Figures\n")
    # Returning the final figure with both plots drawn
    return fig2



def makeEfficiencyVsPtStackedPlot(num_unbinned, den_unbinned, num2_unbinned, den2_unbinned, title, label1, label2, textStr, xvline, verbose=False):
    """
       makeEfficiencyVsPtPlot creates a binned histogram plot of the ratio of num_unbinned and den_unbinned vs pT
       and calls getEfficiciencyHist.

       NOTE: num_unbinned should be a strict subset of den_unbinned.

       INPUT:
             num_unbinned - TYPE: numpy array-like
             den_unbinned - TYPE: numpy array-like
             num2_unbinned - TYPE: numpy array-like
             den2_unbinned - TYPE: numpy array-like
             title - TYPE: String (Plot Title)
             textStr - TYPE: String (Text Box info)
             xvline - TYPE: Float (x value of vertical line)
       OUTPUT:
             fig - TYPE: MatPlotLib PyPlot Figure containing efficiency vs pt plot
    """

    if(verbose):
        print("\nInitializing Figures and Binning Pt Histograms")

    # Initializing bins and binning histograms from unbinned entries
    # Bins start small and get larger toward larger pT
    bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000]
    den_binned, den_bins = np.histogram(den_unbinned, bins, (0,1000))
    num_binned, num_bins = np.histogram(num_unbinned, bins, (0,1000))
    den2_binned, den2_bins = np.histogram(den2_unbinned, bins, (0,1000))
    num2_binned, num2_bins = np.histogram(num2_unbinned, bins, (0,1000))

    if(verbose):
        print("Generating STACKED Efficiency vs Pt Plot")

    # Calling getEfficiciencyHist to get efficiency with Clopper-Pearson error
    efficiency_binned, efficiency_binned_err = getEfficiciencyHist(num_binned, den_binned)
    efficiency2_binned, efficiency2_binned_err = getEfficiciencyHist(num2_binned, den2_binned)

    # Generating a plot with 2 subplots (One to show turn on region, one to show high pT behavior)
    fig2, ax = plt.subplots(2)
    fig2.suptitle(title)

    # Ploting Effiency Function Fit from Andrew's Helper
    eff_func_fit = []
    pt_arr = []
    popt, pcov = optimize.curve_fit(Andrew_Helper.effFunc_v, xdata=[den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)], ydata=efficiency_binned, p0=(xvline, .2), method="lm")
    print(popt)
    for i in range(0, 5000):
        pt_val = i/5 + 0.5 #bins[i] + (bins[i + 1] - bins[i])/2
        pt_arr.append(pt_val)
        eff_func_fit.append(Andrew_Helper.effFunc(pt_val,popt[0],popt[1]))

    # Plotting on first set of axes
    ax[0].errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                    efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5, label=label1, color="royalblue")
    ax[0].plot(pt_arr,
                    eff_func_fit, #yerr=0, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    #linestyle="", marker="v", markersize=3, elinewidth = .5,
                     label=label1+" fit", color="black")
    # ax[0].errorbar([den2_bins[i]+(den2_bins[i+1]-den2_bins[i])/2 for i in range(0, len(den2_bins)-1)],
    #                efficiency2_binned, yerr=efficiency2_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
    #                linestyle="", marker=".", markersize=3, elinewidth = .5, label=label2, color="lightcoral")

    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax[0].set_ylabel("Efficiency")
    ax[0].set_xlabel("$p_T$(GeV)")
    ax[0].axhline(linewidth=.1)
    ax[0].axvline(linewidth=.1)
    ax[0].grid(color='lightgray', linestyle='--', linewidth=.25)
    ax[0].axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    ax[0].axvline(x=xvline, color='r', linewidth=.5, linestyle='--')
    # Adding a text box to bottom right
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax[0].text(0.95, 0.05, textStr + " \n$p_T^{50} = " + str(round(popt[0], 3)) + "$" + " \n$c = " + str(round(popt[1], 3)) + "$", transform=ax[0].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    # Setting axes limits to view turn on region
    ax[0].set_ylim([0,1.2])
    ax[0].set_xlim([0,max(2*xvline,50)])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        item.set_fontsize(8)

    # Plotting on second set of axes
    ax[1].errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                    efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5, label=label1, color="royalblue")
    ax[1].plot(pt_arr,
                    eff_func_fit,
                    label=label1+" fit", color="black")
    # ax[1].errorbar([den2_bins[i]+(den2_bins[i+1]-den2_bins[i])/2 for i in range(0, len(den2_bins)-1)],
    #                efficiency2_binned, yerr=efficiency2_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
    #                linestyle="", marker=".", markersize=3, elinewidth = .5, label=label2, color="lightcoral")
    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax[1].set_ylabel("Efficiency")
    ax[1].set_xlabel("$p_T$(GeV)")
    ax[1].axhline(linewidth=.1)
    ax[1].axvline(linewidth=.1)
    ax[1].grid(color='lightgray', linestyle='--', linewidth=.25)
    ax[1].axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    ax[1].axvline(x=xvline, color='r', linewidth=.5, linestyle='--')
    # Adding a text box to bottom right
#    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
#    ax[1].text(0.35, 0.05, textStr, transform=ax[1].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left', bbox=props)
    # Setting y-axis limit but not x-axis limit to see high pT behavior
    ax[1].set_ylim([0,1.2])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        item.set_fontsize(8)
    ax[1].legend()

    if(verbose):
        print("Finished Creating Pt Figures\n")
    # Returning the final figure with both plots drawn
    return fig2, popt[0], popt[1]


def makeEfficiencyVsEtaStackedPlot(num_unbinned, den_unbinned,num2_unbinned, den2_unbinned, title,label1, label2, textStr, xvline, verbose=False):
    """
       makeEfficiencyVsEtaPlot creates a binned histogram plot of the ratio of num_unbinned and den_unbinned vs eta
       and calls getEfficiciencyHist.

       NOTE: num_unbinned should be a strict subset of den_unbinned.

       INPUT:
             num_unbinned - TYPE: numpy array-like
             den_unbinned - TYPE: numpy array-like
             title - TYPE: String (Plot Title)
             textStr - TYPE: String (Text Box Info)
             xvline - TYPE: Float (x value of vertical line)
       OUTPUT:
             fig - TYPE: MatPlotLib PyPlot Figure containing efficiency vs eta plot
    """

    if(verbose):
        print("\nInitializing Figures and Binning eta Histograms")

    # Binning unbinned entries with 50 bins from -2.5 to 2.5 (Seeing both endcaps)
    den_binned, den_bins = np.histogram(den_unbinned, 50, (-2.5,2.5))
    num_binned, num_bins = np.histogram(num_unbinned, 50, (-2.5,2.5))
    den2_binned, den2_bins = np.histogram(den2_unbinned, 50, (-2.5,2.5))
    num2_binned, num2_bins = np.histogram(num2_unbinned, 50, (-2.5,2.5))

    if(verbose):
        print("Generating STACKED Efficiency vs eta Plot")

    # Calling getEfficiciencyHist to get binned efficiency and Clopper-Pearson error
    efficiency_binned, efficiency_binned_err = getEfficiciencyHist(num_binned, den_binned)
    efficiency2_binned, efficiency2_binned_err = getEfficiciencyHist(num2_binned, den2_binned)

    fig2, ax = plt.subplots(1)
    fig2.suptitle(title)


    # Plot the efficiency and errors on the axes
    ax.errorbar([den_bins[i]+(den_bins[i+1]-den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                 efficiency_binned, yerr=efficiency_binned_err, xerr=[(den_bins[i+1] - den_bins[i])/2 for i in range(0, len(den_bins)-1)],
                 linestyle="", marker=".", markersize=3, elinewidth = .5, label=label1, color="royalblue")
    ax.errorbar([den2_bins[i]+(den2_bins[i+1]-den2_bins[i])/2 for i in range(0, len(den2_bins)-1)],
                 efficiency2_binned, yerr=efficiency2_binned_err, xerr=[(den2_bins[i+1] - den2_bins[i])/2 for i in range(0, len(den2_bins)-1)],
                 linestyle="", marker=".", markersize=3, elinewidth = .5, label=label2, color="lightcoral")
    # Setting Labels, vertical lines, horizontal line at 90% efficiency, and plot configs
    ax.set_ylabel("Efficiency")
    ax.set_xlabel("$\eta$")
    ax.axhline(linewidth=.1)
    ax.axvline(linewidth=.1)
    ax.grid(color='lightgray', linestyle='--', linewidth=.25)
    ax.axhline(y=0.9, color='r', linewidth=.5, linestyle='--')
    ax.axvline(x=xvline, color='r', linewidth=.5, linestyle='--')
    # Add text box in the bottom right
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax.text(0.42, 0.05, textStr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left', bbox=props)
    # Set x and y limits
    ax.set_ylim([0,1.2])
    ax.set_xlim([-2.5,2.5])
    # Setting all font sizes to be small (Less distracting)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
    ax.legend()

    if(verbose):
        print("Finished Creating Eta Figures\n")
    # Returning the final figure with both plots drawn
    return fig2



def makeFitParametersVsPtPlot(pt_val, pt_50, c):
    
    bias = np.divide(pt_50, pt_val) - 1


    fig2, ax = plt.subplots(2)
    fig2.suptitle("Fit Parameters")
    ax[0].errorbar(pt_val, bias, xerr = 0, yerr = 0, linestyle="", marker=".", markersize=3, elinewidth = .5, label= "Pt_50/Pt_Cut", color="royalblue")
    ax[0].set_xlabel("Pt")
    ax[0].set_ylabel("Bias")
    ax[0].grid(color='lightgray', linestyle='--', linewidth=.25)
    ax[0].legend()
    for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        item.set_fontsize(8)
        
    ax[1].errorbar(pt_val, c, xerr = 0, yerr = 0, linestyle="", marker=".", markersize=3, elinewidth = .5, label= "c", color="royalblue")
    ax[1].set_xlabel("Pt")
    ax[1].set_ylabel("c")
    ax[1].grid(color='lightgray', linestyle='--', linewidth=.25)
    ax[1].legend()
    for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        item.set_fontsize(8)  

    return fig2









