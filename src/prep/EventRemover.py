######################################################################
#                                                                    #
#  EventRemover.py                                                   #
#  Author: Jenna Chisholm                                            #
#  Updated: Jun.5/25                                                 #
#                                                                    #
#  A class for removing undesirable events, for training             #
#  purposes. Not intended for systematics.                           #
#                                                                    #
#  Thoughts for improvements:                                        #
#       --> Include systematics
#                                                                    #
######################################################################

import uproot
import os, sys
from argparse import ArgumentParser
from Util import *


class EventRemover:
    """ 
    A class for removing unwanted events from a root file.

        Methods:
            cutOnJets: Removes events that do not have at least min_jets jets. 
            cutOnbJets: Removes events that do not have at least min_bjets b-tagged jets.
            removeNonSemiLep: Removes events that are not semi-leptonic.
            removeNonsense: Removes events that are generally problematic/non-sensical.

    """

    def __init__(self):
        print("Creating EventRemover.")
    
    
    def cutOnJets(self, tree,min_jets,str_jet_n):
        """
        Removes events that do not have at least min_jets jets.
        
            Parameters:
                tree (root tree): Nominal tree from the root file.
                min_jets (int): Minimum number of jets per event.
                str_jet_n (str): Ntuple name for 'jet_n'.

            Returns:
                tree (root tree): Nominal tree, with events of at least min_jets jets.
        """
        
        sel = tree[str_jet_n]>=min_jets
        tree = tree[sel]
        
        print('Events with less than '+str(min_jets)+' jets removed.')
        
        return tree
    
    def cutOnbJets(self, tree,min_bjets,str_jet_isbtag):
        """
        Removes events that do not have at least min_bjets b-tagged jets.
        
            Parameters:
                tree (root tree): Nominal tree from the root file.
                min_bjets (int): Minimum number of b-tagged jets per event.
                n_jet_isbtag (str): Ntuple name for 'jet_isbtag'.

            Returns:
                tree (root tree): Nominal tree, with events of at least min_bjets b-tagged jets.
        """
        
        sel = [len(tree[str_jet_isbtag][i,:])>=min_bjets for i in tree[str_jet_isbtag]]
        tree = tree[sel]
        
        print('Events with less than '+str(min_bjets)+' jets removed.')
        
        return tree
    
    
    def cutOnMinValue(self, tree,min,str_var):
        """
        Removes events that do not have at least the minimum value for a given variable.
        
            Parameters:
                tree (root tree): Nominal tree from the root file.
                min (double): Minimum number of b-tagged jets per event.
                str_var (str): Ntuple name for the variable.

            Returns:
                tree (root tree): Nominal tree, with events of at least min value for given variable.
        """        
        
        sel = tree[str_var]>=min
        tree = tree[sel]
        
        print('Events with less than '+str(min)+' for '+str_var+' removed.')
        
        return tree
    
    
    def cutOnMaxValue(self, tree,max,str_var):
        """
        Removes events that have (strictly) more than the maximum value for a given variable.
        
            Parameters:
                tree (root tree): Nominal tree from the root file.
                min (double): Minimum number of b-tagged jets per event.
                str_var (str): Ntuple name for the variable.

            Returns:
                tree (root tree): Nominal tree, with events of no more than max value for given variable.
        """        
        
        sel = tree[str_var]<max
        tree = tree[sel]
        
        print('Events with more than '+str(max)+' for '+str_var+' removed.')
        
        return tree
    
    
    def removeNonSemiLep(self, tree,str_isSemiLep):
        """
        Removes events that are not semi-leptonic.
        
            Parameters:
                tree (root tree): Nominal tree from the root file.
                str_isSemiLep (str): Ntuple name for 'isSemiLep'.

            Returns:
                tree (root tree): Nominal tree, with non semi-leptonic events removed.
        """
        
        sel = tree[str_isSemiLep]==1
        tree = tree[sel]

        print('Non semi-leptonic events removed.')
        
        return tree
    
    
    def removeNonsense(self, tree,str_ttbar_eta):
        """
        Removes events that are generally problematic/non-sensical.
        
            Parameters:
                tree (root tree): Nominal tree from the root file.
                str_ttbar_eta: Ntuple name for 'ttbar_eta'.

            Returns:
                tree (root tree): Nominal tree, with bad events removed.
        """
        
        sel = tree[str_ttbar_eta]>-100
        tree = tree[sel]
        
        print('Nonsensical events removed.')
        
        return tree
        
        
    def removeEvents(self, input_file,save_dir,var_conf,min_jets,min_bjets,min_met_met,remove_nonSemiLep,remove_nonsense):
        """
        Creates a new root file with the undesired events removed.
        
            Parameters:
                input_file (str): Name (including path) of the root file you'd like to add the jet matches to.
                save_dir (str): Desired directory to save the output root file in.
                var_conf (str): Name (including path) of the config file for the variable names.
                min_jets (int): Minimum number of jets per event.
                min_bjets (int): Minimum number of b-tagged jets per event.
                met_met (double): Minimum value for met_met.
                remove_nonSemiLep (bool): Flag to remove non semi-leptonic events.
                remove_nonsensical (bool): Flag to remove non sensical events.
            
            Returns:
                Creates a new root file with the undesired events removed.
        """
        
        # Separate input file name and its path
        #in_path = os.path.split(input_file)[0]
        in_name = os.path.split(input_file)[1]

        # Open the original file and a new file to write to
        print('Opening root file ...')
        og_file = uproot.open(input_file)
        
        print(og_file["truth"].keys())

        # Get the reco and parton trees from the original file
        nom_name, up_name, down_name = getBranchNames(var_conf)
        #down_tree = og_file[down_name].arrays()
        #up_tree = og_file[up_name].arrays()
        nom_tree = og_file[nom_name].arrays()

        # Save the keys for later
        #down_keys = og_file[down_name].keys()
        #up_keys = og_file[up_name].keys()
        nom_keys = og_file[nom_name].keys()
        
        # Close the original file
        og_file.close()
        
        # Remove non-desired events
        if min_jets > 0:
            print('Removing events with less than '+str(min_jets)+' jets ...')
            str_jet_n = getObservableName('jet_n')
            nom_tree = self.cutOnJets(nom_tree,min_jets,str_jet_n)
        if min_bjets > 0:
            print('Removing events with less than '+str(min_bjets)+' b-tagged jets ...')
            if('jet_isbtag' not in nom_keys):
                print('Binary b-tags missing from nominal tree. Please add these to your ntuple before continuing.')
                sys.exit()
            str_jet_isbtag = getObservableName('jet_isbtag')
            nom_tree = self.cutOnbJets(nom_tree,min_bjets,str_jet_isbtag)
        if min_met_met > 0:
            print('Removing events with met_met less than '+str(min_met_met)+' ...')
            str_met_met = getObservableName('met_met')
            nom_tree = self.cutOnMinValue(nom_tree,min_met_met,str_met_met)
        if remove_nonSemiLep:
            print('Removing non semi-leptonic events ...')
            str_isSemiLep = getObservableName('isSemiLep')
            nom_tree = self.removeNonSemiLep(nom_tree,str_isSemiLep)
        if remove_nonsense:
            print('Removing nonsensical events ...')
            str_ttbar_eta = getObservableName('ttbar_eta')
            nom_tree = self.removeNonsense(nom_tree,str_ttbar_eta)
            
            
            
        # Write the trees to the file
        print('Writing trees to new file...')
        new_file_name = save_dir+'/'+in_name.split('.root')[0]+'_pruned.root'
        new_file = uproot.recreate(new_file_name)
        #new_file[down_name] = {key:down_tree[key] for key in down_keys}
        #new_file[up_name] = {key:up_tree[key] for key in up_keys}
        new_file[nom_name] = {key:nom_tree[key] for key in nom_keys}
        print('Saved file: '+new_file_name)

        # Close new file
        new_file.close()
        
        
        
        
        
        
# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #      
        
# Create the main parser
parser = ArgumentParser()

# Define arguments for EventRemover
parser.add_argument('--input',help='Input file (including path).',required=True)
parser.add_argument('--save_dir',help='Path for directory where file will be saved.',required=True)
parser.add_argument('--var_conf',help='Config file (including path) for names of variables.',required=True)
parser.add_argument('--min_jets',help='Set minimum number of jets per event.',default=0)
parser.add_argument('--min_bjets',help='Set minimum number of b-tagged jets per event.',default=0)
parser.add_argument('--min_met_met',help='Set minimum met_met.',default=20)
parser.add_argument('--remove_nonSemiLep',help='Removes events that are not semi-leptonic.',action='store_true')
parser.add_argument('--remove_nonsense',help='Removes events that are non-sensical.',action='store_true')

# Parse the arguments and proceed with stuff
args = parser.parse_args()
remover = EventRemover()
remover.removeEvents(args.input,args.save_dir,args.var_conf,args.min_jets,args.min_bjets,args.min_met_met,args.remove_nonSemiLep,args.remove_nonsense)

print('Done :)')