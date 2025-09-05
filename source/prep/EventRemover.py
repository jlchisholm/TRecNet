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
import os
from argparse import ArgumentParser
from Util import *


class EventRemover:
    """ 
    A class for removing unwanted events from a root file.

        Methods:
            cutOnMinValue: removes events that do not have at least the minimum value for a given variable.
            removeNonSemiLep: removes events that are not semi-leptonic.
            removeNonsense: removes events that are generally problematic/non-sensical.
            removeEvents: creates a new root file with the undesired events removed.

    """

    def __init__(self):
        print("Creating EventRemover.")
        self.ni = 0
        self.ni_now = 0
        self.nf = 0
    
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
        
        # Keeping track of events removed
        self.nf = len(tree)
        print(str(self.ni_now-self.nf)+' events with less than '+str(min)+' for '+str_var+' removed.')
        self.ni_now = self.nf
        
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

        # Keeping track of events removed
        self.nf = len(tree)
        print(str(self.ni_now-self.nf)+' non semi-leptonic events removed.')
        self.ni_now = self.nf
        
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
        
        sel1 = tree[str_ttbar_eta]>-100
        sel2 = tree[str_ttbar_eta]<100
        tree = tree[sel1*sel2]
        
        # Keeping track of events removed
        self.nf = len(tree)
        print(str(self.ni_now-self.nf)+' nonsensical (|ttbar_eta|>100) events removed.')
        self.ni_now = self.nf
        
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
        
        # Get the initial number of events
        self.ni = len(nom_tree)
        self.ni_now = self.ni
        print('Initial number of events: '+str(self.ni))
        
        # Remove non-desired events
        if (min_jets > 0):
            print('Removing events with less than '+str(min_jets)+' jets ...')
            str_jet_n = getObservableName(var_conf,nom_keys,'jet_n')
            nom_tree = self.cutOnMinValue(nom_tree,min_jets,str_jet_n)
        if (min_bjets > 0):
            print('Removing events with less than '+str(min_bjets)+' b-tagged jets ...')
            str_bjet_n = getObservableName(var_conf,nom_keys,'bjet_n')
            nom_tree = self.cutOnMinValue(nom_tree,min_bjets,str_bjet_n)
        if (min_met_met > 0):
            print('Removing events with met_met less than '+str(min_met_met)+' ...')
            str_met_met = getObservableName(var_conf,nom_keys,'met_met')
            nom_tree = self.cutOnMinValue(nom_tree,min_met_met,str_met_met)
        if remove_nonSemiLep:
            print('Removing non semi-leptonic events ...')
            str_isSemiLep = getObservableName(var_conf,nom_keys,'isSemiLep')
            nom_tree = self.removeNonSemiLep(nom_tree,str_isSemiLep)
        if remove_nonsense:
            print('Removing nonsensical events ...')
            str_ttbar_eta = getObservableName(var_conf,nom_keys,'ttbar_eta')
            nom_tree = self.removeNonsense(nom_tree,str_ttbar_eta)
            
        print("Total number of events removed: "+str(self.ni-self.ni_now))
            
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
parser.add_argument('--min_jets',help='Set minimum number of jets per event.',default=0,type=int)
parser.add_argument('--min_bjets',help='Set minimum number of b-tagged jets per event.',default=0,type=int)
parser.add_argument('--min_met_met',help='Set minimum met_met.',default=20,type=float)
parser.add_argument('--remove_nonSemiLep',help='Removes events that are not semi-leptonic.',action='store_true')
parser.add_argument('--remove_nonsense',help='Removes events that are non-sensical.',action='store_true')

# Parse the arguments and proceed with stuff
args = parser.parse_args()
remover = EventRemover()
remover.removeEvents(args.input,args.save_dir,args.var_conf,args.min_jets,args.min_bjets,args.min_met_met,args.remove_nonSemiLep,args.remove_nonsense)

print('Done :)')