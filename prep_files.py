from MLPrep import jetMatcher, filePrep
import os





# ---------- PUT JET MATCHES IN ROOT FILES ---------- #

matcher = jetMatcher()

directory = '/data/jchishol/mc16d'
for file in sorted(os.scandir(directory),key=lambda f: f.name):
    if file.is_file() and ('jetMatch' not in file.name):
        print('On file',file.name)
        matcher.addMatchesToFile(file,dR_cut=0.4,allowDoubleMatching=True)



# ---------- MAKE INDIVIDUAL H5 FILES ---------- #

# Set number of jets and met_cut (in GeV), and create a prepper tool
# jn = 6
# met_cut = 20
# prepper = filePrep()

# # Iterate over root files
# for file in sorted(os.scandir('/data/jchishol/mc1516_new'),key=lambda f: f.name):
#     if file.is_file() and 'jetMatch' in file.name:
#       print('On file ',file.name)
#       save_name = 'variables_ttbar_'+file.name[8:-5]+'_'+str(jn)+'jets'
#       prepper.makeH5File(file=file,jn=jn,met_cut=met_cut,save_name=save_name,save_loc="/data/jchishol/ML_Data/")


# ---------- COMBINE H5 FILES ---------- #

# Get training and testing file lists
# train_list = []
# test_list = []
# for file in sorted(os.scandir('/data/jchishol/ML_Data'),key=lambda f: f.name):
#     if file.is_file() and 'train' not in file.name and 'test' not in file.name:
#         if '_01' in file.name or '_02' in file.name or '_03' in file.name or '_04' in file.name:
#             test_list.append(file)
#         else:
#             train_list.append(file)

# prepper.combineH5Files(train_list, save_name='variables_ttbar_ljets_jetMatch04_'+str(jn)+'jets_train')
# prepper.combineH5Files(test_list, save_name='variables_ttbar_ljets_jetMatch04_'+str(jn)+'jets_test')



# ---------- SAVE MAXMEAN ---------- #

#prepper.saveMaxMean('variables_ttbar_ljets_jetMatch04_6jets_train')





