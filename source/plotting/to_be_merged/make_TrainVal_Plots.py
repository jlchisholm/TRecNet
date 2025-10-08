import numpy as np
import matplotlib.pyplot as plt



save_dir = './'
ticks_size = 10
axis_size = 12
legend_size = 10 


# ------- TRecNet Hypertuned vs Original ------- #

# TRecNet with hypertuned hp vs original hp
TRecNet_hist_hyp = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_6jets_20230524_144557/TRecNet_6jets_20230524_144557_TrainHistory.npy',allow_pickle=True).item()
TRecNet_hist_org = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_6jets_20230525_111504/TRecNet_6jets_20230525_111504_TrainHistory.npy',allow_pickle=True).item()

# MAE Plot
plt.figure()
plt.plot(TRecNet_hist_hyp['loss'], label='Hypertuned (Training)', color="tab:blue")
plt.plot(TRecNet_hist_hyp['val_loss'], '--',label='Hypertuned (Validation)', color="tab:blue")
plt.plot(TRecNet_hist_org['loss'], label='Original (Training)', color="tab:orange")
plt.plot(TRecNet_hist_org['val_loss'], '--',label='Original (Validation)', color="tab:orange")
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('MAE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('MAE Loss')
plt.savefig(save_dir+'MAE_Loss_TRecNet_Hyp_vs_Org',bbox_inches='tight')

# MSE Plot
plt.figure()
plt.plot(TRecNet_hist_hyp['mse'], label='Hypertuned (Training)', color="tab:blue")
plt.plot(TRecNet_hist_hyp['val_mse'], '--',label='Hypertuned (Validation)', color="tab:blue")
plt.plot(TRecNet_hist_org['mse'], label='Original (Training)', color="tab:orange")
plt.plot(TRecNet_hist_org['val_mse'], '--',label='Original (Validation)', color="tab:orange")
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('MSE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('MSE Loss')
plt.savefig(save_dir+'MSE_Loss_TRecNet_Hyp_vs_Org',bbox_inches='tight')

# ------- TRecNet Number of Jets ------- #

# TRecNet with hypertuned hp vs original hp
TRecNet_hist_4jets = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_4jets_20230602_095841/TRecNet_4jets_20230602_095841_TrainHistory.npy',allow_pickle=True).item()
TRecNet_hist_5jets = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_5jets_20230601_223623/TRecNet_5jets_20230601_223623_TrainHistory.npy',allow_pickle=True).item()
TRecNet_hist_6jets = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_6jets_20230525_111504/TRecNet_6jets_20230525_111504_TrainHistory.npy',allow_pickle=True).item()
TRecNet_hist_7jets = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_7jets_20230526_160845/TRecNet_7jets_20230526_160845_TrainHistory.npy',allow_pickle=True).item()

# MAE Plot
plt.figure()
plt.plot(TRecNet_hist_4jets['loss'], label='4 Jets (Training)', color="tab:purple")
plt.plot(TRecNet_hist_4jets['val_loss'], '--',label='4 Jets (Validation)', color="tab:purple")
plt.plot(TRecNet_hist_5jets['loss'], label='5 Jets (Training)', color="tab:green")
plt.plot(TRecNet_hist_5jets['val_loss'], '--',label='5 Jets (Validation)', color="tab:green")
plt.plot(TRecNet_hist_6jets['loss'], label='6 Jets (Training)', color="tab:orange")
plt.plot(TRecNet_hist_6jets['val_loss'], '--',label='6 Jets (Validation)', color="tab:orange")
plt.plot(TRecNet_hist_7jets['loss'], label='7 Jets (Training)', color="tab:blue")
plt.plot(TRecNet_hist_7jets['val_loss'], '--',label='7 Jets (Validation)', color="tab:blue")
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('MAE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('MAE Loss')
plt.savefig(save_dir+'MAE_Loss_TRecNet_njets',bbox_inches='tight')

# MSE Plot
plt.figure()
plt.plot(TRecNet_hist_4jets['mse'], label='4 Jets (Training)', color="tab:purple")
plt.plot(TRecNet_hist_4jets['val_mse'], '--',label='4 Jets (Validation)', color="tab:purple")
plt.plot(TRecNet_hist_5jets['mse'], label='5 Jets (Training)', color="tab:green")
plt.plot(TRecNet_hist_5jets['val_mse'], '--',label='5 Jets (Validation)', color="tab:green")
plt.plot(TRecNet_hist_6jets['mse'], label='6 Jets (Training)', color="tab:orange")
plt.plot(TRecNet_hist_6jets['val_mse'], '--',label='6 Jets (Validation)', color="tab:orange")
plt.plot(TRecNet_hist_7jets['mse'], label='7 Jets (Training)', color="tab:blue")
plt.plot(TRecNet_hist_7jets['val_mse'], '--',label='7 Jets (Validation)', color="tab:blue")
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('MSE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('MSE Loss')
plt.savefig(save_dir+'MSE_Loss_TRecNet_njets',bbox_inches='tight')


# ------- TRecNet (org hp) vs +ttbar vs +JetPretrain vs Unfrozen ------- #

# Get hists
TRecNet_hist = np.load('/home/jchishol/TRecNet/TRecNet/TRecNet_6jets_20230525_111504/TRecNet_6jets_20230525_111504_TrainHistory.npy',allow_pickle=True).item()
TRecNet_ttbar_hist = np.load('/home/jchishol/TRecNet/TRecNet+ttbar/TRecNet+ttbar_6jets_20230525_214703/TRecNet+ttbar_6jets_20230525_214703_TrainHistory.npy',allow_pickle=True).item()
TRecNet_ttbar_JetPretrain_hist = np.load('/home/jchishol/TRecNet/TRecNet+ttbar+JetPretrain/TRecNet+ttbar+JetPretrain_6jets_20230531_133250/TRecNet+ttbar+JetPretrain_6jets_20230531_133250_TrainHistory.npy',allow_pickle=True).item()
TRecNet_Unfrozen_hist = np.load('/home/jchishol/TRecNet/TRecNet+ttbar+JetPretrainUnfrozen/TRecNet+ttbar+JetPretrainUnfrozen_6jets_20230601_131445/TRecNet+ttbar+JetPretrainUnfrozen_6jets_20230601_131445_TrainHistory.npy',allow_pickle=True).item()
JetPretrainer_hist = np.load('/home/jchishol/TRecNet/JetPretrainer/JetPretrainer_6jets_20230530_112415/JetPretrainer_6jets_20230530_112415_TrainHistory.npy',allow_pickle=True).item()

# Get epoch stopover for jet pretraining unfreezing
last_epoch = len(TRecNet_ttbar_JetPretrain_hist['loss'])
new_epochs = [last_epoch + epoch for epoch in range(len(TRecNet_Unfrozen_hist['loss']))]

# MAE Plot
plt.figure()
plt.plot(TRecNet_hist['loss'], label='TRecNet (Training)', color="tab:orange")
plt.plot(TRecNet_hist['val_loss'], '--',label='TRecNet (Validation)', color="tab:orange")
plt.plot(TRecNet_ttbar_hist['loss'], label='TRecNet+ttbar (Training)', color="tab:pink", alpha=0.7)
plt.plot(TRecNet_ttbar_hist['val_loss'], '--',label='TRecNet+ttbar (Validation)', color="tab:pink", alpha=0.7)
plt.plot(TRecNet_ttbar_JetPretrain_hist['loss'], label='TRecNet+ttbar+JetPretrain (Training)', color="tab:purple")
plt.plot(TRecNet_ttbar_JetPretrain_hist['val_loss'], '--',label='TRecNet+ttbar+JetPretrain (Validation)', color="tab:purple")
plt.plot(new_epochs, TRecNet_Unfrozen_hist['loss'], label='Finetuning (Training)', color="tab:purple", alpha=0.6)
plt.plot(new_epochs, TRecNet_Unfrozen_hist['val_loss'], '--',label='Finetuning (Validation)', color="tab:purple", alpha=0.6)
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('MAE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('MAE Loss')
plt.savefig(save_dir+'MAE_Loss_new',bbox_inches='tight')

# MSE Plot
plt.figure()
plt.plot(TRecNet_hist['mse'], label='TRecNet (Training)', color="tab:orange")
plt.plot(TRecNet_hist['val_mse'], '--',label='TRecNet (Validation)', color="tab:orange")
plt.plot(TRecNet_ttbar_hist['mse'], label='TRecNet+ttbar (Training)', color="tab:pink", alpha=0.7)
plt.plot(TRecNet_ttbar_hist['val_mse'], '--',label='TRecNet+ttbar (Validation)', color="tab:pink", alpha=0.7)
plt.plot(TRecNet_ttbar_JetPretrain_hist['mse'], label='TRecNet+ttbar+JetPretrain (Training)', color="tab:purple")
plt.plot(TRecNet_ttbar_JetPretrain_hist['val_mse'], '--',label='TRecNet+ttbar+JetPretrain (Validation)', color="tab:purple")
plt.plot(new_epochs, TRecNet_Unfrozen_hist['mse'], label='Finetuning (Training)', color="tab:purple", alpha=0.6)
plt.plot(new_epochs, TRecNet_Unfrozen_hist['val_mse'], '--',label='Finetuning (Validation)', color="tab:purple", alpha=0.6)
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('MSE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('MSE Loss')
plt.savefig(save_dir+'MSE_Loss_new',bbox_inches='tight')

# Binary Cross-Entropy Plot
plt.figure()
plt.plot(JetPretrainer_hist['loss'],label='Training',color="tab:purple")
plt.plot(JetPretrainer_hist['val_loss'], '--', label='Validation',color="tab:purple")
plt.xlabel('Epoch', fontsize=axis_size)
plt.ylabel('BCE', fontsize=axis_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
#plt.title('Binary Cross Entropy Loss')
plt.savefig(save_dir+'/BinaryCrossEntropy_Loss_new',bbox_inches='tight')