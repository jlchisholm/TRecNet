import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker

# Load the ROOT file and extract the truth and reco trees
file = uproot.open("/home/dciarniello/summer2023/TRecNet/bbPretrainer/bbPretrainer_8jets_20230621_000853/bbPretrainer_8jets_20230621_000853_nominal_Results.root")
save_dir = "/home/dciarniello/summer2023/TRecNet/"
truth_tree = file["parton"]
reco_tree = file["reco"]

# Define the range of N
N_range = range(1, 9)

# Initialize an empty list to store the confusion matrices
confusion_matrices = []

# Loop over N values and compute the confusion matrix for each
for N in N_range:
    # Extract the jetN_isTruth_bb variables
    jetN_isTruth_bb = truth_tree[f"j{N}_isTruth_bb"].array()
    reco_jetN_isTruth_bb = reco_tree[f"j{N}_isTruth_bb"].array()

    # Convert the variables to numpy arrays
    jetN_isTruth_bb = np.array(jetN_isTruth_bb)
    reco_jetN_isTruth_bb = np.array(reco_jetN_isTruth_bb)

    # Apply thresholding to convert continuous values to binary labels
    jetN_isTruth_bb = np.where(jetN_isTruth_bb > 0.5, 1, 0)
    reco_jetN_isTruth_bb = np.where(reco_jetN_isTruth_bb > 0.4, 1, 0)

    # Create the confusion matrix
    confusion = confusion_matrix(jetN_isTruth_bb, reco_jetN_isTruth_bb, labels=[0, 1])

    # Add the confusion matrix to the list
    confusion_matrices.append(confusion)

# Plot the confusion matrices
labels = ["0", "1"]
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

max_count = np.max(confusion_matrices)  # Maximum count in all confusion matrices

for i, confusion in enumerate(confusion_matrices):
    ax = axs[i // 4, i % 4]
    normalized_confusion = confusion.astype(float) / confusion.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(normalized_confusion, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=1)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("reco")
    ax.set_ylabel("truth")
    ax.set_title(f"Confusion Matrix (jet{N_range[i]})")

    # Add labels to show the percentage of events in each bin
    for row in range(normalized_confusion.shape[0]):
        for col in range(normalized_confusion.shape[1]):
            ax.text(col, row, f"{normalized_confusion[row, col] * 100:.2f}%"+'\n'+f"{normalized_confusion[row, col] * max_count:.0f} events",
                    ha="center", va="center")

    # Calculate and display the total accuracy
    accuracy = np.trace(confusion) / confusion.sum()
    ax.text(0.5, -0.25, f"Accuracy: {accuracy * 100:.2f}%",
            ha="center", va="center", transform=ax.transAxes)

# Add colorbar
cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

#plt.tight_layout()
fig.suptitle('bb dR Matching', fontsize=16)
plt.savefig(save_dir + "confusion_matrices.png")
plt.show()