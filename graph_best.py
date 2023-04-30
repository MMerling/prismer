import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


train_prismer_df = pd.read_json(open('logging/classification_prismer_base_bakcup__5e-9/epoch_train.jsonl'), lines=True)
train_prismerz_df = pd.read_json(open('logging/classification_prismerz_base_saved_5e-9/epoch_train.jsonl'), lines=True)
valid_prismer_df = pd.read_json(open('logging/classification_prismer_base_bakcup__5e-9/epoch_valid.jsonl'), lines=True)
valid_prismerz_df = pd.read_json(open('logging/classification_prismerz_base_saved_5e-9/epoch_valid.jsonl'), lines=True)

def plot_graph(df1, df2, ylabel, max_epoch, train=False):
    if train:
        graph_type = "train"
    else:
        graph_type = "valid"
    plt.figure()
    plt.plot(df1[df1["epoch"]<=max_epoch][ylabel.lower()], label="Prismer Base")
    plt.plot(df2[ylabel.lower()], label="Prismer ZBase")
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=0)
    plt.legend()
    plt.xlabel("Epoch")
    if ylabel=="Acc":
        plt.ylabel("Accuracy")
    else:
        plt.ylabel(ylabel)
    plt.savefig(f"graphs/prismer_{ylabel.lower()}_vs_epoch_{graph_type}.png")
    plt.close()

plot_graph(train_prismer_df, train_prismerz_df, "Loss", 11)
plot_graph(valid_prismer_df, valid_prismerz_df, "Loss", 11)
plot_graph(valid_prismer_df, valid_prismerz_df, "Acc", 11)
plot_graph(valid_prismer_df, valid_prismerz_df, "AUROC", 11)