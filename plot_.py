import json
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as mpatches
import matplotlib.pyplot  
import os 
import glob


def plt_atr(file, attr_1, attr_2, save_fld, ty_):
    # ---------------
    # plot
    # ---------------
    t_1,  at_1 = attr_1
    t_2,  at_2 = attr_2
    title = t_1+"/"+ t_2

    x = range(0,len(at_1),1)
    
    plt.plot(x,at_1,'bo--')
    plt.plot(x,at_2,'ro--')

    # -------------------------
    # Legends
    # -------------------------
    blue = mpatches.Patch(color='blue', label= t_1)
    red  = mpatches.Patch(color='red',  label= t_2)
    plt.legend(handles=[blue,red],fontsize=10)
    # ----------------------------
    plt.xticks(x, fontsize=10)
    # plt.yticks(range(0,100,20),fontsize=25)
    plt.title(title,fontsize=10)

    plt.ylabel(ty_,fontsize=10)
    plt.xlabel("Number of epochs",fontsize=10)

    # Save figure
    file_name = file.split("/")[-1].replace('.json','_'+ty_+'.png')
    file_name = os.path.join(save_fld, file_name)
    plt.savefig(file_name)
    # plt.show()
    plt.close()



parser = argparse.ArgumentParser()
parser.add_argument(
        '--folder', type=str, help='folder containing evaluation data',required=True)
args = parser.parse_args()
# -------------------------------
# attributes for evalution
# -------------------------------
attr_ = {0:"folder", 1:'train_loss', 2:'val_loss', 3:'val_top1_acc', 4:'val_top5_acc', 5:'n_epochs'}
# -------------------------------------

files_ = glob.glob(os.path.join(args.folder,"*.json"))
# print(files_)
for file in files_:
    
    with open(file) as j_file:
        data = json.load(j_file)
    train_loss, val_loss, val_top1_acc, val_top5_acc   = data[attr_[1]], data[attr_[2]], \
                                                         data[attr_[3]], data[attr_[4]]
    plt_atr(file, (attr_[1],train_loss), (attr_[2],val_loss),data[attr_[0]],"Loss")
    plt_atr(file, (attr_[3],val_top1_acc), (attr_[4],val_top5_acc),data[attr_[0]],"Accuracy")
    



    






