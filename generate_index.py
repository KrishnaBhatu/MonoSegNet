import os
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--pred_dir',   type=str,   help='path to the pred directory', default='/home/adi_leo96_av/test_output')
parser.add_argument('--label_dir',   type=str,   help='path to the pred directory', default='/home/adi_leo96_av/training_labels/gtCoarse/train_val')

args = parser.parse_args()


folder_list = []
for folder_name in glob.glob(args.pred_dir + "/*"):
    folder_list.append(folder_name[len(args.pred_dir):])

#print(folder_list)
f = open("/home/adi_leo96_av/MonoSegNet/test_index.txt", 'w')

for folder in folder_list:
    for image_path in glob.glob(args.pred_dir+folder+"/*.png"):
        f.write(image_path + ' ' + args.label_dir + "/" + os.path.basename(image_path)[:-19]+"gtCoarse_labelIds.png"+'\n')
    #print(os.path.basename(i)[:-19])    
f.close
#print(list_q)
