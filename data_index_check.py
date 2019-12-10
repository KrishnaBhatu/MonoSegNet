import os
import matplotlib.pyplot as plt
import matplotlib.image as mimg


f= open("/home/adi_leo96_av/training_index.txt", 'r')

lines = f.readlines()
for line in lines:
  new_line = line.rsplit('\n')
  #print(new_line)
  for subline in new_line[0].split():
      #print(subline)
      image = mimg.imread(subline)
      print(image.shape)
      if(not os.path.exists(subline)):
      	print(subline)

if(os.path.exists("/home/adi_leo96_av/training_labels/gtCoarse/train_extra/konigswinter/tamatar.png")):
  print("Tamatar")
  

