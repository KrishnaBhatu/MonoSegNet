import os

f= open("/home/adi_leo96_av/training_index.txt", 'r')

lines = f.readlines()
for line in lines:
  new_line = line.rsplit('\n')
  print(new_line)
  for subline in new_line.split():
    if(not os.path.exists(subline)):
      print(subline)
  

