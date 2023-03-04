

from PIL import Image
import glob, os

path = "./"
os.chdir(path)
counter = 0
print(os.listdir("."))
for folders in os.listdir("./"):
    for files in glob.glob(folders+"/all_img/" +"*.png"):
        print(files)
        #img = Image.open(files)
    
        #counter += 1
        #img.save( 'all_img/' + str(counter) + '.png', 'png')
