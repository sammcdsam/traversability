from PIL import Image
import glob, os




os.chdir("datasets/cityscape/frank_gray")
counter = 0
#print(os.listdir("."))
for files in os.listdir("./"):
        new = files.replace("_gtFine_labelIds.png", ".png")
        file_name =  new
        os.rename( files, new)
