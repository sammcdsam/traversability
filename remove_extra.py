import os

file_an_path = r"E:\traversability\datasets\Rellis-3D\annotations"
file_an_list = os.listdir(file_an_path)
print (file_an_list)

file_img_path = r"E:\traversability\datasets\Rellis-3D\images"
file_img_list = os.listdir(file_img_path)
print (file_img_list)

for file_name in file_img_list:
    b = file_name.split('.')[0]+".png"

    if b in file_an_list:
        print("Yep cock")
    else:
        print("removed")
        os.remove(os.path.join(file_img_path, file_name))
    


'''
for file_name in file_list:
    if "(1)" not in file_name:
        continue
    original_file_name = file_name.replace('(1)', '')
    if not os.path.exists(os.path.join(file_path, original_file_name):
        continue  # do not remove files which have no original
    os.remove(os.path.join(file_path, file_name))
'''    