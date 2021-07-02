import os
import cv2
def readname():
    # filePath = './div2k'
    # filePath = './pristine_images'
    filePath = './BSD432_color'
    # filePath = './flower_2000'
    # filePath = './coco'
    # filePath = './train20172'
    # filePath = './trainB'
    name = os.listdir(filePath)
    return name, filePath

if __name__ == "__main__":
    name, filePath = readname()
    print(name)
    txt = open("trainA.txt", 'w')
    # txt = open("BSD432_test.txt", 'w')
    for i in name:
        # print(filePath + "/" + i)
        image_dir = os.path.join(filePath + "/", str(i))
        # image_dir = os.path.join('./pristine_images/', str(i))
        # image_dir = os.path.join('./flower_2000', str(i))
        img = cv2.imread(image_dir)
        h, w, c = img.shape
        # if w <= 256 or h <= 256:
        #     continue
        # else:
        #     txt.write(image_dir + "\n")
        txt.write(image_dir + "\n")
