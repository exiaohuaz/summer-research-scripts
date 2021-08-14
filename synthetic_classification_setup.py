import json
import os
from PIL import Image

capture_dir = "captures/"
train_or_val = "train/"
parent_name = "classification/"

def main():
  folder_name = os.path.join(parent_name, train_or_val)
  if(not os.path.isdir(parent_name)):
    os.mkdir(parent_name)
  if(not os.path.isdir(folder_name)):
    os.mkdir(folder_name)

  #classes = {}
  classcounts = {}

  for capture in os.listdir(capture_dir):
      #print(capture)
      with open(os.path.join(capture_dir, capture), 'r') as f1:
        data = json.load(f1)
        data = data["captures"]

        for item in data:

          image_name = item["filename"] + ".jpg"
          img = Image.open(image_name)
          values = item["annotations"][0]["values"]
          for value in values:
            
            class_name = value["label_name"]
            class_name = class_name[:len(class_name)-1]
            #print(class_name)

            #lid = 0

            #if class_name not in classes:
            if class_name not in classcounts:
              #lid = len(classes) + 1
              #classes[class_name] = lid
              classcounts[class_name] = 1
              #class_folder = os.path.join(folder_name, "class" + str(lid))
              class_folder = os.path.join(folder_name, class_name)
              if not os.path.isdir(class_folder):
                os.mkdir(class_folder)

            left = value["x"]
            top = value["y"]
            right = value["x"] + value["width"]
            bottom = value["y"] + value["height"]
            newimg = img.crop((left, top, right, bottom))
            fnum = f'{classcounts[class_name]:05d}'
            newimg.save(os.path.join(folder_name, class_name, class_name + "_" + fnum + ".jpg"))
            classcounts[class_name] += 1
          
if __name__ == '__main__':
  main()
