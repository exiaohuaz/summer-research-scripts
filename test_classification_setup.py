import xml.etree.ElementTree as ET
import os

from PIL import Image

SRC = "test_raw"
DST = "test"

classcounts = {"back": 0, "front": 0, "fronttwo": 0, "full": 0, "mid": 0}

def main():
  for folder in os.listdir(SRC):
    print(folder)
    tree = ET.parse(os.path.join(SRC, folder, 'annotations.xml'))
    root = tree.getroot()
    for item in root.findall("image"):
      bbox = item[0]
      left = float(bbox.attrib["xtl"])
      right = float(bbox.attrib["xbr"])
      top = float(bbox.attrib["ytl"])
      bottom = float(bbox.attrib["ybr"])

      filename = item.attrib["name"] + ".png.jpg"
      image_path = os.path.join(SRC, folder, "images", filename)
      img = Image.open(image_path)
      
      img_cropped = img.crop((left, top, right, bottom))

      class_name = bbox.attrib["label"]
      fnum = f'{classcounts[class_name]:05d}'
      if not os.path.isdir(os.path.join(DST, class_name)):
        os.mkdir(os.path.join(DST, class_name))
      img_cropped.save(os.path.join(DST, class_name, class_name + "_" + fnum + ".jpg"))
      classcounts[class_name] += 1


if __name__ == '__main__':
  main()
