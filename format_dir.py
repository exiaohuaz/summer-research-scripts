import os
from zipfile import ZipFile

ZIP_DIR = 'test/' #slash is important since i dont do os.join

def main():
  for zipname in glob.glob(ZIPDIR + "*.zip"):
    with ZipFile('zipname', 'r') as zip:
      zip.extractall(ZIPDIR + "input_" + zipname)
      print("Unzipped ", zipname)
