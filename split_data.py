import os
import shutil

SRC = 'classification/train'
DST = 'split_data'

NUM_TRAIN = 8000
NUM_VAL = 1889
#NUM_TEST = 0


def main():
    train_dir = os.path.join(DST, 'train')
    os.mkdir(train_dir)

    val_dir = os.path.join(DST, 'val')
    os.mkdir(val_dir)

    """test_dir = os.path.join(DST, 'test')
    os.mkdir(test_dir)"""

    for class_label in os.listdir(SRC):
        train_dir_class = os.path.join(train_dir, class_label)
        os.mkdir(train_dir_class)

        src_dir_class = os.path.join(SRC, class_label)
        image_paths = iter(sorted(os.listdir(src_dir_class)))

        for i in range(NUM_TRAIN):
            src_path = os.path.join(src_dir_class, next(image_paths))
            shutil.copy(src_path, train_dir_class)

        val_dir_class = os.path.join(val_dir, class_label)
        os.mkdir(val_dir_class)

        for i in range(NUM_VAL):
            src_path = os.path.join(src_dir_class, next(image_paths))
            shutil.copy(src_path, val_dir_class)



if __name__ == '__main__':
    main()