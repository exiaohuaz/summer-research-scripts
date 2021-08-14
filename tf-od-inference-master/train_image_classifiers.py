import os
import subprocess

STATES = [
    ('1screw', 'noscrews'),
    ('noscrews', 'nopad'),
    ('nopad', 'noring'),
    ('noring', 'nocylinder'),
    ('nocylinder', 'nopiston'),
]

for state in STATES:
    dst_prefix = os.path.join('/sterling/model_training', '{}_{}'.format(*state))
    
    src_subdir = '/sterling/jan28_training_data/'
    dst_subdir = '{}_training_data'.format(dst_prefix)
    src = os.path.join(src_subdir, state[0])
    dst = os.path.join(dst_subdir, state[0])
    os.symlink(src, dst)
    src = os.path.join(src_subdir, state[1])
    dst = os.path.join(dst_subdir, state[1])
    os.symlink(src, dst)

    subprocess.check_call([
        'make_image_classifier', '--image_dir', dst_subdir,
        '--tfhub_module' 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
        '--image_size', '224', '--saved_model_dir', '{}_model'.format(dst_prefix),
        '--labels_output_file', '{}_labels.txt'.format(dst_prefix),
        '--summaries_dir', '{}_summary'.format(dst_prefix), '--train_epochs', '10'
    ])
