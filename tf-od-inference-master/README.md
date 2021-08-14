To convert from checkpoint files to a saved model, run
```bash
python object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path <PATH TO pipeline.config> \
--trained_checkpoint_dir <Path to model_dir> --output_directory <Where to save model>
```

from `~/models/research` in the object detection Docker container.

Run these servers with the clients in
https://github.com/cmusatyalab/gabriel/tree/master/examples/round_trip
