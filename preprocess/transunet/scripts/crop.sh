python transunetval.py --model_path /path/to/model --input_folder /path/to/input/folder --output_folder /path/to/output/folder
python crop_by_mask.py -O /path/to/originals -S /path/to/masks -D /path/to/output
python grayscale.py /path/to/input/folder /path/to/output/folder