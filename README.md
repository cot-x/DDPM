# DDPM
Unofficial custmized DenoisingDiffusionProbabilityModel.

## python DDPM.py --help
```
usage: DDPM.py [-h] [--image_dir IMAGE_DIR] [--result_dir RESULT_DIR]
               [--weight_dir WEIGHT_DIR] [--image_size IMAGE_SIZE] [--lr LR]
               [--beta_1 BETA_1] [--beta_t BETA_T] [--num_times NUM_TIMES]
               [--dim_hidden DIM_HIDDEN] [--batch_size BATCH_SIZE]
               [--num_train NUM_TRAIN] [--cpu] [--generate GENERATE]
               [--noresume]

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
  --result_dir RESULT_DIR
  --weight_dir WEIGHT_DIR
  --image_size IMAGE_SIZE
  --lr LR
  --beta_1 BETA_1
  --beta_t BETA_T
  --num_times NUM_TIMES
  --dim_hidden DIM_HIDDEN
  --batch_size BATCH_SIZE
  --num_train NUM_TRAIN
  --cpu
  --generate GENERATE
```

### for example
```
python DDPM.py --image_dir "/usr/share/datasets/image_dir"
```
and
```
python DDPM.py --generate 10
```

**Note:**
- If a weight.pth file exists in the current directory, the network weights will be automatically read.
