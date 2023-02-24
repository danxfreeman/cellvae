# A VAE for cell encoding

## Setup

* Create Conda environment with `conda create --name cellvae_env --file=cellvae_env.yml`
* Activate Conda environment with `conda activate cellvae_env`

## Instructions

* Update `config.json`
* Run `python run.py <PATH_TO_CONFIG>`

## Parameters

#### Uploading

* `markers`: Path to CSV containing channel metadata (must contain columns `marker_name` and `channel_number`)
* `img`: Path to TIFF image
* `csv`: Path to CSV containing cell coordinates
* `csv_cols`: Columns to use as cell coordiantes
* `output`: Directory to store results in

#### Preprocessing

* `min_quantile`: Clips pixels below this nonzero quantile (useful if image contains zeroes where the microscope skipped empty tiles)
* `max_quantile`: Clips pixels above this quantile (useful if image contains a few extremely bright pixels)
* `crop_size`: Width and height to crop around each cell (use a larger `crop_size` to incorporate more context)

#### Loading

* `train_size`: Proportion of cells to use for training
* `batch_size`: Number of cells to feed to the model at a time
* `shuffle`: Whether or not to reshuffle data at each epoch
* `workers`: Number of subprocesses used for data loading

#### Modeling

* `conv_dim`: Number of filters to use in each convolutional layer (and, implicitly, number of layers)
* `kernel_size`: Kernel size
* `stride`: Stride
* `latent_dim`: Number of latent dimensions
* `kld_anneal`: Whether or not to increase KLD weight over initial epochs (used to prevent posterior collapse)
* `kld_weight`: Constant used to scale KLD loss relative to MSE loss (can be zero)
* `cpu_threads`: Number of threads to use for inference on CPU
* `seed`: Use this value to reproduce results

## Notes

* Code is organized according the PyTorch convention described [here](https://hagerrady13.github.io/posts/2012/08/blog-post-1/)
* Model calculates mean MSE loss per pixel and mean KLD loss per latent variable: `mse_per_pix + kld_per_lat * kld_scaler`
