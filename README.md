# ContextEncoder4Inpainting

## Running Training

The training can be ran either by using main.py or context_encoder_for_inpainting.ipynb (jupyter notebook).

The system supports training for a specific kind of task on every run of the notebook.
The specific tasks is dictated bu the constants in config.py. The main configurations to set 
a specific training are:

- DATASET_SELECT : can be either 'photo' or 'monet'
- MASKING_METHOD : cab be "RandomBlock" / "CentralRegion" / "RandomRegion"
- MASK_SIZE : we tested for 64 and 32. In theory it can be a smaller power of 2

These main constants select the various tasks we trained, i.e. specific mask type for specific dataset.

## Inference Module

The inference module is inference_module.py. It features a function called infer_inpainting which can be used to apply inpainting for a given input image and mask image. The user should specify the model to be used : either "photo" (Photo dataset trained model) or "monet" (Monet dataset trained model).

Note that the model loaded is always the one trained for the random region mask as this is the most generalized one.
