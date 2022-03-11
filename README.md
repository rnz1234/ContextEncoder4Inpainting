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

Use cases:
- For interactive training with intermediate results (loss prints + images prints), best to run the jupyter notebook (after setting the above constants as wanted for specific required task)
- While notebook is running, training metrics (losses) can be observed using Tensorboard (we stream into it). In order to run it within Colab, need to add the following to the notebook:
```
%load_ext tensorboard
%tensorboard --logdir logs
```
If running outside colab, just run this in the project root directory:
```
tensorboard --logdir logs
```
and open browser on http://localhost:6006

## Inference Module

The inference module is inference_module.py. It features a function called infer_inpainting which can be used to apply inpainting for a given input image and mask image. The user should specify the model to be used : either "photo" (Photo dataset trained model) or "monet" (Monet dataset trained model).

Note that the model loaded is always the one trained for the random region mask as this is the most generalized one. However, we have the final models we trained for every variation under 
models/<dataset name>/good_model_<task type>

## Running Inference

We also have a jupyter notebook called context_encoder_inference_on_test.ipynb that run the inference module over all images supplied as test (under data/test).

You can just run it and observe the results within the notebook.