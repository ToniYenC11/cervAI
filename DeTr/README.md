# Detection Transformer

The Detection Transformer is a transformer-based model, and hence, the training and validation for this model are not stored in separate scripts, but in a notebook with different explanations for the various parts of the pipeline. The notebook is named `DeTr_v2.ipynb`. For uniformity and rigidity in the documentation, some parts of the code will be reiterated here.

## Training

### Data Preparation

Initially, three datasets are used for the training. These three datasets are combined with each other, and all labels are re-labelled for uniformity between *Positive* and *Negative*. However, all three datasets are the same source (IARC Cervical Cancer Image Bank) and moreover, the augmentations are done primarily in **Roboflow**. For the sake of training on a new dataset, you can define the transformations inside the dataset class 

```Python
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize()]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize()])}

train_dataset = CocoDetection(img_folder='./datasets/train', processor=processor,transform=data_transform['train'])
val_dataset = CocoDetection(img_folder='./datasets/test', processor=processor, train=False, data_transform=data_transform['test'])
```

### Training Using Pytorch Lightning

`pytorch_lightning` is a wrapper for pytorch for the training pipeline, transforming the definition of pytorch pipeline into something similar to tensorflow where running the pipeline is as simple as calling `trainer.fit`.

To install `pytorch_lightning`, simply type this in the terminal:

```bash
pip install pytorch_lightning
```

The `MetricsLogger` logs the metrics it is defined. However, this is inefficient and to have a more efficient logging of every aspects of the training, we can replace this with MLFLow's MLFlowLogger. Replace it with the following line:

```Python
from lightning.pytorch.loggers import MLFlowLogger
metrics_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
```

and in the `trainer` in the next snippet of the notebook, simply call it as this instead:

```Python
trainer = Trainer(
    max_epochs=100,
    gradient_clip_val=0.1,
    logger=[tensorboard_logger],  
    callbacks=[checkpoint_callback, lr_monitor, metrics_logger],
    log_every_n_steps=1,  
    val_check_interval=50, 
    enable_progress_bar=True,
    enable_model_summary=True,
    precision=16,  
    logger = metrics_logger #This added line allows us to track all the logging.
)
trainer.fit(model)
```

Finally, you can change the directory to save the model by modifying this line:

```Python
model_save_dir = "./saved_models/detr-finetuned-cerv_AI-v2"
```

## Validation

The validation of the training is found within the `trainer`. Opening the tensorboard will show the training and validation loss as well as the performance in terms of coco metrics. The dataset does not have its own testing set, hence it is better to apply inference for outside datasets instead. This will result in unlabeled testing for the model.

## Loading Pretrained Weights 

Loading pretrained weights is as simple as the following:

```Python
model = AutoModelForObjectDetection.from_pretrained("./saved_models/detr-finetuned-cerv_AI-v2", local_files_only=True)
processor = AutoImageProcessor.from_pretrained("./saved_models/detr-finetuned-cerv_AI-v2", local_files_only=True)
```

The `local_files_only` is an argument that will only search for the directory locally. It may search the hugging face repositories if not set to `True`. However, the pretrained weights are too big for GitHub to be stored (and Git LFS does not work for my machine). Just recall from the training that your model after being trained is saved in `model_save_dir`.

## Inference

`visualize_predictions` allows you to understand the prediction of the model on your external dataset. The validation of this function involves a domain expert who will validate the performance of the model itself on the image. As such, in the case of poor performance, the validation can become a new source of training data that may improve the model for its next iteration. 

`compare_predictions_with_gt` compares a dataset path with its own ground truth. This is more appropriate for validation sets, but may be used for other dataset in case of comparison versus other models.

## **Attention Rollout**: Grad-CAM for Transformers

Attention Rollout is a visualization technique for transformer models based from [this paper](https://arxiv.org/pdf/2005.00928). It is similar in the sense to Grad-CAM, but involves visualizing the attention layer, in the case of DeTr the self-attention layer. It applies a **gaussian filter** to generate a heatmap. The plot will show four quadrants:

1. Upper left quadrant shows the original image

2. Upper right quadrant shows the generated heatmap for the attention 

3. Lower left quadrant shows the imposing of the heatmap with the original image

4. Lower right shows the histogram of the weights distribution for the attention.

For more information on the attention rollout itself, you can visit [this article](https://medium.com/@nivonl/exploring-visual-attention-in-transformer-models-ab538c06083a).

---
##### Written By:

Toni "Sniper" Yenisei Czar S. Castañares