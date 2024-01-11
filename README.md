# Satellite Image Building Area Segmentation
This is the code for DACON SW중심대학 공동 AI 경진대 2023
* Link: https://dacon.io/competitions/official/236092/overview/description
* Data: https://dacon.io/competitions/official/236092/data
* base code: https://dacon.io/competitions/official/236092/codeshare/8465?page=1&dtype=recent


## My solution is based on Unet++ CNN models and below you will find description of full pipeline.
### Summary of Model
The First stage was ensemble of Unet++ models trained on noisy labels with hard image augmentations.
The next two stages were trained on noisy train data and test data that was made by earlier stage.

#### Data Augmentation
* train_transform_1() : RandomCrop, Flip, RandomBrightnessContrast
* train_transform_2() : RandomScale, PadIfNeeded, RandomCrop, Flip, Downscale, one of color, noise transforms
* train_transform_3() : RandomScale, PadIfNeeded, RandomCrop, Flip, Downscale, one of color, noise transforms
* train_transform_4() : ShiftScaleRotate, PadIfNeeded, RandomCrop, Flip, Downscale, MaskDropout, one of color, noise, distortion transforms
#### Train Model
* Stage 1
  * applying "5 fold cross-validation" per Model so that We could get 10 weights
    * Model: Unet++
    * Encoder_name = efficientnet-b7
    * encoder_weights = imagenet
    * classes = 1
    * lr = 0.0001
    * num_epochs = 50
    * optimizer : Adam
    -----------------------
    * Model: Unet++
    * Encoder_name = se_resnext101_32x4d
    * encoder_weights = imagenet
    * classes = 1
    * lr = 0.0001
    * num_epochs = 50
    * optimizer : Adam
 
  * "Ensemble the 10 models with average, threshold 0.5" 
  * We can get new dataset from stage 1 to predict test data

* Stage 2
  * same as Stage 1, but different "using dataset (from stage 1 + train_dataset)" 

* Stage 3
  * applying 5 fold cross-validation per Model so that We could get 10 weights
  * We select 5 weights from below
  * "Ensemble the 10 models with average, threshold 0.5" 
    * Model: Unet++
    * Encoder_name = efficientnet-b7
    * encoder_weights = imagenet
    * classes = 1
    * lr = 0.0001
    * num_epochs = 50
    * optimizer : Adam
    -----------------------
    * Model: Unet++
    * Encoder_name = se_resnext101_32x4d
    * encoder_weights = imagenet
    * classes = 1
    * lr = 0.0001
    * num_epochs = 50
    * optimizer : Adam
    -----------------------
    * Model: Unet++
    * Encoder_name = inceptionresnetv2
    * encoder_weights = imagenet
    * classes = 1
    * lr = 0.0001
    * num_epochs = 50
    * optimizer : Adam
  
Now, we really predict test data through 3 stages.

#### Result


