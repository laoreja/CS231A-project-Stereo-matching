# CS231A Course project

## Reimplementation-of-GC-Net

I mainly reimplement the GC-Net https://arxiv.org/pdf/1703.04309.pdf.

I implement two versions of the GC-Net model: one with a mask (the losses are masked), and one without the mask.

## Results

### Qualitative results

The without-mask-version, original images and predictions samples on SceneFlow:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/gt_2.png" width=400px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/pre_2.png" width=400px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/gt_4.png" width=400px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/pre_4.png" width=400px/>

The with-mask-version, masked ground truth, masked predictions, and unmasked predictions on SceneFlow:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/gt_masked_1.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_masked_1.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_1.png" width=290px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/gt_masked_2.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_masked_2.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_2.png" width=290px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/gt_masked_3.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_masked_3.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_3.png" width=290px/>

On KITTI training set, ground truth, masked predictions, and unmasked predictions:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_train/gt_1.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_train/predict_masked_1.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_train/predict_1.png" width=290px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_train/gt_2.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_train/predict_masked_2.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_train/predict_2.png" width=290px/>

On KITTI testing set, original image and prediction samples:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_test/left_1.png" width=400px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_test/pre_1.png" width=400px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_test/left_2.png" width=400px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/KITTI_test/pre_2.png" width=400px/>

### Quantitative results

Since the KITTI dataset is very sparse, the provided groundtruths are with masks, I implement and train the with mask version first. But I found some predictions are very blurry. (My masks are a bit too much). 

Though the qualitative results looks fine, but the quantitative results on SceneFlow test set is not very good. As for KITTI, this is not a satisfying version, so I do not submit it. And due to the limit of time and resources, I do not do validation. I give the quantitative results on the training set for reference. It is very strange that this results are good, maybe overfitting exists, but I train it for less iterations than that is reported in the paper.

| Dataset        | MAE(px) | >3px | >5px  | >7px  |
| :------------- |:----:|:-----:|:-----:|:-------:| 
| SceneFlow test | 14.04 | 0.66 | 0.53 | 0.45
| KITTI train    | 2.86  | 0.28 | 0.146 | 0.0866|

## Run

The scripts whose names contain 'KITTI' are for the KITTI dataset, others are for SceneFlow.

The scripts whose names contain 'no_mask' are for the no-mask-version, others are for the with mask version. The image data building step is not affected. 


First, you need to run the `build_image_data_xxx.py` to convert the data into tfrecords. 
For SceneFlow, you need to run `SceneFlow_divide_test_train.py` first to divide the train/test partitions and save the info in a pickle. 
If you download the webp formatted data for SceneFlow, you need to run `web2png.py` to transform the image format. And you will need the dwebp tool.

To train, for example, train on SceneFlow with 4 GPUS:

```
python gcnet_multi_gpu_train.py --log_root xxx --num_gpus 4 --max_steps 375000 --mode train 2>&1 | tee train_log
```

The train script also support resuming, you need to identify the checkpoint path and change the mode.

To evaluate, 
```
python gcnet_eval.py --log_root xxx --checkpoint_dir xxx --run_once True 2>&1 | tee train_log
```

To summarize, here is the list of all the source code files' use:

| Filename        | Use  |
| :------------- |:----:|
| webp2png.py | Transform the image format for SceneFlow. |
| pfm.py | Load/Save pfm disparity information for Sceneflow. |
| uint16png.py | Load uint16 png file for KITTI. |
| timeline.py | Obtain the timeline.json to analysis the time usage.|
| SceneFlow_divide_test_train.py | Divide the Sceneflow dataset into train and test subsets, saving in a pickle. |
| build_image_data_xxx.py | Transfer xxx dataset into tfrecords |
| dataset.py | The abstract dataset class. |
| xxx_data.py | The inherited xxx dataset class. |
| image_processing_xxx[_no_mask].py | Image processing module, build input for the model. |
| test_image_processing[_KITTI].py | Test the image processing module. |
| gcnet_model[_no_mask].py | The model definition. |
| gcnet_train.py | Train on a single CPU or GPU. |
| gcnet_multi_gpu_train[_KITTI/_no_mask].py | Train on multiple GPUs. |
|gcnet_multi_gpu_eval[_no_mask].py | Eval on multiple GPUs. |


## Problems I met 

* For the with-mask-version (the without-mask-version is still training), During evaluation, if using a fixed mean and variance (during training, we would estimate a mean and variance, this is what the Resnet model do), the results are terrible. If changing that, using the real-time mean and variance, results are better.

One bad results:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/wrong_left.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/wrong_gt.png" width=290px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/wrong_predict.png" width=290px/>

However, using fixed mean and variance should be the more common practice, I don't know what's goes on.

* The GPU memory problem, in the original paper, one place is very confusing, in Section 4.1, they first say they use a 256*512 crop, then they set the H, W to the image size. But H and W exist in Table 1, which is the network's input size, so I am very confused about what they input to the network.

Besides, if I use H=540 and W=960, the GPU memory is not enough. So what I met is that, though the network is fully convolutional, I cannot feed the whole image in.

The GPU I use is TITAN X, 12G memory is not small. I wonder how the authors manage to do so (they also say they use TITAN X GPU).

* The evaluation runs extremely slowly on CPU, but runs well on GPU. I located the problem to the last `conv3d_transpose` layer. I think this is a problem with TensorFlow, so I raised an [issue here](https://github.com/tensorflow/tensorflow/issues/10535).

<!-- Since it's my first time using TensorFlow, I met lots of problems. In my experience playing with other frameworks, the common practice is to do validation after training some epochs. But for TensorFlow, someone says it's better and safer using separate processes, and we can use CPU to do validation. If the CPU performance is not good, then it will work -->

## More details

For more details, please refer to the `final_report.pdf` and `supplemental.pdf` in this repo.

Since I sorted the files recently, there may exist some bugs, feel free to report them.
