# CS231A Course project

## Reimplementation-of-GC-Net

I mainly reimplement the GC-Net https://arxiv.org/pdf/1703.04309.pdf.

I implement two versions of the GC-Net model: one with a mask (the losses are masked), and one without the mask.

## Results

### Qualitative results

The without-mask-version, original images and predictions on SceneFlow:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/gt_2.png" width=400px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/pre_2.png" width=400px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/gt_4.png" width=400px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_without_mask/pre_4.png" width=400px/>

The with-mask-version, masked ground truth, masked predictions, and unmasked predictions:

<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/gt_masked_1.png" width=300px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_masked_1.png" width=300px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_1.png" width=300px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/gt_masked_2.png" width=300px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_masked_2.png" width=300px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_2.png" width=300px/>
<img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/gt_masked_3.png" width=300px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_masked_3.png" width=300px/> <img src="https://raw.githubusercontent.com/laoreja/CS231A-project-stereo-matching/master/qualitative_results/SceneFlow_train_with_mask/predict_3.png" width=300px/>
