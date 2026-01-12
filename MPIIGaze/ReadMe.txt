=======================================================================================

This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License.
See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.
The full text of the license can be found at
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

BY DOWNLOADING AND USING THIS DATASET YOU AGREE TO THE TERMS OF THE ABOVE LICENSE
AND TO CITE THE ABOVE PAPERS IN YOUR OWN WORK.

You are free:
 * to Share - to copy, distribute and transmit the work
 * to Remix - to adapt the work

Under the following conditions:
* Attribution - You must attribute the work in the manner specified by the author
  or licensor (but not in any way that suggests that they endorse you or your use of the work).
* Noncommercial - You may not use this work for commercial purposes.
* ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

=======================================================================================

In case of questions please feel free to contact:
Andreas Bulling <andreas.bulling@vis.uni-stuttgart.de>
http://perceptualui.org/

=======================================================================================

Here you can find the contents for the MPIIGaze dataset.

Data: The main samples in original and normalized format.

Evaluation Subset:
We used an evaluation subset of MPIIGaze for the experiments in our paper, where we randomly select 3000 samples (1500 left eyes and 1500 righ eyes) for each participants. The txt files include the image file name and "left" or "right" eye we used in the image.

Annotation Subset:
We also annotated 10,848 images with 12 facial landmarks, face bounding box, two eye bounding boxes, and pupil position.

=======================================================================================
Collection Procedure:
We implemented custom software running as a background service on participants’ laptops. Every 10 minutes the software automatically asked participants to look at a random sequence of 20 on-screen positions (a recording session), visualized as a grey circle shrinking in size and with a white dot in the middle. Participants were asked to fixate on these dots and confirm each by pressing the spacebar once the circle was about to disappear. This was to ensure participants concentrated on the task and fixated exactly at the intended on-screen positions. No other instructions were given to them, in particular no constraints as to how and where to use their laptops.

Characteristics:
We collected a total of 213,659 images from 15 participants. The number of images collected by each participant varied from 34,745 to 1,498.

Dataset
We present the MPIIGaze dataset that contains 213,659 images that we collected from 15 participants during natural everyday laptop use over more than three months. The number of images collected by each participant varied from 34,745 to 1,498. Our dataset is significantly more variable than existing ones with respect to appearance and illumination.

The dataset contains three parts: “Data”, “Evaluation Subset” and “Annotation subset”.

* The “Data” includes “Original”, “Normalized” and “Calibration” for all the 15 participants.
* The “Evaluation Subset” contains the image list that indicates the selected samples for the evaluation subset in our paper.
* The “Annotation Subset” contains the image list that indicates 10,848 samples that we manually annotated, following the annotations with (x, y) position of 6 facial landmarks (four eye corners, two mouth corners) and (x,y) position of two pupil centers for each of above images.

The dataset contains three parts: "Data'', "Evaluation Subset'' and "Annotation subset''. The "Data'' folder includes "Original'' and "Normalized'' for all the 15 participants. You can also find the 6 points-based face model we used in this dataset.

The "Original'' folders are the cropped eye rectangle images with the detection results based on face detector [1] and facial landmark detector [2]. For each participants, the images and annotations are organized by days. For each day's folder, there are the image collected by that participants and corresponding "annotation.txt" files. The annotations includes:

* Dimension 1~24: Detected eye landmarks position in pixel on the whole image coordinate
* Dimension 25~26: On-screen gaze target position in screen coordinate
* Dimension 27~29: 3D gaze target position related to camera
* Dimension 30~35: The estimated 3D head pose based on 6 points-based 3D face model, rotation and translation: we implement the same 6 points-based 3D face model in [3], which includes the four eye corners and two mouth corners
* Dimension 36~38: The estimated 3D right eye center in the camera coordiante system.
* Dimension 39~41: The estimated 3D left eye center in the camera cooridnate system.

Besides, there is also "Calibration" folder for each participants, which contains:

* Camera.mat: the intrinsic parameter of the laptop camera. "cameraMatrix": the projection matrix of the camera. "distCoeffs": camera distortion coefficients. "retval": root mean square (RMS) re-projection error. "rvecs": the rotation vectors. "tvecs": the translation vectors.
* monitorPose.mat: the position of image plane in camera coordinate. "rvecs": the rotation vectors. "tvecs": the translation vectors.
* screenSize.mat: the laptop screen size. "height_pixel": the screen height in pixel. "width_pixel": the screen width in pixel. "height_mm": the screen height in millimeter. "width_mm": the screen widht in millimeter.

The "Normalized'' folders are the eye patch images after the normalization that canceling scaling and rotation via perspective transformation in Sugano et al. [3]. Similar to the "Original'' folders, all the data are organized by each days for each participants, and the file format is ".mat". The annotation includes: * 3D gaze head pose and 3D gaze direction. The generation of 2D screen gaze target to this 3D gaze direction is described in our paper.

The folder "Evaluation Subset'' contains:

* The image list that indicates the selected samples for the evaluation subset in our paper. We performed evaluations on this evaluation subset of our MPIIGaze dataset, which includes equal number of samples for each participants.

The folder "Annotation Subset'' contains:

* The image list that indicates 10,848 samples that we manually annotated
* Following the annotations with (x, y) position of 6 facial landmarks (four eye corners, two mouth corners) and (x,y) position of two pupil centers for each of above images.

=======================================================================================

Q&A
How do you convert .mat file to .h5 file?
Please find the example Matlab script here: https://perceptualui.org/files/datasets/mat2h5.zip

How do you convert 3d directional vector to 2d angle?
We refer to the paper [3] for the data normalization. Briefly to say, the 3D gaze direction (x, y, z) can be converted to 2D representation (theta, phi) like:

theta = asin(-y)
phi = atan2(-x, -z)
The negative representation has been used so that camera-looking direction becomes (0,0).
And in contrast, 3D head rotation (x, y, z) can be converted to (theta, phi) like:

M = Rodrigues((x,y,z))
Zv = (the third column of M)
theta = asin(Zv[1])
phi = atan2(Zv[0], Zv[2])

[3] Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-synthesis for appearance-based 3d gaze estimation. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1821–1828. IEEE, 2014.

Why did I get "nan" during training?
Usually, there are two reasons to cause "nan". Firstly, it can be the value is out of the float type range if you got "nan" all the time. It can happen with inappropriate layer initialization. Since Caffe is keeping update, my configuration also can fail the layer initialization. So please modify the layer initialization parameter by yourself, like the "std" value. Secondly, it can be caused by calculation exception if you got "nan" from time to time. I modify the accuracy layer and euclidean loss layer to report the "angle difference", where the function "acos" is been called. It sometimes output "nan" because the variable is out the range of (-1,1). However, it is just for showing, so that wouldn't affect the training.

How can I do data normalization?
Please refer to this example code (https://perceptualui.org/files/datasets/data_normalization_code.zip) for how to convert the data from "Original" to "Normalized".

=======================================================================================

Please also see this link for more information and links to the papers: https://perceptualui.org/research/datasets/MPIIGaze/