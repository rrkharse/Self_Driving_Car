# Self Driving Car in TensorFlow!

Video tutorial link:

[![IMAGE ALT TEXT](https://img.youtube.com/vi/6YC5p-r-Xhg/0.jpg)](http://www.youtube.com/watch?v=6YC5p-r-Xhg "Video Tutorial")
https://img.youtube.com/vi/6YC5p-r-Xhg/0.jpg

1. Clone this repo
2. Install Conda if not already installed (Python 3.6 version)

   https://www.anaconda.com/download/
3. Download and extract Udacity's Self-Driving Car Simulator

   [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip)  
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)  
[Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)  
4. Run Udacity's driving simulator in training mode

   When prompted where to store recorded data, create a folder called "data" in this cloned repo folder and save there.
5. Create conda environment

   Non-GPU users: `conda env create -f environments.yml`  
   GPU users: `conda env create -f environment-gpu.yml`
6. Run model_tf.py to train. models are output to this repo folder

   `python model_tf.py -N [A model name of your choosing]`  
   Run `python model_tf.py --help` for more options
7. Start Udacity's driving simulator and enter autonomous mode
8. Run `python drive_tf.py [model_name]` to connect model to Udacity's driving simulator

   Note: model name is xy...z.meta without ".meta"

