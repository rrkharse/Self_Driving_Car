# Self_Driving_Car

1. Clone this repo
2. Install Conda if not already installed (Python 3.6 version)

   https://www.anaconda.com/download/
3. Download and extract Udacity's Self-Driving Car Simulator

   [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip)  
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)  
[Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)  
4. run udacity driving simulator in training mode and copy output driving_log.csv to ./data folder in same directory as this cloned repo
5. create conda environment using .yml files
6. run model_tf.py to train. models are output to this repo folder
7. start udacity driving simulator and enter autonomous mode
8. run drive_tf.py <model_name> to connect model to udacity driving simulator
  Note: model name is xy...z.meta without ".meta"

