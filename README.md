# Vision transformer attention with multi-reservoir echo state network for anomaly recognition
 Video anomaly recognition in smart cities is an important computer vision task that plays a vital role in smart surveillance and public safety but is challenging due to its diverse, complex, and infrequent occurrence in real-time surveillance environments. Various deep learning models use significant amounts of training data without generalization abilities and with huge time complexity. To overcome these problems, in the current work, we present an efficient light-weight convolutional neural network (CNN)-based anomaly recognition framework that is functional in a surveillance environment with reduced time complexity. We extract spatial CNN features from a series of video frames and feed them to the proposed residual attention-based long short-term memory (LSTM) network, which can precisely recognize anomalous activity in surveillance videos. The representative CNN features with the residual blocks concept in LSTM for sequence learning prove to be effective for anomaly detection and recognition, validating our model’s effective usage in smart cities video surveillance. Extensive experiments on the real-world benchmark UCF-Crime dataset validate the effectiveness of the proposed model within complex surveillance environments and demonstrate that our proposed model outperforms state-of-the-art models with a 1.77%, 0.76%, and 8.62% increase in accuracy on the UCF-Crime, UMN and Avenue datasets, respectively.

# Anomalies recognition 
This work has been published in Information Processing & Management journal.
The title of the paper is "Vision transformer attention with multi-reservoir echo state network for anomaly recognition"

# Required packages

Python: Python 3.6 or higher
NumPy: NumPy (Recommended version: 1.19.5 or higher)
TensorFlow: TensorFlow 2.x (Recommended version: 2.4.1 or higher)
Keras: Keras is included with TensorFlow. The code should work with the Keras version that comes with TensorFlow 2.x.
NumPy: NumPy (Recommended version: 1.19.5 or higher)
scikit-learn: scikit-learn (Recommended version: 0.24.2 or higher)
SciPy: SciPy (Recommended version: 1.7.1 or higher)
Matplotlib: Matplotlib (Recommended version: 3.4.3 or higher)
echo-state-network (ESN) library (if used, make sure it is installed)

Please note that the vit_keras module may not be a standard package but rather a custom implementation or a local file in the project directory.




@article{ullah2023vision,
  title={Vision transformer attention with multi-reservoir echo state network for anomaly recognition},
  author={Ullah, Waseem and Hussain, Tanveer and Baik, Sung Wook},
  journal={Information Processing \& Management},
  volume={60},
  number={3},
  pages={103289},
  year={2023},
  publisher={Elsevier}
}