# Pytorch-3D-R<sup>2</sup>N<sup>2</sup>: 3D Recurrent Reconstruction Neural Network

This is a Pytorch implementation of the paper ["3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction"](http://arxiv.org/abs/1604.00449) by Choy et al. Given one or multiple views of an object, the network generates voxelized ( a voxel is the 3D equivalent of a pixel) reconstruction of the object in 3D.  
See [chrischoy/3D-R2N2](http://github.com/chrischoy/3D-R2N2) for the original paper author's implementation in Theano, as well as overview of the method.

## Pre-trained model
For now, only the non-residual LSTM-based architecture with neighboring recurrent unit connection is implemented. It is called *3D-LSTM-3* in the paper.  
A pre-trained model based on this architecture can be downloaded from [here](https://mega.nz/file/BHQQVJ6D#zVukPkk1dXI4qnPxzz3naoYi1RUY6wKLcLiq3q90jPU). It obtains the following result on the ShapeNet rendered images test dataset:    
IoU | Loss |
--- | --- |
0.591 | 0.093 | 

## Installation
The code was tested with Python 3.6.  

- Download the repository
```
git clone https://github.com/alexgo1/pytorch-3d-r2n2.git
```

- Install the requirements
```
pip install -r requirements.txt
```

## Training the network

- Download and extract the ShapeNet rendered images dataset:  
```
mkdir ShapeNet/
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
tar -xzf ShapeNetRendering.tgz -C ShapeNet/
tar -xzf ShapeNetVox32.tgz -C ShapeNet/
```

- Rename the ```config.ini.example``` config template file to e.g ```your_config.ini```, and change parameters in it as required.
- Run ```python train.py --cfg=your_config.ini```. Or simply ```python train.py``` if you named your config file ```config.ini```.

## Test your trained model
- Run ```python test.py --cfg=your_config.ini```. Or simply ```python test.py``` if your config file is named ```config.ini```.  
  This can be the same config file used for training the model. Note that when testing, you probably want to set ```resume_epoch``` to the number of epochs that your model was trained for.

## License

MIT License

