# ResNet and FractalNet 
A ResNet (91% accuracy) and FractalNet (92% accuracy) trained on CIFAR-10 using 1 Tesla K80 GPU and 4 CPUs 

#### Code Heirarchi:
##### 1. loading_data.ipynb

   Loads the downloaded data, preprocesses it and stores it in a numpy array for feeding into the neural network. This code should be run after downloading the CIFAR 10 data from [here](https://www.cs.toronto.edu/~kriz/cifar.html). 
   

##### 2. resnet_training_testing.ipynb and fractalnet_training_testing.ipynb

   Contains building model, training, and testing. All training information is provided in the respective notebooks.


##### 3. The fully trained model architectures and weights are provided. 

   The resnet model architecture :'resnet_model_epoch127_json.pkl'.

   The resnet trained weights: 'resnet_model_epoch127_weights.h5'
   
   The fractalnet model architecture:'fractal_model_epoch157_json.pkl'.

   The fractalnet trained weights: 'fractal_model_epoch157_weights.h5'
   
   The models can be reconstructed by using the following lines of code in keras after importing the required libraries:
   
   
```
   import keras
   import pickle 
   from keras.models import load_model
   from keras.models import model_from_json

   json_string = pickle.load( open( "resnet_model_epoch127_json.pkl", "rb" ) )
   model = model_from_json(json_string)
   model.load_weights('resnet_model_epoch127_weights.h5')
```
   
# Results 
## Loss history:

### Resnet: 

![loss_history](https://user-images.githubusercontent.com/18056877/37247169-528a648a-2485-11e8-9314-7a57829586ab.png)

### FractalNet:

![loss](https://user-images.githubusercontent.com/18056877/37561415-683e70b0-2a24-11e8-811f-ccf760d252de.png)

## Training accuracy:
### Resnet
![training_accuracy](https://user-images.githubusercontent.com/18056877/37247175-6e7c5f90-2485-11e8-8625-20d30b260d9f.png)
### FractalNet
![trainacc](https://user-images.githubusercontent.com/18056877/37561427-9d81d4ec-2a24-11e8-900e-1d107ca7c302.png)

## Testing accuracy:
### Resnet
![testing_accuracy](https://user-images.githubusercontent.com/18056877/37247178-77daca04-2485-11e8-8a3e-68364a027be6.png)
### FractalNet

![valacc](https://user-images.githubusercontent.com/18056877/37561433-b2f92b7c-2a24-11e8-9084-9c981f3537c7.png)


# Model Architectures
##### 1. The original resnet paper can be obtained [here](https://arxiv.org/abs/1512.03385).
The model architecture used here is shown below: 

![resnetv1_model](https://user-images.githubusercontent.com/18056877/37247163-194b92f2-2485-11e8-9a3d-2732ef511976.png)

##### 2. The original FractalNet paper can be obtained [here](https://arxiv.org/abs/1605.07648).
The model architecture used here (Number of columns = 4, Number of blocks = 3) is shown below: 

###### Fractal Block:

![fractal_block_picture](https://user-images.githubusercontent.com/18056877/37569882-fbc670c2-2abe-11e8-8c35-43bc59f131a7.png)

###### 3 of the above blocks concatenated: 

![fractalnet_model](https://user-images.githubusercontent.com/18056877/37561437-e3c9b762-2a24-11e8-9d87-21c33392558c.png)
