# PETs2020 Attacks implementation

> This project contains the implementation of the Model Inversion Attack,Membership Inference Attack and Model Stealing Attack.

# Please follow following instruction to run the attacks :
>STEP 1 : Run the main.py file
```shell
python main.py
```
>STEP 2 : Select option by pressing number to execute attacks


## Model Inversion Attack
> Both target model are fully connected neural network with one hidden layer taking as input a gray scale image. MNIST are 28x28 
> and CIFAR are 32x32 pixels. Due to their simplicity and resemblance, the images of the MNIST database provide way better results than the CIFAR images (which 
> are not recognizable).
>
### Usage
> Prepare the neural networks for the inversion attack, this should generate two neural networks.
```shell
python NeuralNetworks/MI_CIFAR_generator.py
python NeuralNetworks/MI_MNIST_generator.py
```
> run the attack with the database name as the argument:
```shell
python Model_inversion_attack/ExampleAttack.py MNIST
python Model_inversion_attack/ExampleAttack.py CIFAR
```

> The result will be in the Model_inversion_attack/result.png file and an example of the database class will be in the Model_inversion_attack/test.png


## Membership Inference Attack
> The target model and the shadow model are each trained on 25% of the cifar training dataset.
> When running the generator, there are also two csv file created with the outputs of the shadow 
> and target models. The  attack model is trained and tested with the data from the .csv

###Usage
> Prepare the data for the Membership inference attack. Add --train if you want to train the target and shadow model. 
```shell
python NeuralNetworks/MIA_nn_generator.py [--train]
```

> Run the attack with --train if you want to train the attack model before evaluating:
```shell
python Membership_Inference_attack/ExampleAttack.py [--train]
```

## Model Stealing Attack
> The Model Stealing Attack uses a dataset of 5000 randomly generated tensors to steal the target model parameters.
> The outputs of both the target and attack(stolen) model are compared for 1000 epocs and the difference is used to adjust the
> attack(stolen) model.
###Usage
> Prepare the target model for the model stealing attack. 
```shell
python NeuralNetworks/MSA_MNIST_generator.py
```

> Run the attack with --train if you want to train the attack(stolen) model before evaluating:
```shell
python Model_Stealing_Attack/ExampleAttack.py [--train]
```

