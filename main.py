import sys
import os
import subprocess


path_proj_dir = os.path.dirname(os.path.realpath(__file__))

path_mia_train = os.path.join(os.path.join(path_proj_dir, "Membership_Inference_Attack"),"ExampleAttack.py")
path_mia_generator = os.path.join(os.path.join(path_proj_dir, "NeuralNetworks"),"MIA_nn_generator.py")


path_mi_mnist_generator = os.path.join(os.path.join(path_proj_dir, "NeuralNetworks"),"MI_MNIST_generator.py")
path_mi_cifar_generator = os.path.join(os.path.join(path_proj_dir, "NeuralNetworks"),"MI_CIFAR_generator.py")
path_mi_attack = os.path.join(os.path.join(path_proj_dir, "Model_Inversion_Attack"),"ExampleAttack.py")

path_msa_mnist_generator = os.path.join(os.path.join(path_proj_dir, "NeuralNetworks"),"MSA_MNIST_generator.py")
path_msa_attack = os.path.join(os.path.join(path_proj_dir, "Model_Stealing_Attack"),"ExampleAttack.py")


while True:

    print("\nPlease press number associated with any option given below :\n")
    print("0. To Quit")
    print("1. Execute Membership inference attack")
    print("2. Execute Model Inversion attack")
    print("3. Execute Model Stealing attack")

    user_input=input("Enter input: ")


    try:
        user_input = int(user_input)
    except:
        print("Error: Please enter valid input")
        continue

    if (user_input > 3 or user_input < 0):
        print("Error: Please enter valid input")

    if user_input == 1:
        print("\n ********** Membership inference attack ********** \n")
        print("\nPlease press number associated with any option given below :\n")
        print("1. Train the target and shadow model and execute Membership inference attack")
        print("2. Execute Membership inference attack without training shadow and train model. Note : select this option if you have train target and "
            "and shadow model for atleast one time")
        user_input = input("Enter input: ")
        try:
            user_input = int(user_input)
        except:
            print("Error: Please enter valid input")
            continue

        if ( user_input > 2 or user_input < 1 ):
            print("Error: Please enter valid input")


        if user_input == 1:
            subprocess.call(["python", path_mia_generator, '--train'])
            subprocess.call(["python",path_mia_train,'--train'])
        elif user_input == 2:
            subprocess.call(["python", path_mia_train])

    elif user_input == 2:
        print("\n ********** Model Inversion attack ********** \n")
        print("\nPlease press number associated with any option given below :\n")
        print("1. Train mnist and CIFAR target models")
        print("2. Execute attack using Cifar dataset without training the target model")
        print("3. Execute attack using Mnist dataset without training the target model")

        user_input = input("Enter input: ")


        try:
            user_input = int(user_input)
        except:
            print("Error: Please enter valid input")
            continue

        if (user_input > 2 or user_input < 1):
            print("Error: Please enter valid input")

        if user_input == 1:
            subprocess.call(["python", path_mi_cifar_generator])

        if user_input == 2:
            subprocess.call(["python", path_mi_attack, 'CIFAR'])

        elif user_input == 3:
            subprocess.call(["python", path_mi_attack, 'MNIST'])


    elif user_input == 3:
        print("\n ********** Model Stealing attack ********** \n")
        print("\nPlease press number associated with any option given below :\n")
        print("1. train the target model")
        print("2. Execute the attack without training the target model")
        print("3. only evaluate the stolen model (target and stolen model should be already trained)")



        user_input = input("Enter input: ")
        try:
            user_input = int(user_input)
        except:
            print("Error: Please enter valid input")
            continue
        if (user_input > 1 or user_input < 1):
            print("Error: Please enter valid input")

        if user_input == 1:
            subprocess.call(["python", path_msa_mnist_generator])
        if user_input == 2:
            subprocess.call(["python", path_msa_attack,'--train'])
        if user_input == 3:
            subprocess.call(["python", path_msa_attack])





    elif user_input == 0:
        exit(0)











