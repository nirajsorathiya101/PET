import torch
import torch.nn as nn
from tqdm import tqdm


class InversionAttack:
    def __init__(self, classifier, _image, _labels, _target):
        self.classifier = classifier  # the neural network which will be attacked
        self._labels = _labels  # labels are not strictly necessary
        # The target class is the index of the target in the output of the NN
        self._target = torch.tensor([_target.item() if torch.is_tensor(_target) else _target])
        # The image that will be returned is initialized with zeros
        self._image = torch.zeros(1, 1, _image.size()[2], _image.size()[3]).requires_grad_()

    @staticmethod
    def _auxterm(self, x):
        """uses available auxiliary information to inform the cost function"""
        return 0

    @staticmethod
    def _c(self, _input, gradient=False):
        """cost function c, it represents the actual classifier we are trying to inverse"""
        outputs = self.classifier(_input)
        m = nn.Softmax(dim=1)
        normalized_outputs = m(outputs) # The sum of all outputs is normalized to 1
        score = normalized_outputs[0][self._target] # The performance of the model on the class target w.r.t the input
        if gradient:
            target_vector = [0] * len(self._labels) # creating a one hot vector
            target_vector[self._target] = 1
            target_vector = torch.tensor([target_vector], dtype=float)
            return 1 - torch.autograd.grad(normalized_outputs, _input, grad_outputs=target_vector)[0]
        return 1 - score + self._auxterm(self, _input)

    @staticmethod
    def _process(self, x):
        """_the function process can perform various image manipulations
            such as denoising and sharpening,as necessary for a given attack"""
        return x

    def miface(self, alpha=5000, beta=100, gamma=0.99, lambda_=0.1):
        # kept the variable name for clarity, but this is not very Pytonic
        """function MI-Face(label,alpha,beta,gamma, lambda)
                classifier - the model on which we want to make the inversion
                label - the valid name of a category outputed by the algorithm
                alpha -  the maximum number of gradient descent made by the function MI-Face
                beta - the maximum number of iterations made without improvement (beta < alpha)
                gamma - the cost for which the gradient descent should terminate
                lambda - the size of the gradient descend steps """

        last_performances = [None]*beta
        for j in tqdm(range(alpha)):
            self._image = self._image.detach().requires_grad_() # Reset the gradient of the image
            c = self._c(self, self._image, True) # Calculate the gradient
            self._image = self._process(self, self._image - lambda_ * c)
            loss = self._c(self, self._image)
            if j >= beta:
            #if len([value for value in last_performances if value is not None]) > 0:
                if loss.item() >= max([value for value in last_performances if value is not None]):
                    break
            if loss.item() >= gamma:
                break
            last_performances[j % beta] = loss.item()
        return self._image.detach()