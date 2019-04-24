YaleFace Code consists of three scripts, which should reside within the same directory.

prepare_dataset.py creates a utility for reading in the images and labels

facelayers.py is where the classes are defined for the four networks that were created

YaleFace Model.py is the main code, which calls the classes and functions from the prior two scripts
 - FaceCascade uses the file haarcascade_frontalface_default.xml. Once pip FaceCascade is pip installed, this code will need to point to the directory in which it is located.
