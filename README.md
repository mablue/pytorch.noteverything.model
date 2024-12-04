# pytorch.noteverything.model
A Pytorch model to nothing to detect everything 

## A research about everything detection by training to nothing!

I strat with an example. think to the object detection models
all have a list of objects in the datasets for example cats and dogs and birds segmented in folders that we train the model to detect these catagories. but its not a good idea when we dont know what our model will be detect. I mean if we should detect things that not existed in list what should we do? our trained model can not detect uncategorized objects or unknown objects but a human can do that. we just will show the image of an ufo to a human and he/she/it will detect things that similar to the image for us. for this porpoise we should train our model to nothing there will not be a list of catagory of objects like yolo models. there will be an object and the images that this object existed there. and than the user will just give it the image of object that wants to detect and the model will detect the object from camera or any image.
our model will learn someting more than the shapes and corners of a known object like cat body.
it will train to more deep thing to behave like a human.
