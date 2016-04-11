##Modulas & Classes
###class trainier
It organizes the _dataBase_, _optimizer_, _validator_ and _nnet_.
You should call __compile(self)__ before you start training or validating.
However, if you __dumped__ a _compiled trainer_ and then __load__ it back, you needn't call _compile_ anymore since the optimizer and the validator the __set_owner__ method are all called before.

###package optimizer
In package optimizer, there are several sub-modulas.
- __core__ defines the interface of an optimizer.
- __optimizer__ defines the specific algorithnms for the optimizer
- __loss__ defines several loss functions used in these optimizers

###class tracker
Wrapper class for the optmizers.
You can pass a tracker instead of an optmizer to a trainer.
