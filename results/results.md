# Exponentiation

Changing the variable `exponentiation_base` to a real number `a` (rather than `None`) results in the image pixels `x` being transformed as `a^(x - x_max)` where `x_max` is the maximum pixel value in the band1 image.
The idea of this is that it appears to pick out the object more clearly to the human eye, and the image is already on a logarithmic scale.

![Exponentiation of images](./exp.png)

Results are in the directories `exp_on`, `exp_off` and `exp_off_demean`.
Runs were performed with version `27c08c7308b9fa6e32dad7f64541b4ea3b5b97f7`.

It doesn't look like it works very well - any what's going on with the demeaned case? Why does the validation cross-entropy suddenly jump?
