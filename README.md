# ALGA: Arbitrary Length Generalization for Addition in R and Python


Abstract: This paper introduces a novel training methodology that enables a  Transformer model to generalize the addition of two-digit numbers to numbers with unseen lengths of digits. The proposed approach employs an autoregressive generation technique, processing from right to left, which mimics a common manual method for adding large numbers. To the best of my knowledge, this methodology has not been previously explored in the literature. All results are reproducible, and the corresponding R and Python codes are available at: [github](https://github.com/AGPatriota/ALGA-R/) and [kaggle notebook](https://www.kaggle.com/code/agpatriota/alga-py/notebook).

This is an implementation of ALGA for R and Python.

[paper uploaded in ArXiv](https://drive.google.com/file/d/1vztXI8m6_qhIi69d4RiggaUGLs_HWGxD/view?usp=sharing)

## In Python
See the kaggle [notebook](https://www.kaggle.com/code/agpatriota/alga-py).

## Dependencies in R:

- [torch](https://cran.r-project.org/web/packages/torch/index.html) 
- [stringr](https://cran.r-project.org/web/packages/stringr/index.html)
- [stringdist](https://cran.r-project.org/web/packages/stringdist/index.html)
- [gmp](https://cran.r-project.org/web/packages/gmp/)

## quick start

In order to train from scratch, you want to make sure to set `train = TRUE` in the `config.R` file. You can run on the terminal or directly in you R session.

Terminal: run the following bash command inside the main folder:

```
for i in $(seq 1 20); do echo "i = $i; source('Arithmetic.R')" | R --no-save >> Resultados; done
```

R session: open the R in the main folder, uncomment lines 54 and 107 in file `Arithmetic.R`and run:

```
source('Arithmetic.R')"
```

In order to run the inferences for numbers with 1000 digits, you want to make sure to set `train = FALSE` in the `config.R` file. Open an R session in the main folder and run:

```
source('Arithmetic.R')"
```

In order to test for random numbers with 100 digits run

```
source('Testing-Digits.R')
 Digit #100 
[1] "The output is TRUE"
[1] "x=8967562796917765651481949069183172510990887498067132029996942446258125998499183326035828805968783222"
[1] "y=5905754374160803754361006754299754810645452277514852730076956303947441956535852107660028556910352668"
[1] "Real x+y=14873317171078569405842955823482927321636339775581984760073898750205567955035035433695857362879135890"
[1] "Pred x+y=14873317171078569405842955823482927321636339775581984760073898750205567955035035433695857362879135890"
```

You can change the seed and the number of digits or you can consider specific numbers x and y to test the function.


### License

MIT
