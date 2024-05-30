# ALGA: Arbitrary Length Generalization for Addition in R


Abstract: Under autoregressive generation, if a (small) transformer can perform addition of two numbers with two digits each, then it should also be able to perform addition for numbers of unseen digit lengths, whenever the training procedure is properly defined. 

This is an R implementation of ALGA for R.


## Dependencies:

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



### License

MIT
