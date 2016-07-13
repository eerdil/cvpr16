% -g option of mex is necessary when you debug the code in C files
mex mcmcShapeSampling.c
mex -g mcmcShapeSampling.c

mex evaluateEnergyWithShapePrior.c
mex -g evaluateEnergyWithShapePrior.c

mex EvolveWithDataTerm.c
mex -g EvolveWithDataTerm.c