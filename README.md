# EconOfPrivacy
Code for seminar paper in Economics of Privacy.
The code is set in python, but relies for most parts on Matlab code from the seminar, Dynamic Programming, from University of Copehagen.

# Files in the folder
*BufferStockEGM.py* contains code for the solution of a consumption/savings model. 

*BFClass.py* contains the class for parameter values for the model. In order to change the parameters, start here.

*GE_model.py* contains the code for solving the general equilibrium model.

*FigModule.py* contains a local module holding functions for solving and plotting both the consumption/savings model and the General Equilibrium model.

*funs.py* contains functions used for solving the models.

# How to use the code
Solve the consumptions savings model by running *BufferStockEGM.py*, and solve the general equilibrium model bu running *GE_model.py*. The files *FigModule.py* and *funs.py* must be in working directory as they are called from the other files.