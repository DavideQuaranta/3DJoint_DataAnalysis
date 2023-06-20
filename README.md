# 3DJoint_DataAnalysis

I changed the purpose of this repository since we decided to keep everything on Google Drive.
Maybe we'll use this space to make some tests on fitting algorithm and in general for data analysis until 
we have refined results and then upload on google drive only the the final results.
Anyway i'll put here the raw data we take in the lab as a backup.

## Joint3DLib
This library contains some useful functions that are used often so maybe it's better to speed up the work. The included functions are:
- `error` : this function computes the voltage error as indicated on the manufacturer datasheet
- `chidof`: computes the normalized chi squared between observed and expected data
- `singleExp` : definition of an exponential function $a e^{(-x/\tau)}$
- `ExpConst` : definition of an exponential function plus a constant $a e^{(-x/\tau)} + c$
- `fitSheet` : takes an excel sheet and fits the chosen function to data.Returns the Q factor, the temperature and their errors and the normalized chisq. The needed argument are:
    - `directory` of the excel file
    - `sheet` : the sheet name
    - `initial` : initial conditions for the fit parameters
    - `f` : the function to fit. Default is singleExp.
    - `cut` : select only data with from voltage < cut.
    - `verbose`: boolean, if True prints the best fit parameters
  
