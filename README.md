# LFP Cell HPPC Parameterization

This repository is based on the Battery-HPPC-model-fitting repo by [Manh Kien Tran](https://github.com/kmtran95), Chemical Engineering PhD Candidate at the University of Waterloo <br>
The data used to develop both these repositories was the same and can be found in his [repo](https://github.com/kmtran95/Battery-HPPC-model-fitting/blob/main/HPPC_LFP.zip.). His model is in MatLab, mine is in Python.

The HPPC test is characterized using the [second order RC model](#ecm) of the ecm.<br>
`scipy.curvefit()` parameterizes the ecm function and the resulting modeled and experimental voltage are plotted by the end of the parameterization

## ECM <a id = "ecm"></a>

$$ U_{1,k+1} = exp(-\Delta t/\tau_1)\cdot U_{1,k} + R_1[1 - exp(-\Delta t/\tau_1)]\cdot I_k $$

$$ U_{2,k+1} = exp(-\Delta t/\tau_2)\cdot U_{2,k} + R_2[1 - exp(-\Delta t/\tau_2)]\cdot I_k $$

$$ \tau_1 = R_1C_1 $$

$$ \tau_2 = R_2C_2 $$
 
$$ V_k = OCV - R_0I_k - \sum_{i=1}^{2}U_{i,k} $$
