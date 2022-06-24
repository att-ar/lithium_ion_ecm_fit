import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import curve_fit

## First Order RC-Model

def model_1rc(current, delta_t, ocv, u_rc, r_int, r_1, c_1):
    #returns the new voltage and polarization voltage
    tau_1 = r_1 * c_1

    u_rc = np.exp( -delta_t/tau_1 )*u_rc + r_1*( 1-np.exp( -delta_t/tau_1 ) )*current

    return ocv - r_int*current - u_rc, u_rc

def f_volt(data, r_int, r_1, c_1):
    #the function to fit
    current = data["Current"].iloc[1:]
    #constants:
    ocv = data["Voltage"].iloc[0]
    #initial
    u_rc = 0

    model_v = pd.Series(ocv, name = "Model-V")

    for i in current.index:
        delta_t = data.loc[i,"Test Time (sec)"] - data.loc[i-1,"Test Time (sec)"]
        model_v.loc[i], u_rc = model_1rc(current[i],
                                         delta_t,
                                         ocv,
                                         u_rc,
                                         r_int,
                                         r_1,
                                         c_1)

    return model_v

## Second Order RC-Model

def model_2rc(current, delta_t, ocv, u_rc, r_int, r_1, c_1, r_2, c_2):
    #returns the new voltage and polarization voltage
    tau_i = np.array([[r_1 * c_1],[r_2 * c_2]])

    u_rc[0] = np.exp( -delta_t/tau_i[0] )*u_rc[0] + r_1*( 1-np.exp( -delta_t/tau_i[0] ) )*current
    u_rc[1] = np.exp( -delta_t/tau_i[1] )*u_rc[1] + r_2*( 1-np.exp( -delta_t/tau_i[1] ) )*current

    return ocv - r_int * current - u_rc.sum(), u_rc

def f_2volt(data, r_int, r_1, c_1, r_2, c_2):
    #the function to fit
    current = data["Current"].iloc[1:]
    #constants:
    ocv = data["Voltage"].iloc[0]
    #initial
    u_rc = np.zeros((2,1))

    model_v = pd.Series(ocv, name = "Model-V")

    for i in current.index:
        delta_t = data.loc[i,"Test Time (sec)"] - data.loc[i-1,"Test Time (sec)"]

        model_v.loc[i], u_rc = model_2rc(current[i],
                                         delta_t,
                                         ocv,
                                         u_rc,
                                         r_int,
                                         r_1,
                                         c_1,
                                         r_2,
                                         c_2)
    return model_v

## Parameterization

def ecm_param(data, order = 2, to_csv = False):
    '''
    .txt file[, bool] -> pandas.DataFrame, plotly line plot
    Precondition: .txt file of an HPPC test from the Maccor

    This function takes a .txt file from an HPPC test ran by the Maccor
    and parameterizes the equivalent circuit model (ECM) of the cell.

    It first splits all the HPPC test data at 10 SOCs (100 - 10),
    and then runs a helper function that does the fitting all in a loop.
    The resulting parameters from the different SOCs are stored in a pd.DataFrame
    A plot showing the experimental vs. modeled voltage is generated using plotly


    Parameter:
    `order` int
        the RC order of the ECM
        restricted to 1 or 2 as of now

    `to_csv` bool True or False
        defaults to True
        if True, the function outputs a csv file and returns a plotly line plot of the experimental and modeled voltage
        if False, outputs a pandas.DataFrame and returns a plotly line plot of the experimental and modeled voltage
    '''

    if type(order) != int:
        return "Error: input type incorrect."
    if to_csv:
        file = data #the path

    data = pd.read_csv(data,skiprows = [0,1,2], sep="\t")

    df = data[["Test Time (sec)","Step","Current","Voltage"]][
               data["Step"].isin([3,4,5,6])]

    index = df[df.index.to_series().diff(periods=-1) != -1.0].index

    ecm_params = pd.DataFrame(columns = ["r_int","r_1","c_1","r_2","c_2"],
                              dtype = "float64")
    # ecm_params = pd.DataFrame(data = np.array( [np.nan]*95 ).reshape(19,5),
    #                           columns = ["r_int","r_1","c_1","r_2","c_2"],
    #                           dtype = "float64")

    fig = make_subplots(x_title = "Test Time (sec)",
                        y_title = "Voltage (V)",
                        subplot_titles = "Voltage vs Time"
                        )

    for i in range(len(index)):
        if i == 0:
            df_section = df[["Test Time (sec)",
                             "Step","Current",
                             "Voltage"]].loc[ : index[i] ]

        else:
            df_section = df[["Test Time (sec)",
                             "Step","Current",
                             "Voltage"]].loc[ index[i-1] + 1 : index[i] ]

        df_section2 = (pd.DataFrame(df_section[df_section["Step"] == 3].iloc[-1])
                       .transpose()
                       .append(
                               df_section[ df_section["Step"].isin([4,5,6]) ]
                              )
                      )

        df_section2.loc[df_section2["Step"] == 6,
                        "Current"] = df_section2.loc[df_section2["Step"] == 6,
                                                     "Current"] * -1
        print(f"SOCs Completed: {i}", end="\r")

        if order == 2:
            params = curve_fit(f_2volt,
                               df_section2,
                               df_section2["Voltage"],
                               p0 = [0.0247,0.0312,600,0.003,400],
                               bounds = [[0.001,0.00001,10,0.00009,10],
                                         [10,10,40000,1,4000]
                                        ]
                              )


            ecm_params.loc[i] = params[0]
            fit = f_2volt(df_section2,
                         params[0][0],
                         params[0][1],
                         params[0][2],
                         params[0][3],
                         params[0][4]
                         )


        elif order == 1:
            params = curve_fit(f_volt,
                               df_section2,
                               df_section2["Voltage"],
                               p0 = [0.0247,0.0312,620],
                               bounds = [[0.001,0.00001,10],
                                         [10,10,40000]
                                        ]
                              )

            ecm_params.loc[i, ["r_int","r_1","c_1"]] = params[0]
            fit = f_volt(df_section2,
                        params[0][0],
                        params[0][1],
                        params[0][2])

        fig.add_trace(
                      go.Scatter(
                                 x = df_section2["Test Time (sec)"],
                                 y = df_section2["Voltage"],
                                 name = f"Experimental SOC {100-i*10}",
                                 mode = "markers"
                                )
                     )
        fig.add_trace(
                      go.Scatter(
                                 x = df_section2["Test Time (sec)"],
                                 y = fit,
                                 name = f"Model SOC {100-i*10}",
                                 mode = "lines"
                                )
                     )
    fig.show()

    ecm_params = pd.DataFrame( {"SOC":np.arange(100,9,-10)} ).join(ecm_params)
    #pd.DataFrame( {"SOC":np.arange(100,9,-5)} ).join(ecm_params.interpolate())
    if to_csv:
        ecm_params.to_csv(file[:-4] + "_params.csv", index=False)
        print("\n" + file[:-4] + "_params.csv was created. \n")

    return ecm_params


if __name__ == "__main__":
    order = 0
    while order not in ["1","2"]:
        order = input("Order of RC-Model (1 or 2): " )
    order = int(order)

    file = input("Path of file to analyze: ")

    df = ecm_param(file, order = order)
    print(df)
