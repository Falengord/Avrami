import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import t

π = np.pi


#######################################################################################
# Params (from MAUD fit on LaB6)
λCuα1 = 1.5405929e-10
λCuα2 = 1.5437975e-10
Cuα1_weight = 0.69138354
Cuα2_weight = 1 - Cuα1_weight
gaus0 = 0.05775627
gaus1 = 0.0038439028


#######################################################################################
# Options
inputfile = "350/6/GIXRD abs_350°C_"
Graphs = False
Save = False
Converted = False
Sg = True


#######################################################################################
# Functions
def pol(p, x):
    a, b, c, d = p
    return a * x ** 3 + b * x ** 2 + c * x + d


def polm1(p, x):
    a, b, c, d = p
    return 3 * a * x ** 2 + 2 * b * x + c


def gaus(x, a, x0, sigma):

    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def linear(x, a, b):

    return a * x + b


def peak(x, A, x0, sigma, a, b):

    return gaus(x, A, x0, sigma) + linear(x, a, b)


def pseudo_voigt(x, A, x0, gamma):
    L = A * gamma / ((2 * np.pi) * ((x - x0) ** 2 + (gamma / 2) ** 2))
    G = (
        A
        * 2
        * np.sqrt(np.log(2))
        * np.exp(-np.log(2) * (x - x0) ** 2 / gamma ** 2)
        / (np.abs(gamma) * np.sqrt(np.pi))
    )
    eta = gaus0 + gaus1 * x
    return (1 - eta) * G + L * eta  # non physical quantity


def doublePV(x, A, x0, gamma, a, b):
    A2 = A * Cuα2_weight / Cuα1_weight
    x02 = 2 * np.arcsin(np.sin(x0 * np.pi / 360) * λCuα2 / λCuα1) * 180 / np.pi
    gamma2 = (
        gamma
        * λCuα2
        * np.cos(x0 * np.pi / 360)
        / (λCuα1 * np.cos(np.arcsin(np.sin(x0 * np.pi / 360) * λCuα2 / λCuα1)))
    )
    return (
        pseudo_voigt(x, A, x0, gamma)
        + pseudo_voigt(x, A2, x02, gamma)
        + linear(x, a, b)
    )


#######################################################################################
# Set seaborn
sns.set_theme()
sns.set_style("darkgrid")


#######################################################################################
# Import Data
Data = []
for i in range(1, 180):
    f = open(inputfile + str(i) + ".xy")
    data = []
    x = []
    y = []
    for line in f:
        xi, yi = line.split()
        x.append(float(xi))
        y.append(float(yi))
    data.append(np.array(x))
    data.append(np.array(y))
    data = np.array(data)
    Data.append(data)
    f.close()
Data = np.array(Data)


#######################################################################################
# Peak Fitting
popt = []
pcov = []
pval = []
redc = []
for i, data in enumerate(Data):
    sigma = np.sqrt(data[1])
    if Graphs:
        plt.errorbar(data[0], data[1], linewidth=0.5, yerr=sigma)

    # peak curve fitting
    _poptg, _pcovg = curve_fit(
        doublePV,
        data[0],
        data[1],
        p0=[50, 25.15, 0.19, -0.01, 10],
        bounds=[[0, 25.1, 0.1, -10, -10], [1e5, 25.2, 0.25, 10, 500]],
    )
    chig = chisquare(data[1] / sigma, doublePV(data[0], *_poptg) / (sigma), 5)
    redcg = chig[0] / (len(data[1]) - 5)

    # no peak curve fitting
    _poptl, _pcovl = curve_fit(linear, data[0], data[1], p0=[-0.1, 30])
    chil = chisquare(data[1] / sigma, linear(data[0], *_poptl) / (sigma), 2)
    redcl = chil[0] / (len(data[1]) - 2)

    # Best fit choice
    if redcl < redcg:
        popt.append(np.array([0, 25, 1, _poptl[0], _poptl[1]]))
        tcovl = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, _pcovl[0, 0], _pcovl[0, 1]],
            [0, 0, 0, _pcovl[1, 0], _pcovl[1, 1]],
        ]
        pcov.append(np.array(tcovl))
        pval.append(chil[1])
        redc.append(chil[0] / (len(data[1]) - 2))
        if Graphs:
            plt.plot(data[0], linear(data[0], *_poptl))
        if 1 - chil[1] > 0.05:
            print(i, chil)
            print(redcl, redcg)
            plt.plot(data[0], data[1], linewidth=0.5)
            plt.plot(data[0], linear(data[0], *_poptl))
            plt.show()
    else:
        popt.append(np.array(_poptg))
        pcov.append(np.array(_pcovg))
        pval.append(chig[1])
        redc.append(redcg)
        if Graphs:
            plt.plot(data[0], doublePV(data[0], *_poptg))
            plt.plot(data[0], pseudo_voigt(data[0], *_poptg[:3]))
            plt.plot(
                data[0],
                pseudo_voigt(
                    data[0],
                    _poptg[0] * Cuα2_weight / Cuα1_weight,
                    2
                    * np.arcsin(np.sin(_poptg[1] * np.pi / 360) * λCuα2 / λCuα1)
                    * 180
                    / np.pi,
                    _poptg[2]
                    * λCuα2
                    * np.cos(_poptg[1] * np.pi / 360)
                    / (
                        λCuα1
                        * np.cos(
                            np.arcsin(
                                np.sin(_poptg[1] * np.pi / 360) * λCuα2 / λCuα1
                            )
                        )
                    ),
                ),
            )
        if 1 - chig[1] > 0.05:
            print(i, chig)
            plt.plot(data[0], data[1], linewidth=0.5)
            plt.plot(data[0], doublePV(data[0], *_popt))
            plt.show()
    if Graphs:
        plt.show()
popt = np.array(popt)
pcov = np.array(pcov)
pval = np.array(pval)


#######################################################################################
# Data cleaning
popt[popt[:, 0] == 0, 0] = min(popt[popt[:, 0] != 0, 0])


#######################################################################################
#  Crystal fraction calculation
iA0 = np.argmax(popt[:, 0])
x = popt[1:, 0] / popt[iA0, 0]
sx = np.sqrt(
    pcov[1:, 0, 0] / popt[iA0, 0] ** 2
    + pcov[iA0, 0, 0] * popt[1:, 0] ** 2 / popt[iA0, 0] ** 4
)
t = np.arange(len((x)) + 1) * 996.5 + 197.5
plt.errorbar(t[1:], x, sx, fmt=".")
plt.xlabel("time (s)")
plt.ylabel("x - crystallization fraction")
if Sg:
    plt.savefig("x.png")
plt.show()


#######################################################################################
# n estimate from Crystal fraction
lopt, lcov = curve_fit(linear, np.log(t[8:30]), np.log(x[7:29]), p0=[1, 0])
plt.errorbar(np.log(t[1:]), np.log(x), sx / x, fmt=".", label="data")
plt.plot(np.log(t[8:30]), linear(np.log(t[8:30]), *lopt), "--", label="fitted growth")
plt.xlabel("ln(t) - Time")
plt.ylabel("ln(x) - Crystal fraction")
plt.legend()
plt.title("n = %.2f $\pm$ %1.0e" % (lopt[0], np.sqrt(lcov[0, 0])))
if Sg:
    plt.savefig("growthspeed.png")
plt.show()


#######################################################################################
# Avrami analisys
yAvrami = np.log(-np.log(1 - x[x != 1]))
sAvrami = sx[x != 1] / (-np.log(1 - x[x != 1]) * (1 - x[x != 1]))
ps = []
Avopt, Avcov = curve_fit(linear, np.log(t[1:][x != 1][8:40]), yAvrami[7:39], p0=[1, 0])
plt.errorbar(np.log(t[1:][x != 1]), yAvrami, yerr=sAvrami, fmt=".", label="data")
plt.plot(
    np.log(t[1:][x != 1][8:40]),
    linear(np.log(t[1:][x != 1][8:40]), *Avopt),
    "--",
    label="fitted growth",
)
plt.xlabel("ln(t) - Time")
plt.ylabel("ln[-ln(1-x)] - Crystal fraction")
plt.legend()
plt.title("n = %.2f $\pm$ %1.0e" % (Avopt[0], np.sqrt(Avcov[0, 0])))
plt.ylim(-6.2, 2)
if Sg:
    plt.savefig("Avrami.png")
plt.show()
