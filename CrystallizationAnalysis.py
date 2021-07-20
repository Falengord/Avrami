import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import t
from tqdm import tqdm
from scipy.integrate import quad

π = np.pi
R = 8.314  # J/molK
kB = 1.380649e-23  # J/K


#######################################################################################
# Params (from MAUD fit on LaB6)
# !!Do Not Change!!
λCuα1 = 1.5405929e-10
λCuα2 = 1.5437975e-10
Cuα1_weight = 1
Cuα2_weight = 0.54098135
gaus0 = -0.11733604
gaus1 = 0.0044349586
W = 0.12129322
V = -0.07644411
U = 0.041425925


#######################################################################################
# Options (some of these ones can conflict with each other, pay attention)
Graphs = False  # print for each analisys step the experimental data as well as the fit function
Tgraph = False  # print a cumulative graph with all the peaks from a dataset
lt_vs_A = True
FullAvrami = True
Mean_Radius = True
exp_r = True
Gn = -1


#######################################################################################
# Set seaborn
sns.set_theme()
sns.set_style("darkgrid")


#######################################################################################
# Functions
def read_XRD(name, sep=" ", dtype="float", integrationTime=1):
    data_temp = pd.read_csv(name, sep=sep)
    numpy_temp = data_temp.to_numpy(dtype=dtype)
    numpy_temp[:, 1] = numpy_temp[:, 1] / integrationTime

    return numpy_temp


def pseudo_voigt(x, A, x0, gamma):
    x = x.astype(np.float)
    x0 = x0.astype(np.float)
    L = A * gamma / ((2 * np.pi) * ((x - x0) ** 2 + (gamma / 2) ** 2))
    G = (
        A
        * 2
        * np.sqrt(np.log(2))
        * np.exp(-np.log(2.) * (x - x0) ** 2. / gamma ** 2.)
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


def linear(x, a, b):
    return a * x + b


def dlinear(x, a, b):
    δa = x
    δb = np.ones(len(x))

    return np.array([δa, δb])


def model_linear():
    return linear, dlinear


def confidence_band(
    x, f_x, χ2, dof, cov_matrix, partial_derivatives, confidence_level=0.95
):
    α = 1 - confidence_level
    prb = 1.0 - α / 2
    tval = t.ppf(prb, dof)

    variance = np.einsum(
        "ik,jk,ij->k", partial_derivatives, partial_derivatives, cov_matrix
    )

    # d_func = np.sqrt(np.sqrt(χ2 * variance / dof))
    d_func = np.sqrt(variance)
    d_prediction = tval * d_func

    upper = f_x + d_prediction
    lower = f_x - d_prediction

    return lower, upper


def plot_with_confidence(
    x,
    y,
    f_x,
    partial_derivatives,
    cov_matrix,
    χ2,
    dof,
    confidence_level=0.95,
    color="r",
    alpha=[0.8, 1, 0.4],
    linewidth=[0.8, 3],
):
    low, upp = confidence_band(
        x, f_x, χ2, dof, cov_matrix, partial_derivatives, confidence_level
    )
    (pd,) = plt.plot(x, y, alpha=alpha[0], linewidth=linewidth[0], color=color)
    (pf,) = plt.plot(x, f_x, "--", alpha=alpha[1], linewidth=linewidth[1], color=color)
    pb = plt.fill_between(x, low, upp, alpha=alpha[2], color=color)

    return [pd, pf, pb]


def caglioti(θ):
    return U * np.tan(θ) ** 2 + V * np.tan(θ) + W


def radius(x, t, b, c, t0):
    g_t = (t - x) ** c
    ex = 1 - Avrami(x, b, c, 0)
    return g_t * ex


def N(x, t, b, c, t0):
    return 1 - Avrami(x, b, c, 0)


def importdata(path, dmin, dmax):
    temp_data = []
    for i in range(dmin, dmax):
        name = path + str(i) + ".xy"
        t_data = read_XRD(name, sep=" ", dtype="float")
        temp_data.append(t_data)
    temp_data = np.array(temp_data)

    return temp_data


def Avrami(x, B, C, x0):
    a1 = np.exp((-B * 1e-10 * (x - x0) ** (3 * C + 1)) / (3 * C + 1))
    return 1 - a1


#######################################################################################
# Import Data
Data = []
Data.append(importdata("5/GIXRD abs_380°C_", 1, 99))
Data.append(importdata("6/GIXRD abs_350°C_", 1, 99))
Data.append(importdata("7/GIXRD abs_340°C_", 1, 80))
Data.append(importdata("9/GIXRD abs_330°C_", 1, 99))
#Data.append(importdata("../720C/2teta abs_S18118 A_720°C_", 1, 175))
Data = np.array(Data, dtype=object)

T_treat = [380, 350, 340, 330]
T_label = [str(T) + " °C" for T in T_treat]
T_treat = np.array(T_treat) + 273


#######################################################################################
# Peak Fitting
Popt = []
Pcov = []
Pval = []
Redc = []

for j, temperature in enumerate(tqdm(Data, desc="Peak fitting", colour="green")):
    popt = []
    pcov = []
    pval = []
    redc = []
    for i, data in enumerate(
        tqdm(
            temperature,
            desc="Analizing T  " + str(j) + "/" + str(len(Data)),
            leave=False,
            colour="blue",
        )
    ):
        sigma = (data[:, 1]) ** 0.5
        if Graphs or Gn == i or Tgraph:
            plt.errorbar(data[:, 0], data[:, 1], linewidth=0.5, yerr=sigma)
            #plt.show()

        # peak curve fitting
        _poptg, _pcovg = curve_fit(
            doublePV,
            data[:, 0],
            data[:, 1], #doublePV(x, A, x0, gamma, a, b)
            p0=[25, 25.2, 0.19, -0.01, 10],  
            bounds=[[0, 25.0, 0.1, -10, -10], [1e5, 25.4, 0.7, 10, 500]],
        )
        chig = chisquare(data[:, 1] / sigma, doublePV(data[:, 0], *_poptg) / (sigma), 5)
        redcg = chig[0] / (len(data[:, 1]) - 5)

        # no peak curve fitting
        _poptl, _pcovl = curve_fit(linear, data[:, 0], data[:, 1], p0=[-0.1, 30])
        chil = chisquare(data[:, 1] / sigma, linear(data[:, 0], *_poptl) / (sigma), 2)
        redcl = chil[0] / (len(data[:, 1]) - 2)

        # Best fit choice
        if redcl < redcg:
            popt.append(np.array([0, 23, 0.8, _poptl[0], _poptl[1]]))
            tcovl = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, _pcovl[0, 0], _pcovl[0, 1]],
                [0, 0, 0, _pcovl[1, 0], _pcovl[1, 1]],
            ]
            pcov.append(np.array(tcovl))
            pval.append(chil[1])
            redc.append(chil[0] / (len(data[:, 1]) - 2))
            if Graphs or Gn == i:
                plt.plot(data[:, 0], linear(data[:, 0], *_poptl))
            if 1 - chil[1] > 0.05:
                print(i, chil)
                print(redcl, redcg)
                plt.plot(data[:, 0], data[:, 1], linewidth=0.5)
                plt.plot(data[:, 0], linear(data[:, 0], *_poptl))
                plt.show()
        else:
            popt.append(np.array(_poptg))
            pcov.append(np.array(_pcovg))
            pval.append(chig[1])
            redc.append(redcg)
            if Graphs or Gn == i:
                plt.plot(
                    data[:, 0],
                    doublePV(data[:, 0], *_poptg),
                    "--",
                    label="fitted model",
                )
                plt.plot(
                    data[:, 0],
                    pseudo_voigt(data[:, 0], *_poptg[:3]),
                    label="Copper $K-\\alpha_1$ peak",
                )
                plt.plot(
                    data[:, 0],
                    pseudo_voigt(
                        data[:, 0],
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
                    label="Copper $K-\\alpha_2$ peak",
                )
            if 1 - chig[1] > 0.05:
                print(i, chig)
                plt.plot(data[:, 0], data[:, 1], linewidth=0.5)
                plt.plot(data[:, 0], doublePV(data[:, 0], *_popt))
                plt.show()

        if Graphs or Gn == i:
            plt.xlabel("$2\\theta$ (deg)")
            plt.ylabel("Intensity (a.u.)")
            plt.legend()
            plt.savefig("Fit.png")
            plt.show()

    if Tgraph:
        plt.xlabel("$2\\theta$ (deg)")
        plt.ylabel("Intensity (a.u.)")
        plt.savefig("Tgraph.png")
        plt.show()

    Popt.append(np.array(popt))
    Pcov.append(np.array(pcov))
    Pval.append(np.array(pval))
    Redc.append(np.array(redc))

Popt = np.array(Popt, dtype=object)
Pcov = np.array(Pcov, dtype=object)
Pval = np.array(Pval, dtype=object)
Redc = np.array(Redc, dtype=object)


#######################################################################################
# Avrami Analysis
P = []
C = []
u = []
maxsT = [580, 340, 155, 580, 300]
mins = [0.01, 0.01, 0.02, 0, 0]
if FullAvrami:
    for j, temp in enumerate(tqdm(Popt, desc="Avrami Analysis", colour="green")):
        minA = temp[:, 0] >= 0 * max(temp[:, 0])
        maxA = temp[:, 0] <= max(temp[:, 0])

        subtemp = temp[minA * maxA]

        t = 18 * np.array(range(len(temp))) + 4
        subt = t[minA * maxA]

        # plt.plot(subt, subtemp[:,0])
        p, c = curve_fit(Avrami, subt, subtemp[:, 0] / maxsT[j], p0=[25.2, 25.0, 25.4])
        P.append(p)
        C.append(c)
        # print(p)
        plt.plot(t, temp[:, 0] / maxsT[j], label=T_label[j])
        plt.plot(t, Avrami(t, *p), "--")
    plt.legend()
    plt.xlabel("time (minutes)")
    plt.ylabel("Crystal fraction")
    plt.savefig("Avrami.png")
    plt.tight_layout()
    plt.show()

    P = np.array(P)
    C = np.array(C)

    print(P)
    print(np.sqrt(C))


#######################################################################################
# log t vs Avrami transformation
P1 = []
P2 = []
m1 = [70, 25, 12, 6, 1]
m2 = [100, 50, 20, 15, 4]
m3 = [110, 60, 30, 25, 6]
m4 = [90, 120, 50, 70, 30]

if lt_vs_A:
    for j, temp in enumerate(
        tqdm(Popt, desc="log t vs Avrami transformation", colour="green")
    ):        
        t = 18 * np.array(range(len(temp))) + 4
        t = t * 60
        A_t = -np.log(1 - temp[:, 0] / (max(temp[:, 0] + 0.1)))
        A = np.log(A_t)
        lt = np.log(t)
        plt.plot(lt, A, label=T_label[j])
        #p1, c1 = curve_fit(linear, lt[m1[j] : m2[j]], A[m1[j] : m2[j]])
        #plt.plot(lt[m1[j] : m2[j]], linear(lt[m1[j] : m2[j]], *p1), "k--")
        #P1.append(p1[1])  # /p1[0])
        #p2, c2 = curve_fit(linear, lt[m3[j] : m4[j]], A[m3[j] : m4[j]])
        #plt.plot(lt[m3[j] : m4[j]], linear(lt[m3[j] : m4[j]], *p2), "k--")
        #P2.append(p2[1])  # /p2[0])
    plt.xlabel("ln(t)")
    plt.ylabel("ln(-ln(1-x))")
    # plt.savefig("ln_t_vs_ln_A.png")
    plt.legend()
    plt.tight_layout()
    plt.show()
    P1 = np.array(P1)
    P2 = np.array(P2)


#######################################################################################
# Arrhenius from k
if lt_vs_A:
    plt.plot(1 / T_treat, P1, ".")
    p1, c1 = curve_fit(linear, 1 / T_treat, P1)
    plt.plot(1 / T_treat, linear(1 / T_treat, *p1), "--", label="first phase")
    plt.plot(1 / T_treat, P2, ".")
    p2, c2 = curve_fit(linear, 1 / T_treat, P2)
    plt.plot(1 / T_treat, linear(1 / T_treat, *p2), "--", label="second phase")
    print(-p1[0] / R, -p2[0] / R)
    plt.xlabel("ln($\dot{G}$)")
    plt.ylabel("ln(-ln(1-x))")
    # plt.savefig("Energy activation.png")
    plt.legend()
    plt.tight_layout()
    plt.show()


#######################################################################################
# mean radius by experimental data
Size = []
v = []
if exp_r:
    for j, temp in enumerate(
        tqdm(Popt, desc="mean radius experimental", colour="green")
    ):
        broad = np.sqrt(Popt[j][:, 2] ** 2 - caglioti(Popt[j][:, 1] * π / 180) ** 2)
        broad = broad * π / 180
        size = 0.9 * λCuα1 / (broad * np.sin(Popt[j][:, 1] * π / 360))
        Size.append(size)
        plt.plot(Popt[j][:, 2])
    #        plt.plot(caglioti(Popt[j][:,1]*π/180)**2)
    plt.show()

    for j, s in enumerate(Size):
        t = 18 * np.array(range(len(s))) + 4
        plt.plot(t, 1e9 * s, ".", label=T_label[j])
    plt.legend()
    plt.xlabel("time (minutes)")
    plt.ylabel("Mean radius (nm)")
    plt.ylim(0, 400)
    plt.tight_layout()
    plt.savefig("broad.png")
    plt.show()

    b1 = [21, 10, 6, 4, 0]
    b2 = [45, 25, 18, 10, 3]
    for j, s in enumerate(Size):
        t = 18 * np.array(range(len(s))) + 4
        plt.plot(t, 1e9 * s, ".", label=T_label[j])
        p, c = curve_fit(linear, t[b1[j] : b2[j]], s[b1[j] : b2[j]])
        v.append(np.array([p[0] * 2, np.sqrt(c[0, 0]) / 2]))
        plt.plot(
            t[b1[j] : b2[j]],
            1e9 * linear(t[b1[j] : b2[j]], *p),
            "k--",
            alpha=0.6,
            color="C" + str(j),
        )
        plt.legend()
    plt.xlabel("time (minutes)")
    plt.ylabel("Mean radius (nm)")
    plt.ylim(0, 400)
    plt.xlim(0, 1500)
    plt.tight_layout()
    plt.savefig("mean_radius.png")
    plt.show()

    v = np.array(v)
    plt.errorbar(
        1 / (kB * T_treat),
        np.log(v[:, 0]),
        np.log(v[:, 1] + v[:, 0]) - np.log(v[:, 0]),
        fmt=".",
    )
    p, c = curve_fit(linear, 1 / (kB * T_treat), np.log(v[:, 0]))
    plt.plot(1 / (kB * T_treat), linear(1 / (kB * T_treat), *p), "--")
    plt.xlabel("$1/k_BT$  $(J^{-1})$")
    plt.ylabel("ln($v$)")
    plt.tight_layout()
    plt.savefig("activation.png")
    plt.show()
    print(p[0], np.sqrt(c[0, 0]))
    print(v[:, 0], v[:, 1])
    print(np.log(v[:, 0]))
    print(p[0] / (kB * (np.log(22e-9) - p[1])))

    n = []
    b1 = [32, 16, 11, 6, 0]
    b2 = [45, 25, 18, 10, 3]
    for j, temp in enumerate(Popt):
        t4 = t[b1[j] : b2[j]] ** 4
        plt.plot(t4, temp[b1[j] : b2[j], 0] / v[j, 0] ** 3, label=T_label[j])
        t4f = t4 * 1e-11
        nf = temp[b1[j] : b2[j], 0] * 1e-27 / v[j, 0] ** 3
        p, c = curve_fit(linear, t4f, nf, p0=[1, 0])
        plt.plot(t4, linear(t4f, *p) * 1e27, "--", alpha=0.6, color="C" + str(j))
        n.append(np.array([p[0] * 3 * 1e16 / π, 1e16 * np.sqrt(c[0, 0]) * 3 / π]))
        plt.ylabel("$x/v^3$  $(min^3/nm^3)$")
        plt.xlabel("$t^4$ $(min^4)$")
        plt.tight_layout()
        plt.savefig("nucleation.png")
        plt.show()

    n = np.array(n)

    plt.errorbar(T_treat[:-1], n[:-1, 0], n[:-1, 1], fmt=".")
    plt.show()

    print(n)
    plt.errorbar(
        T_treat, np.log(n[:, 0]), np.log(n[:, 0] + n[:, 1]) - np.log(n[:, 1]), fmt="."
    )
    p, c = curve_fit(linear, T_treat, np.log(n[:, 0]))
    plt.plot(T_treat, linear(T_treat, *p), "--")
    plt.xlabel("Temperature  $(K)$")
    plt.ylabel("ln($\\dot{n}$)")
    plt.tight_layout()
    plt.savefig("log_nucleation.png")
    print(np.exp(linear(973, *p)))
    print(np.exp(linear(960, *p)))
    plt.show()

#######################################################################################
# mean radius by numerical integration
Mean_R = []
Mean_N = []
if FullAvrami:
    if Mean_Radius:
        for j, temp in enumerate(
            tqdm(Popt, desc="mean radius Analysis", colour="green")
        ):
            t = 18 * np.array(range(len(temp))) + 4
            mean_r = []
            mean_n = []
            for time in t:
                if time >= P[j][2]:
                    dr = lambda x: radius(x, time - P[j][2], *P[j])
                    dn = lambda x: N(x, time - P[j][2], *P[j])
                    a = quad(dr, 0, time - P[j][2])[0]
                    mn = quad(dn, 0, time - P[j][2])[0]
                    mean_r.append(a / mn)
                    mean_n.append(mn)
                else:
                    mean_r.append(0)
                    mean_n.append(0)

            mean_r = np.array(mean_r)
            mean_n = np.array(mean_n)
            Mean_R.append(mean_r)
            Mean_N.append(mean_n)
            plt.plot(t[mean_r != 0], size[mean_r != 0] / mean_r[mean_r != 0])
        plt.show()


for j, s in enumerate(Size):
    noz = Mean_N[j] != 0
    plt.plot(Popt[j][noz][:, 0] / (s[noz] ** 3 * maxsT[j] * Mean_N[j][noz]))
plt.show()


for j, temp in enumerate(tqdm(Popt, desc="mean density Analysis", colour="green")):
    t = 18 * np.array(range(len(temp))) + 4
    dt = 18
    # for time in t:
    # plt.plot()

# plt.errorbar(range(len(P[:,1])), P[:,1], np.sqrt(C[:,1,1]))
# plt.errorbar(range(len(P[:,1])), P[:,2], np.sqrt(C[:,2,2]))
# plt.show()

# v = 9.143e-11
# v3 = v**3
# plt.errorbar(range(len(P[:,1])), np.sqrt(np.sqrt(P[:,1]/v3)), np.sqrt(np.sqrt(np.sqrt(C[:,1,1])/v3)))
# plt.show()
#
# plt.errorbar(range(len(P[:,1])), np.sqrt(np.sqrt(P[:,1]/v3))*v, np.sqrt(np.sqrt(np.sqrt(C[:,1,1])/v3))*v)
# plt.show()

# Size = []
# for j in range(len(Popt)):
#    broad = np.sqrt(Popt[j][:,2]**2 - caglioti(Popt[j][:,1]*π/180)**2)
#    broad = broad*π/180
#    size = 0.9*λCuα1/(broad*np.sin(Popt[j][:,1]*π/360))
#    #plt.plot(j,size[-1])
#    #plt.plot(Size[0][-1]size)
#    Size.append(size)
#
# plt.plot([Size[0][-1],Size[1][-1],Size[2][-1],Size[3][-1],Size[4][-1]])
# plt.show()
