import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    from scipy.optimize import fsolve, curve_fit
    from scipy.stats import binomtest
    from scipy.linalg import eigvals
    return binomtest, curve_fit, math, np, pd, plt


@app.cell
def _(pd, plt, run):
    @run
    def _():
        df = pd.read_csv('data/HW4_all_weeks.csv')
        # df.plot(x='Week', y='New Cases')
        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(15,3))
        axs[0].plot(df['Week'], df['New Cases'])
        # axs[1].plot(df['Week'][9:9+30], df['New Cases'][9:9+30] * 10)
        axs[1].plot(df['Week'], df['New Cases'] * 10)
        axs[2].plot(df['Week'], (df['New Cases'] * 10).cumsum())
        for ax in axs:
            ax.set_xlabel('Week')
        axs[0].set_title("Identified New Cases")
        axs[1].set_title("Actual New Cases")
        axs[2].set_title("Cumulative Infections")
        # ax.legend()
        # ax.set_ylim(bottom=0)
        # ax.set_xlim(left=0)
        # ax.set_title("Infected vs Time")
        plt.savefig('out.nosync/q1a1.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(curve_fit, np, pd, plt, run):
    @run
    def _():
        df = pd.read_csv('data/HW4_all_weeks.csv')

        n_weeks = 8
        T = df['Week'][9:9+n_weeks]
        dI = (df['New Cases'] * 10)[9:9+n_weeks]
        I = dI.cumsum()
        log_I = np.log(I)
        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(15,3))

        axs[0].plot(T, dI)
        axs[1].plot(T, I)
        axs[2].plot(T, log_I, label=r"$\log(I)$")
        # axs[2].set_yscale('log')

        popt, pcov = curve_fit(lambda x,a,b: a*x+b, T, log_I)
        print(popt)
        print(pcov)
        m = popt[0]
        e = np.sqrt(np.diagonal(pcov))[0]
        print(e)
        mc = np.array([m-e, m+e])
        rc = 1+mc/(1/2 + 1/100)
        print(f"{mc = }")
        print(f"{rc = }")

        axs[2].plot(T, T*popt[0]+popt[1], label="Fit curve: 0.513x+2.05")
    

        m = popt[0]
        mu = 1/100
        gamma = 1/2
        R0 = 1 + m / (gamma + mu)
        print(f"\n{R0 = }")

        for ax in axs:
            ax.set_xlabel('Week')
        axs[0].set_title(r"$\Delta I$ per week")
        axs[0].set_ylabel(r"$\Delta I$")
        axs[1].set_title("$I$")
        axs[1].set_ylabel("$I$")
        axs[2].set_title(r"$\log(I)$")
        axs[2].legend()
    
        plt.savefig('out.nosync/q1a2.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(binomtest, np, run):
    @run
    def _():
        interval = np.array(binomtest(7, 1000).proportion_ci())
        I = 1/(1-interval*(0.5/0.01+1))
        print("1b:", I, I[1]-I[0])

        interval = np.array(binomtest(517, 1000).proportion_ci())
        I = 1/(1-interval)
        print("1c:", I, I[1]-I[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#1e""")
    return


@app.cell
def _(math, pd, plt, run):
    @run
    def _():
        df = pd.read_csv('data/HW4_all_weeks.csv')
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,3))

        T = df['Week']
        dI = df['New Cases'] * 10
        I = dI.cumsum()
        # ax.plot(T, dI)

        gamma = 1/2
        w = lambda t: math.exp(-gamma*t) if t > 0 else 0

        # T = np.arange(-2, 5, 0.01)
        # ax.plot(T, [w(t) for t in T])

        # for j in range(len(dI)):
        #     # denom = sum(w(i-k) for k in range())
        #     for i, di enumerate(dI[j+1:]):
        #         pij = di*w(i-j)/)
        
        #     # pij_denom = 0
        #     # for j in range(t):
        #         # pij_denom += dI[j] * w(t-j)
        #     # pij = w()

        plt.show()
    return


@app.cell
def _(binomtest, np, run):
    @run
    def _():
        interval = np.array(binomtest(39, 100).proportion_ci())
        I = (interval-(1-0.98))/(0.90+0.98-1)
        print("Maria:", I, I[1]-I[0])
    
        interval = np.array(binomtest(18, 50).proportion_ci())
        I = (interval-(1-0.98))/(0.90+0.98-1)
        print("Burt:", I, I[1]-I[0])

        interval = np.array(binomtest(18+39, 100+50).proportion_ci())
        I = (interval-(1-0.98))/(0.90+0.98-1)
        print("Both:", I, I[1]-I[0])
    return


@app.cell
def _(run):
    @run
    def _():
        p = 0.4090909091
        se = 0.90
        sp = 0.98
        pr = (se*p)/(se*p + (1-sp)*(1-p))
        print(pr)
    return


@app.cell
def _(np, pd, plt, run):
    @run
    def _():
        fig, ax = plt.subplots(figsize=(7,7))
        neg = pd.read_csv('data/HW4_Q3_neg-1.csv')
        pos = pd.read_csv('data/HW4_Q3_pos-1.csv')
        data = pd.read_csv('data/HW4_Q3_data-1.csv')

        # colors = 
        plts = [
            ["Negative Controls", "Positive Controls", "Field Data"],
            [neg, pos, data],
            ["r", "k", "b"],
        ]
        w = 10
        p=3
        for i, (t,d,c) in enumerate(zip(*plts)):
            ax.scatter(i*(w+p) + np.random.random(d.size) * w - w / 2, d, color=c, alpha=0.3, label=t)
        ax.bar([i*(w+p) for i in [0,1,2]],
       height=[np.mean(yi) for yi in [neg, pos, data]],
       width=w,    # bar width
       tick_label=plts[0],
       color=(0,1,0,0),  # face color transparent
       )

        ax.set_ylabel('Assay Value')
        ax.set_title("Prevalence Study Data")
        ax.legend()
        plt.savefig('out.nosync/q3a.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(np, pd, plt, run):
    from scipy.optimize import minimize, basinhopping
    @run
    def _():
        neg = pd.read_csv('data/HW4_Q3_neg-1.csv')
        pos = pd.read_csv('data/HW4_Q3_pos-1.csv')
        data = pd.read_csv('data/HW4_Q3_data-1.csv')

        se = lambda c: int(sum(np.array(pos) > c)[0]) / pos.size
        sp = lambda c: int(sum(np.array(neg) < c)[0]) / neg.size
        phi = lambda c: int(sum(np.array(data) > c)[0]) / data.size
        # theta = lambda c: (phi(c) - (1 - sp(c))) / (se(c) + sp(c) - 1)
        def theta(c):
            try:
                return (phi(c) - (1 - sp(c))) / (se(c) + sp(c) - 1)
            except:
                return np.nan
        J = lambda c: se(c) + sp(c) - 1

        # r = minimize(lambda c: -J(c), 0, method="Nelder-Mead", bounds=[(0,50)])
        # Y = basinhopping(lambda c: -J(c), [10,10])

        fig, ax = plt.subplots()

        C = np.arange(0, 50, 0.01)
        Js = np.vectorize(J)(C)
        y = C[np.argmax(Js)]
        print(y)
        ax.plot(C, Js)
        plt.axvline(y, color="r")
        plt.text(y+0.5,0, 'Youden Choice = 14.76', rotation=90)
        ax.set_ylabel('$J(c)$')
        ax.set_xlabel('$c$')
        ax.set_title("Maximize $J(c)$ to find Youden choice")

        plt.savefig('out.nosync/q3b.svg', bbox_inches='tight')
        plt.show()

        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(15,5))

        ax = axs[0]
        ax.plot(1-np.vectorize(sp)(C), np.vectorize(se)(C))
        ax.scatter([1-np.vectorize(sp)(y)], np.vectorize(se)(y), color="r", label="Youden choice")

        ax.set_title("Receiver Operator Curve")
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.legend()

        ax = axs[1]

        ax.set_title("Corrected Prevalence")
        ax.plot(C, np.vectorize(theta)(C))
        ax.scatter(y, theta(y), color="r", label="Youden choice")
        ax.set_ylabel(r'$\hat\theta(c)$')
        ax.set_xlabel('$c$')
        ax.legend()
    
        plt.savefig('out.nosync/q3c.svg', bbox_inches='tight')
        plt.show()
    

    return


@app.cell
def _():
    import marimo as mo
    import joblib
    cache = joblib.Memory("out.nosync/joblib_cache", verbose=0).cache
    def run(f): return f()
    def unique_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    return mo, run


if __name__ == "__main__":
    app.run()
