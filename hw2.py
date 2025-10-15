import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from scipy.optimize import fsolve
    from scipy.linalg import eigvals
    return eigvals, math, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Forward Euler solver""")
    return


@app.cell
def _(np):
    def SIS(s0, i0, beta, gamma, duration, dt):
        T = np.arange(0, duration+dt, dt)
        s = np.zeros(len(T))
        i = np.zeros(len(T))
        s[0] = s0
        i[0] = i0
        ds = lambda s, i: -beta*s*i + gamma*i
        di = lambda s, i: beta*s*i - gamma*i
        for idx in range(1, len(T)):
            s[idx] = s[idx-1] + dt * ds(s[idx-1], i[idx-1])
            i[idx] = i[idx-1] + dt * di(s[idx-1], i[idx-1])
        return (s, i, T)
    return (SIS,)


@app.cell
def _(math, np):
    def SIS_ana_i_at_t(s0, i0, beta, gamma, t):
        R0 = beta/gamma
        return (1-1/R0) / (1 + ((1 - 1/R0 - i0)/i0) * math.exp(-(beta-gamma) * t))

    def SIS_ana(s0, i0, beta, gamma, duration, dt):
        T = np.arange(0, duration+dt, dt)
        i = np.array([SIS_ana_i_at_t(s0, i0, beta, gamma, t) for t in T])
        return (1-i, i, T)
    return (SIS_ana,)


@app.cell
def _(SIS, SIS_ana, plt, run):
    @run
    def _():
        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(15,3))

        sa, ia, Ta = SIS_ana(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=0.05)

        for dt, ax in zip([2, 1, 1/2], axs):
            s,i,T = SIS(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=dt)
            ax.plot(T,i, color='r', label='Forward Euler')
            ax.plot(Ta,ia, color='k', label='Analytical', linestyle='--')

            ax.set_title(f"SIS with Δt={dt}")
            ax.set_xlabel('time')
            ax.set_ylabel('infected portion')
            ax.set_ylim(bottom=0, top=0.5)
            ax.set_xlim(left=0)
            ax.legend()

        plt.savefig('out.nosync/q1a.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(SIS, SIS_ana, np):
    def SIS_max_err(s0, i0, beta, gamma, duration, dt):
        s,i,T = SIS(s0, i0, beta, gamma, duration, dt)
        sa,ia,Ta = SIS_ana(s0, i0, beta, gamma, duration, dt)
        err = np.abs(i-ia)
        return np.max(err)

    def SIS_final_err(s0, i0, beta, gamma, duration, dt):
        s,i,T = SIS(s0, i0, beta, gamma, duration, dt)
        sa,ia,Ta = SIS_ana(s0, i0, beta, gamma, duration, dt)
        err = np.abs(i-ia)
        return err[-1]
    return SIS_final_err, SIS_max_err


@app.cell
def _(SIS_max_err, plt, run):
    @run
    def _():
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
        dts = [1/2**(x-1) for x in range(7)]
        E = [SIS_max_err(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=dt) for dt in dts]
        ax.plot(dts,E, color='k', label='Forward Euler')

        ax.set_title(f"Error vs Step Size")
        ax.set_xlabel('Δt')
        ax.set_ylabel('E(Δt)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(dts)
        ax.set_xticklabels(f"{r[0]}" if (r:=x.as_integer_ratio())[1]==1 else f"{r[0]}/{r[1]}" for x in dts)
        plt.savefig('out.nosync/q1d.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(SIS_final_err, plt, run):
    @run
    def _():
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
        dts = [1/2**(x-1) for x in range(12)]
        E = [SIS_final_err(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=dt) for dt in dts]
        ax.plot(dts,E, color='k', label='Forward Euler')

        ax.set_title(f"Final Error vs Step Size")
        ax.set_xlabel('Δt')
        ax.set_ylabel('E(Δt)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(dts)
        ax.set_xticklabels(f"{r[0]}" if (r:=x.as_integer_ratio())[1]==1 else f"{r[0]}/{r[1]}" for x in dts)
        # plt.savefig('out.nosync/q1d-2.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(SIS, SIS_ana, SIS_final_err, plt, run, unique_legend):
    @run
    def _():
        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(15,3), constrained_layout=True)
        axi,axe,axf = axs
        axee = axe.twinx()
        dts = [1/2**(x-1) for x in range(10)]
        for dt in dts:
            _,i,T = SIS(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=dt)
            _,ia,_ = SIS_ana(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=dt)
            E = abs(i-ia)
            axee.plot(T, E, color='r', label='Euler Error')
            axi.plot(T, i, color='r', label='Euler')
        _,i,T = SIS_ana(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=0.1)
        axe.plot(T, i, color='k', label='Analytical')
        axi.plot(T, i, color='k', label='Analytical')

        axi.set_title("Euler Approximation")
        unique_legend(axi)
        axe.set_title("Euler Error")
        # unique_legend(axe)

        for ax in (axi, axe):
            ax.set_xlabel('time')
            ax.set_ylabel('infected')
        axee.set_ylabel('error', color="red")
        axee.tick_params(axis='y', labelcolor="red")

        E = [SIS_final_err(s0=0.99, i0=0.01, beta=3, gamma=2, duration=25, dt=dt) for dt in dts]
        axf.plot(dts,E, color='k', label='Forward Euler')
        axf.set_title(f"Euler Final Error vs Step Size")
        axf.set_xlabel('Δt')
        axf.set_ylabel('E(Δt)')
        axf.set_xscale('log')
        axf.set_yscale('log')
        axf.set_xticks(dts)
        axf.set_xticklabels(f"{r[0]}" if (r:=x.as_integer_ratio())[1]==1 else f"{r[0]}/{r[1]}" for x in dts)

        plt.savefig('out.nosync/q1d-2.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(eigvals, np, run):
    @run
    def _():
        G_0_Pemic = np.array([
            [3.1, 200/1800*42.9],
            [1800/200*4.77, 25.0],
        ])

        G_0_Uenza = np.array([
            [3.0, 210/1750*42.25],
            [1750/210*5.07, 25.1],
        ])

        R_0_Pemic = float(np.max(eigvals(G_0_Pemic)).real)
        R_0_Uenza = float(np.max(eigvals(G_0_Uenza)).real)
        print(f"{R_0_Pemic = }")
        print(f"{R_0_Uenza = }")
    return


@app.cell
def _():
    import marimo as mo
    def run(f): f()
    def unique_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    return mo, run, unique_legend


if __name__ == "__main__":
    app.run()
