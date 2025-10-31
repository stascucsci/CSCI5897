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
    return eigvals, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Forward Euler solver""")
    return


@app.cell
def _(np):
    def sir(s0, i0, r0, pC, gamma, t_max, stepsize):
        T = np.arange(0,t_max+stepsize,stepsize)
        G = len(s0)
        s = np.zeros([len(T),G])
        i = np.zeros([len(T),G])
        r = np.zeros([len(T),G])

        for idx,t in enumerate(T):
            if idx==0:
                s[idx] = s0
                i[idx] = i0
                r[idx] = r0
            else:
                Q = np.diag(s[idx-1,:]) @ pC @ i[idx-1,:]
                ds_dt = -Q
                di_dt = Q - gamma*i[idx-1,:]
                dr_dt = gamma*i[idx-1,:]

                s[idx] = s[idx-1,:] + ds_dt * stepsize
                i[idx] = i[idx-1,:] + di_dt * stepsize
                r[idx] = r[idx-1,:] + dr_dt * stepsize

        return s, i, r, T
    return (sir,)


@app.cell
def _(eigvals, np, run):
    from scipy.optimize import root_scalar

    @run
    def _():
        def G0c(c):
            return np.array([[i/3*c]*4 for i in range(1,5)])
        def R0(G0):
            return float(np.max(eigvals(G0)).real)

        c = round(root_scalar(lambda c: R0(G0c(c))-1.5, bracket=[0.1, 10]).root, 10)

        print(f"c={c} which gives R0 of", R0(G0c(c)))
    return


@app.cell
def _(np, plt, run, sir):
    def Q1_sim():
        c = 0.45
        pC = np.array([[i/3*c]*4 for i in range(1,5)])
        s,i,r,T = sir([0.999]*4,[0.001]*4,[0]*4, pC,1,30,0.01)
        return s, i, r, T

    @run
    def _():
        s,i,r,T = Q1_sim()
        fig, ax = plt.subplots()
        for g in range(4):
            ax.plot(T,i[:,g], color=(0.6+0.4*(1-g/3),0,0), linewidth=(g+1)/4*2, label=f"$i_{g+1   }(t)$")
        ax.set_xlabel('time')
        ax.set_ylabel('population portion')
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.set_title("Infected vs Time")
        plt.savefig('out.nosync/q1c.svg', bbox_inches='tight')
        plt.show()
    return (Q1_sim,)


@app.cell
def _(Q1_sim, plt, run):
    @run
    def _():
        s,i,r,T = Q1_sim()
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
        ax = axs[0]
        for g in range(4):
            ax.plot(T,s[:,g], color=(0.6+0.4*(1-g/3),0,0), linewidth=(g+1)/4*2, label=f"$s_{g+1}(t)$")
        ax.set_xlabel('time')
        ax.set_ylabel('population portion')
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_title("Susceptible vs Time")

        ax = axs[1]
        P = [1,2,3,4]
        p_t = [sum(P[g] * s[i,g] for g in range(4))/sum(s[i,:]) for i,t in enumerate(T)]
        ax.plot(T,p_t, color="k")
        ax.set_ylabel(r"$\overline{p}\,(t)$")
        ax.set_xlabel('time')
        ax.set_title("Average Relative Susceptibility")

        plt.savefig('out.nosync/q1d.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(np):
    from scipy.stats import nbinom

    def NB(R0, k, N=1) -> np.ndarray[int]:
        mean = R0
        variance = mean + (mean**2)/k
        p = mean/variance
        n = mean**2 / (variance - mean)
        return nbinom.rvs(n=n, p=p, size=int(N))

    def BP(R0, k, G, cutoff=None):
        I = np.zeros(G)
        I[0] = 1
        for g in np.arange(1, G):
            I[g] = np.sum(NB(R0, k, I[g-1]))
            if I[g] == 0 or (cutoff and I[g] > cutoff):
                I[G-1] = I[g]
                break
        return I
    return (BP,)


@app.cell
def _(BP, plt, run):
    @run
    def _():
        fig,ax = plt.subplots()
        for i in range(30):
            trajectory = BP(3, 1, 10)
            ax.semilogy(trajectory,'-o')

        ax.set_xlabel('generation')
        ax.set_ylabel('branching process size')
        plt.show()
    return


@app.cell
def _(BP, mo, np):
    # @run
    def _():
        R0 = 3
        ks = [0.1, 0.5, 1.0, 5.0, 10.0]
        G = 10 # 12
        n_tests = 1000 # 2000
        results = []
        with mo.status.progress_bar(range(n_tests * len(ks))) as bar:
            for k in ks:
                died = 0
                for _ in range(n_tests):
                    if BP(R0, k, G)[-1] == 0:
                        died += 1
                    bar.update()
                results.append((k, died/n_tests))

        print("\n".join(f"{r[0]}, {r[1]}" for r in results))
        np.savetxt("out.nosync/q2a.csv", results, "%.3f", ",")
        # return mo.ui.table([{'k':r[0], 'q':r[1]} for r in results])
    return


@app.cell
def _(BP, cache, mo, np, plt, run):
    @cache
    def _compute():
        K = np.arange(0.1, 10, 0.01)
        Q = np.zeros(len(K))
        R0 = 3
        G = 20
        n_tests = 10000
        for i, k in mo.status.progress_bar(list(enumerate(K))):
            Q[i] = sum(BP(R0, k, G, cutoff=100)[-1] == 0 for _ in range(n_tests))/n_tests
        return K, Q

    @run
    def _():
        K, Q = _compute()

        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,5))
        ax = axs[0]
        ax.plot(K,Q, color="k")
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$q$')
        ax.set_title("$q$ vs $k$")

        ax = axs[2]
        ax.plot(K,1-Q, color="k")
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$p$')
        ax.set_title("$p$ vs $k$")

        ax = axs[1]
        ax.plot(K,Q, color="k")
        ax.set_yscale('log')
        ax.set_ylim(top=1)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$q$')
        ax.set_title("$q$ vs $k$ (log scale)")

        plt.savefig('out.nosync/q2b.svg', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(BP, cache, mo, np, plt, run):
    _N = 100_000

    @cache(ignore=["bar"])
    def compute_finite_outbreak_sizes(k=1, N=_N, bar=None):
        F = np.zeros(N)
        Fi = 0
        R0 = 3
        G = 20
        while True:
            f = BP(R0, k, G, cutoff=100)
            if f[-1] == 0:
                F[Fi] = np.max(f)
                Fi += 1
                if bar: bar.update()
                if Fi >= N:
                    break
        return F

    def _(f):
        def _(k=1,N=_N,bar=None):
            if f.check_call_in_cache(k,N,bar):
                if bar: bar.update(N)
            return f(k,N,bar)
        return _
    compute_finite_outbreak_sizes = _(compute_finite_outbreak_sizes)

    @run
    def _():
        K = [0.1, 0.5, 1, 2, 5, 10]
        with mo.status.progress_bar(range(_N*len(K))) as bar:
            for k in K:
                compute_finite_outbreak_sizes(k, bar=bar)
        fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(20,5*2))
        for ax,k in zip(axs.flat, K):
            F = compute_finite_outbreak_sizes(k)
            ax.hist(F)
            ax.set_xlabel('Maximum Outbreak Size')
            ax.set_ylabel('Instances')
            ax.set_ylim(bottom=0, top=_N)
            ax.set_title(f"Maximum Outbreak Size with $k={k}$")
            ax.locator_params(axis='x', integer=True)

        plt.savefig('out.nosync/q2e.svg', bbox_inches='tight')
        plt.show()
    return (compute_finite_outbreak_sizes,)


@app.cell
def _(compute_finite_outbreak_sizes, mo, np, plt, run):
    @run
    def _():
        N = 100_000
        K = np.arange(0.1, 10, 0.1)
        F = np.zeros(len(K))
        avg = 10
        with mo.status.progress_bar(range(N*len(K)*avg)) as bar:
            for i,k in enumerate(K):
                for a in range(avg):
                    F[i] += np.max(compute_finite_outbreak_sizes(k, N+a, bar))
                F[i] /= avg

        fig,ax = plt.subplots()
        ax.plot(K, F, color="k")
        ax.set_ylim(bottom=0)
        ax.set_xlabel('$k$')
        ax.set_ylabel('maximum outbreak size')
        ax.set_title('maximum outbreak size vs $k$ for 1M finite simulations with $R_0=3$')

        plt.savefig('out.nosync/q2e2.svg', bbox_inches='tight')
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
    return cache, mo, run


if __name__ == "__main__":
    app.run()
