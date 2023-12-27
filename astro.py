import numpy as np
from physconst import DAY,YR, HOUR
from multistar import multi as M
from multistar.constants import STATUS_COLLIDE
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import ticker
from pickle import load, dump
from pathlib import Path
from scipy import signal
import multiprocessing
import contextlib
import time
from math import log10
from tqdm import tqdm
import scales

template = "~/Monash/summer2024/arrokoth_template.toml"
template_w = "~/Monash/summer2024/arrokoth_template_w.toml"

class Astro(object):
    def save(self, filename):
        filename = Path(filename).expanduser()
        with filename.open("wb") as f:
            dump(self, f)
    
    @classmethod
    def load(cls, filename):
        filename = Path(filename).expanduser()
        with filename.open("rb") as f:
            self = load(f)
        return self
    
    def _create_log_ticks(self,max: int):
        ticks = [1.0,2.0,5.0]
        if max < 10:
            ticks.append(max)
        else:
            elements = list(np.logspace(1, log10(max), num = int(log10(max))%10 + 1).round(decimals=1))
            for item in elements:
                ticks.append(item)

        ticks = sorted(ticks)
        labels = [str(i) for i in ticks] 
        return ticks, labels       



class AstroDist(Astro):
    def __init__(self):
        distance = np.linspace(10,20,101)
        result = list()

        for d in distance:
            m = M(template, update={'trajectory.rp_km': d})
            m.rund(100*DAY)
            result.append((d,m))
        self.result = result

    def plot(self):
        distance = list()
        collide = list()
        for r in self.result:
            distance.append(r[0])
            collision = (r[1].status == STATUS_COLLIDE)
            if collision:
                collide.append(1)
            else:
                collide.append(0)
        
        fig, ax = plt.subplots()
        ax.plot(distance, collide)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Collision (yes/no)")
        fig.tight_layout()

class AstroZOrientation(Astro):
    def __init__(self):
        zdeg = np.linspace(0, 90, 91) # increments of 1-degree hardcoded for now
        result = list()

        for angle in zdeg:
            m = M(template, update={'body 1.orientation_deg': [0, angle, 0]})
            m.rund(100*DAY)
            result.append((angle, m))
        self.result = result

    def plot(self):
        z_orientation = list()
        collide = list()
        for r in self.result:
            z_orientation.append(r[0])
            collision = (r[1].status == STATUS_COLLIDE)
            if collision:
                collide.append(1)
            else:
                collide.append(0)
        
        fig, ax = plt.subplots()
        ax.plot(z_orientation, collide)
        ax.set_xlabel("z-axis orientationn (km)")
        ax.set_ylabel("Collision (yes/no)")
        fig.tight_layout()

class AstroYOrientationDist(Astro):
    def __init__(self):
        ydeg = np.linspace(0, 90, 91) # increments of 1-degree hardcoded for now
        distance = np.linspace(17.8,19.2,91)
        self.dist = distance
        result = list()
        for d in distance:
            partial = list()
            for angle in ydeg:
                m = M(template, update={'trajectory.rp_km': d, 'body 1.orientation_deg': [0, angle, 0]})
                m.rund(10*DAY)
                collision = (m.status == STATUS_COLLIDE)
                if collision:
                    collide = 1
                else:
                    collide = 0
                r = dict()
                r["ydeg"] = angle
                r["collide"] = collide
                partial.append(r)
            result.append(partial)

        
        self.result = result


    def plot(self):
        fig, ax = plt.subplots()
        ndist = len(self.result)
        ndeg = len(self.result[0])
        deg = np.array([x['ydeg'] for x in self.result[0]])
        collide = np.ndarray((ndist,ndeg))

        for idist in range(ndist):
            for ideg, y in enumerate(self.result[idist]):
                collide[idist,ideg] = y["collide"]

        mesh1 = ax.pcolormesh(self.dist, deg, collide.T)
        cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks=[0.0,1.0], shrink = 0.5)
        cbar.set_ticklabels(["escape", "collision"]) 
        cbar.ax.tick_params(axis='y', which='both', left=False, right=False)
        
# 2d plot of orientation and rotationperiod
class AstroYOrientationPeriod(Astro):
    def __init__(self):
        zdeg = np.linspace(40, 140, 91) # increments of 1-degree hardcoded for now
        period = np.linspace(0.1, 60, 60)
        self.period = period
        result = list()
        for p in period:
            partial = list()
            for angle in zdeg:
                m = M(template, update={'body 1.P_d': p, 'body 1.orientation_deg': [0, angle, 0]})
                m.rund(100*DAY)
                collision = (m.status == STATUS_COLLIDE)
                if collision:
                    collide = 1
                else:
                    collide = 0
                r = dict()
                r["ydeg"] = angle
                r["period"] = period
                r["collide"] = collide
                partial.append(r)
            result.append(partial)

        
        self.result = result


    def plot(self):
        fig, ax = plt.subplots()
        nperiod = len(self.result)
        ndeg = len(self.result[0])
        deg = np.array([x['ydeg'] for x in self.result[0]])
        # period = np.array([x['period'] for x in self.result[0]])
        collide = np.ndarray((nperiod,ndeg))
        cmap = ListedColormap(['purple', 'yellow'])

        for iperiod in range(nperiod):
            for ideg, y in enumerate(self.result[iperiod]):
                collide[iperiod,ideg] = y["collide"]

        mesh1 = ax.pcolormesh(self.period, deg, collide.T, cmap = cmap)
        ax.set_xlabel("period (days)")
        ax.set_ylabel("orientation (degrees)")
        # fig.colorbar(mesh1, ax=ax, location="right", ticks=["collision", "escape"])
        cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks=[0.0,1.0], shrink = 0.5)
        cbar.set_ticklabels(["escape", "collision"]) 
        cbar.ax.tick_params(axis='y', which='both', left=False, right=False)

# Parallelised (to do)
class AstroYOrientationPeriod2(Astro):
    def __init__(self):
        self.result = []

    def create(self, p, angle):
        with contextlib.redirect_stdout(None):
            m = M(template, update={'body 1.P_d': p, 'body 1.orientation_deg': [0, angle, 0]}, silent=True)
            m.rund(100 * DAY, silent = True)
            collision = (m.status == STATUS_COLLIDE)
            collide = 1 if collision else 0
            r = {
                "ydeg": angle,
                "period": p,
                "collide": collide,
                "time": m.t[-1],
                "radius": m.ron.flatten(),
                "orbits": len(signal.find_peaks(m.ron.flatten())[0])
            }
            return r

    def run(self):
        zdeg = np.linspace(60, 65, 200)
        period = np.linspace(20, 30, 200)

        results = []
        items = [(i, j) for i in period for j in zdeg]

        with tqdm(total=len(items), desc="Single-threaded Progress") as pbar:
            start_time = time.time()

            for p, angle in items:
                result = self.create(p, angle)
                results.append(result)
                pbar.update(1)

            elapsed_time = time.time() - start_time
            print(f"Single-threaded execution time: {elapsed_time:.2f} seconds")

        self.result = results

            

    def plot2(self, plot_orbits=False):
        fig, ax = plt.subplots()
        deg_r = np.array([x['ydeg'] for x in self.result])
        period_r = np.array([x['period'] for x in self.result])
        deg = np.unique(deg_r)
        period = np.unique(period_r)
        nperiod = len(period)
        ndeg = len(deg)
        time = np.ndarray((nperiod, ndeg))
        degi = np.searchsorted(deg, deg_r)
        periodi = np.searchsorted(period, period_r)

        if plot_orbits:
            orbits = np.ndarray((nperiod, ndeg))
            orbits[periodi, degi] = np.array([x['orbits'] for x in self.result])
            max_orbits = int(np.max(orbits))
            min_orbits = 0
        # cmap = ListedColormap(['purple', 'yellow'])



        time[periodi, degi] = np.array([x['time'] for x in self.result])
        
        max_time = np.max(time)
        min_time = np.min(time)

        # for iperiod in range(nperiod):
        #     for ideg, y in enumerate(self.result[iperiod]):
        #         collide[iperiod,ideg] = y["collide"]

        # mesh1 = ax.pcolormesh(period, deg, collide.T)
        if plot_orbits:
            mesh1 = ax.pcolormesh(np.log(period), deg, orbits.T)
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.linspace(min_orbits, max_orbits, max_orbits+1), shrink = 0.5, label = "no of orbits before collision")    
        else:
            mesh1 = ax.pcolormesh(period, deg, time.T/DAY)
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.linspace(min_time/DAY, max_time/DAY, 5), shrink = 0.5, label = "time to collision (days)")

# For now this class will store the orbital radii and corresponding times
# We want to investigate how many orbits are made before collision
class OrbitRadius(Astro):
    def __init__(self, params: dict = {'trajectory.rp_km': 15, 'body 1.orientation_deg': [0,0,0], 'body 1.P_d': 0}):
        super().__init__()
        m = M(template, update=params)
        m.runs(150*DAY/1000000, 1000000)
        # result = list()
        r = dict()
        r['time'] = m.t
        r['radius'] = m.ron.flatten()
        self.result = r

    @property
    def no_orbits(self):
        orbit_radius = self.result['radius']
        no_of_peaks = signal.find_peaks(orbit_radius) 
        return len(no_of_peaks[0])
           
    def plot(self):
        fig, ax = plt.subplots()
        result = self.result
        print(result['time'])
        ax.set_xscale('timescale')
        ax.set_ylabel('orbital radius')
        ax.set_xlabel('time (days)')
        ax.plot(result['time'], result['radius'])

class AstroYOrientationPeriod3(Astro):
    def __init__(self, pmode = 'linear'):
        if pmode == 'linear':
            self._zdeg = np.linspace(60, 65, 200)
            self._period = np.linspace(20, 30, 200)
        elif pmode == 'log':
            self._zdeg = np.linspace(66, 70, 100)
            self._period = np.logspace(np.log10(8.5),np.log10(9.6), 100)

        self.pmode = pmode
        self.result = []

    def create_batch(self, batch):
        with contextlib.redirect_stdout(None):
            p, angle = batch
            m = M(template, update={'body 1.P_d': p, 'body 1.orientation_deg': [0, angle, 0]}, silent=True)
            m.rund(1 * YR, silent=True)

            collision = (m.status == STATUS_COLLIDE)
            collide = 1 if collision else 0

            r = {
                "ydeg": angle,
                "period": p,
                "collide": collide,
                "time": m.t[-1],
                "radius": m.ro[0,-1],
                "velocity": m.vo[0,-1],
                "orbits": len(signal.find_peaks(m.ron.flatten())[0])
            }
        return r

    def run(self):
        results = []
        items = [(i, j) for i in self._period for j in self._zdeg]

        with multiprocessing.Pool(16) as pool, tqdm(total=len(items)) as pbar:
            start_time = time.time()
            for batch_results in pool.imap_unordered(self.create_batch, items, chunksize=10):
                results.append(batch_results)
                pbar.update(1)

        self.result = results


            

    def plot2(self, plot_orbits=False):
        fig, ax = plt.subplots()
        deg_r = np.array([x['ydeg'] for x in self.result])
        period_r = np.array([x['period'] for x in self.result])
        deg = np.unique(deg_r)
        period = np.unique(period_r)
        nperiod = len(period)
        ndeg = len(deg)
        time = np.ndarray((nperiod, ndeg))
        degi = np.searchsorted(deg, deg_r)
        periodi = np.searchsorted(period, period_r)

        if plot_orbits:
            orbits = np.ndarray((nperiod, ndeg))
            orbits[periodi, degi] = np.array([x['orbits'] for x in self.result])
            max_orbits = np.max(orbits)
            min_orbits = 0
            print(np.argwhere(orbits == 34))

        time[periodi, degi] = np.array([x['time'] for x in self.result])
        
        max_time = np.max(time)
        min_time = np.min(time)

        if plot_orbits:
            mesh1 = ax.pcolormesh(period, deg, np.log10(orbits.T))
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.log10([1,2,5,10]), shrink = 0.5, label = "no of orbits before collision") 
            cbar.set_ticklabels(['1', '2', '5', '10'])   
        else:
            mesh1 = ax.pcolormesh(period, deg, time.T/DAY)
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.linspace(min_time/DAY, max_time/DAY, 5), shrink = 0.5, label = "time to collision (days)")

class AstroYOrientationW(Astro):
    def __init__(self, search_space = [(80,90), (-0.02, 0.02)], res = (100,100), pmode = 'linear'):
        if pmode == 'linear':
            self._zdeg = np.linspace(search_space[0][0], search_space[0][1], res[0])
            self._w = np.linspace(search_space[1][0], search_space[1][1], res[1])*2*np.pi/HOUR
        elif pmode == 'log':
            self._zdeg = np.linspace(search_space[0][0], search_space[0][1], res[0])
            self._w = np.logspace(search_space[1][0], search_space[1][1], res[1])*2*np.pi/HOUR

        self.pmode = pmode
        self.result = []

    def create_batch(self, batch):
        with contextlib.redirect_stdout(None):
            w, angle = batch
            p = 2*np.pi/(w*DAY) if w != 0 else 0
            # j = [0, 0, np.sign(w)]
            m = M(template_w, update={'body 1.wn': w, 'body 1.orientation_deg': [0, angle, 0]}, silent=True)
            m.rund(1 * YR, silent=True)

            collision = (m.status == STATUS_COLLIDE)
            collide = 1 if collision else 0
            orbits = signal.find_peaks(m.ron.flatten())[0]
            time_of_pass = [m.t[i] for i in orbits]


            r = {
                "ydeg": angle,
                "period": p,
                "collide": collide,
                "ang_vel": w/(2*np.pi)*HOUR,
                "time": m.t[-1],
                "passes": [(passes, t) for passes in orbits for t in time_of_pass],
                "radius": m.ro[0,-1],
                "velocity": m.vo[0,-1],
                "orbits": len(signal.find_peaks(m.ron.flatten())[0])
                }
        return r

    def run(self):
        results = []
        items = [(i, j) for i in self._w for j in self._zdeg]

        with multiprocessing.Pool(16) as pool, tqdm(total=len(items)) as pbar:
            start_time = time.time()
            for batch_results in pool.imap_unordered(self.create_batch, items, chunksize=10):
                results.append(batch_results)
                pbar.update(1)

        self.result = results    

    def plot2(self, plot_orbits=False):
        fig, ax = plt.subplots()
        deg_r = np.array([x['ydeg'] for x in self.result])
        w_r = np.array([x['ang_vel'] for x in self.result])
        deg = np.unique(deg_r)
        w = np.unique(w_r)
        ndeg = len(deg)
        nw = len(w)
        time = np.ndarray((nw, ndeg))
        degi = np.searchsorted(deg, deg_r)
        wi = np.searchsorted(w, w_r)

        if plot_orbits:
            orbits = np.ndarray((nw, ndeg))
            orbits[wi, degi] = np.array([x['orbits'] for x in self.result])
            max_orbits = float(np.max(orbits))
            min_orbits = 0

        time[wi, degi] = np.array([x['time'] for x in self.result])
        
        max_time = np.max(time)
        min_time = np.min(time)

        if plot_orbits:
            mesh1 = ax.pcolormesh(w, deg, np.log10(orbits.T))
            ax.set_xlabel("angular velocity (rev/hr)")
            ax.set_ylabel("orientation (degrees)")
            ticks, labels = self._create_log_ticks(max = max_orbits)
            print(ticks)
            print(labels)
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.log10(ticks), shrink = 0.5, label = "no of orbits before collision")   
            cbar.set_ticklabels(labels)
        else:
            mesh1 = ax.pcolormesh(w, deg, time.T/DAY)
            ax.set_xlabel("angular velocity (rev/hr)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.linspace(min_time/DAY, max_time/DAY, 5), shrink = 0.5, label = "time to collision (days)")

class AstroW(Astro):
    def __init__(self, search_space = [(-0.02,0.02), (-0.02, 0.02)], res = (100,100), pmode = 'linear'):
        if pmode == 'linear':
            self._w1 = np.linspace(search_space[0][0], search_space[0][1], res[0])*2*np.pi/HOUR
            self._w2 = np.linspace(search_space[1][0], search_space[1][1], res[1])*2*np.pi/HOUR
        elif pmode == 'log':
            self._w1 = np.linspace(search_space[0][0], search_space[0][1], res[0])*2*np.pi/HOUR
            self._w2 = np.logspace(search_space[1][0], search_space[1][1], res[1])*2*np.pi/HOUR

        self.pmode = pmode
        self.result = []

    def create_batch(self, batch):
        with contextlib.redirect_stdout(None):
            w1, w2 = batch
            p1 = 2*np.pi/(w1*DAY) if w1 != 0 else 0
            p2 = 2*np.pi/(w2*DAY) if w2 != 0 else 0
            # j = [0, 0, np.sign(w)]
            m = M(template_w, update={'body 1.wn': w1, 'body 2.wn': w2, 'body 1.orientation_deg': [0,90,0], 'body 2.orientation_deg': [0,90,0]}, silent=True)
            m.rund(1 * YR, silent=True)

            collision = (m.status == STATUS_COLLIDE)
            collide = 1 if collision else 0

            r = {
                "period": (p1,p2),
                "collide": collide,
                "ang_vel": (w1/(2*np.pi)*HOUR,w2/(2*np.pi)*HOUR),
                "time": m.t[-1],
                "radius": m.ro[0,-1],
                "velocity": m.vo[0,-1],
                "orbits": len(signal.find_peaks(m.ron.flatten())[0])
                }
        return r

    def run(self):
        results = []
        items = [(i, j) for i in self._w1 for j in self._w2]

        with multiprocessing.Pool(16) as pool, tqdm(total=len(items)) as pbar:
            start_time = time.time()
            for batch_results in pool.imap_unordered(self.create_batch, items, chunksize=10):
                results.append(batch_results)
                pbar.update(1)

        self.result = results    

    def plot2(self, plot_orbits=False):
        fig, ax = plt.subplots()
        w1_r = np.array([x['ang_vel'][0] for x in self.result])
        w2_r = np.array([x['ang_vel'][1] for x in self.result])
        w1 = np.unique(w1_r)
        w2 = np.unique(w2_r)
        nw1 = len(w1)
        nw2 = len(w2)
        time = np.ndarray((nw1, nw2))
        w1i = np.searchsorted(w1, w1_r)
        w2i = np.searchsorted(w2, w2_r)

        if plot_orbits:
            orbits = np.ndarray((nw1, nw2))
            orbits[w1i, w2i] = np.array([x['orbits'] for x in self.result])
            max_orbits = float(np.max(orbits))
            min_orbits = 0

        time[w1i, w2i] = np.array([x['time'] for x in self.result])
        
        max_time = np.max(time)
        min_time = np.min(time)

        if plot_orbits:
            mesh1 = ax.pcolormesh(w1, w2, np.log10(orbits.T))
            ax.set_xlabel("body 1 angular velocity (rev/hr)")
            ax.set_ylabel("body 2 angular velocity (rev/hr)")
            ticks, labels = self._create_log_ticks(max = max_orbits)
            print(ticks)
            print(labels)
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.log10(ticks), shrink = 0.5, label = "no of orbits before collision")   
            cbar.set_ticklabels(labels)
        else:
            mesh1 = ax.pcolormesh(w1, w2, time.T/DAY)
            ax.set_xlabel("body 1 angular velocity (rev/hr)")
            ax.set_ylabel("body 2 angular velocity (rev/hr)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.linspace(min_time/DAY, max_time/DAY, 5), shrink = 0.5, label = "time to collision (days)")