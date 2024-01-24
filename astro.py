import numpy as np
from physconst import DAY,YR, HOUR
from multistar import multi as M
from multistar.constants import STATUS_COLLIDE
from multistar.grid.distributions import euleruniformdeg
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

template = "~/Monash/summer2024/arrokoth_template.toml"
template_w = "~/Monash/summer2024/arrokoth_template_w.toml"
template_monopole = "~/Monash/summer2024/arrokoth_template_monopole.toml"

class Astro(object):
    def save(self, filename):
        filename = Path(filename).expanduser()
        with filename.open("wb") as f:
            dump(self, f)
    
    @classmethod
    def load(cls, filename):
        filename = Path(filename).expanduser()
        with open(filename, 'rb') as f:
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
    def __init__(self, search_space = [(0, 90), (17,22)], res = (100, 100)):
        ydeg = np.linspace(search_space[0][0], search_space[0][1], res[0])
        distance = np.linspace(search_space[1][0], search_space[1][1], res[1])
        self.dist = distance
        self.ydeg = ydeg

        self.result = []


    def create_batch(self, batch):
        with contextlib.redirect_stdout(None):
            ydeg, dist = batch
            m = M(template, update={'trajectory.rp_km': dist, 'body 1.orientation_deg': [0, ydeg, 0]}, silent=True)
            m.rund(3 * YR, silent=True)

            # collision = (m.status == STATUS_COLLIDE)
            # collide = 1 if collision else 0
            # orbits = signal.find_peaks(m.ron.flatten())[0]
            # time_of_pass = [m.t[i] for i in orbits]


            r = {
                "ydeg": ydeg,
                # "collide": collide,
                "time": m.t[-1],
                "rp": dist,
                "erot": m.erot[0, -1],
                # "radius": m.ro[0,-1],
                # "velocity": m.vo[0,-1],
                "orbits": len(signal.find_peaks(m.ron.flatten())[0])
                }
        return r

    def run(self):
        results = []
        items = [(i, j) for i in self.ydeg for j in self.dist]

        with multiprocessing.Pool() as pool, tqdm(total=len(items), miniters=len(items)/100) as pbar:
            start_time = time.time()
            for batch_results in pool.imap_unordered(self.create_batch, items, chunksize=10):
                results.append(batch_results)
                pbar.update(1)

        self.result = results


    def plot2(self, plot_orbits=False, plot_erot=False):
        fig, ax = plt.subplots()
        deg_r = np.array([x['ydeg'] for x in self.result])
        dist_r = np.array([x['rp'] for x in self.result])

        deg = np.unique(deg_r)
        dist = np.unique(dist_r)
        ndist = len(dist)
        ndeg = len(deg)
        time = np.ndarray((ndist, ndeg))
        degi = np.searchsorted(deg, deg_r)
        disti = np.searchsorted(dist, dist_r)

        if plot_orbits:
            orbits = np.ndarray((ndist, ndeg))
            orbits[disti, degi] = np.array([x['orbits'] for x in self.result])
            max_orbits = int(np.max(orbits))
            min_orbits = 0
        elif plot_erot:
            erot = np.ndarray((ndist, ndeg))
            erot[disti, degi] = np.array([x['erot'] for x in self.result])

        time[disti, degi] = np.array([x['time'] for x in self.result])

        if plot_orbits:
            mesh1 = ax.pcolormesh(dist, deg, orbits.T)
            ax.set_xlabel("distance (km)")
            ax.set_ylabel("gamma orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", shrink = 0.5, label = "no of orbits before collision")  
        elif plot_erot:
            mesh1 = ax.pcolormesh(dist, deg, np.log10(erot.T))
            ax.set_xlabel("distance (km)")
            ax.set_ylabel("gamma orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", shrink = 0.5, label = "log rotational energy (ergs)")  
        else:
            mesh1 = ax.pcolormesh(dist, deg, time.T/DAY)
            ax.set_xlabel("distance (km)")
            ax.set_ylabel("gamma orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", shrink = 0.5, label = "time to collision (days)")
            
    def plot_scatter(self):
        time = np.array([x['time'] for x in self.result])
        erot = np.array([x['erot'] for x in self.result])
        rp = np.array([x['rp'] for x in self.result])

        max_time_i = np.argwhere(time==time.max())

        time = np.delete(time, max_time_i)
        erot = np.delete(erot, max_time_i) 
        rp = np.delete(rp, max_time_i)

        print(len(time))
        print(len(erot))
        print(len(rp))

        # time = np.unique(time_r)
        # erot = np.unique(erot_r)

        fig, ax = plt.subplots()

        sc = ax.scatter(erot, time/DAY, c=rp)
        # mesh1 = ax.pcolormesh(erot, time, rp.T)
        ax.set_xlabel("rotational energy (ergs)")
        ax.set_ylabel("time (days)")
        cbar = fig.colorbar(sc, ax=ax, location="right", shrink = 0.5, label = "periapsis distance (km)")

        
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
            self._zdeg = np.linspace(60, 65, 3000)
            self._period = np.linspace(20, 30, 1000)
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
                "orbits": len(signal.find_peaks(m.ron.flatten())[0]),
                "eccentricity": m.eno(0)[-1],
            }
        return r

    def run(self):
        results = []
        items = [(i, j) for i in self._period for j in self._zdeg]

        with multiprocessing.Pool() as pool, tqdm(total=len(items)) as pbar:
            start_time = time.time()
            for batch_results in pool.imap_unordered(self.create_batch, items, chunksize=10):
                results.append(batch_results)
                pbar.update(1)

        self.result = results


            

    def plot2(self, plot_orbits=False, plot_eccentricity=False):
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

        elif plot_eccentricity:
            eccentricity = np.ndarray((nperiod, ndeg))
            # eccentricity[periodi, degi] = np.array([x['eccentricity'] if x['collide'] == False else np.NAN for x in self.result ] ) # if the object has collided we want a NAN entry for plotting clarity
            eccentricity[periodi, degi] = np.array([x['eccentricity'] for x in self.result ] ) 

        time[periodi, degi] = np.array([x['time'] for x in self.result])
        
        max_time = np.max(time)
        min_time = np.min(time)

        if plot_orbits:
            mesh1 = ax.pcolormesh(period, deg, np.log10(orbits.T))
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.log10([1,2,5,10]), shrink = 0.5, label = "no of orbits before collision") 
            cbar.set_ticklabels(['1', '2', '5', '10'])
        elif plot_eccentricity:
            mesh1 = ax.pcolormesh(period, deg, eccentricity.T) 
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", shrink = 0.5, label = "final orbit eccentricity")  
        else:
            mesh1 = ax.pcolormesh(period, deg, time.T/DAY)
            ax.set_xlabel("period (days)")
            ax.set_ylabel("orientation (degrees)")
            cbar = fig.colorbar(mesh1, ax=ax, location="right", ticks = np.linspace(min_time/DAY, max_time/DAY, 5), shrink = 0.5, label = "time to collision (days)")

class AstroYOrientationW(Astro):
    # w in revs/hr
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

class AstroRandOrientationDist(Astro):
    # w in revs/hr
    def __init__(self, search_space = (15, 22), res: int = 100, seed=None):
        self.rng = np.random.default_rng(seed)
        self.res = res
        self.dist = np.linspace(search_space[0], search_space[1], res)

    def create_batch(self, batch):
        with contextlib.redirect_stdout(None):
            dist, orientation = batch
            m = M(template, update={'trajectory.rp_km': dist, 'body 1.orientation_deg': list(orientation)}, silent=True)
            m.rund(1 * YR, silent=True)

            # collision = (m.status == STATUS_COLLIDE)
            # collide = 1 if collision else 0
            # orbits = signal.find_peaks(m.ron.flatten())[0]
            # time_of_pass = [m.t[i] for i in orbits]


            r = {
                # "orientation": orientation,
                # "collide": collide,
                "time": m.t[-1],
                # "passes": [(passes, t) for passes in orbits for t in time_of_pass],
                "distance": dist,
                # "velocity": m.vo[0,-1],
                # "orbits": len(signal.find_peaks(m.ron.flatten())[0])
                }
        return r
    
    def run(self):
        results = []
        orientations = [euleruniformdeg(rng = self.rng, size=1) for _ in range(self.res)]
        items = [(i, j.flatten()) for i in self.dist for j in orientations]

        with multiprocessing.Pool() as pool, tqdm(total=len(items), miniters=len(items)/100) as pbar:
            start_time = time.time()
            for batch_results in pool.imap_unordered(self.create_batch, items, chunksize=10):
                results.append(batch_results)
                pbar.update(1)

        self.result = results    

    def plot2(self, plot_orbits=False):
        fig, ax = plt.subplots()
        dist_r = np.array([x['distance'] for x in self.result])
        dist = np.unique(dist_r)
        ndist = len(dist)
        time = np.ndarray(ndist)
        disti = np.searchsorted(dist, dist_r)

        time[disti] = np.array([x['time'] for x in self.result])
        time_bins = np.linspace(((time/DAY).min()), ((time/DAY).max()))
        histogram, xedges, yedges = np.histogram2d(dist, time/DAY, bins = [self.res//5, time_bins])
        x, y = np.meshgrid(xedges, yedges)

        mesh1 = ax.pcolormesh(x, y, histogram.T)
        ax.set_xlabel("periapsis distance (km)")
        ax.set_ylabel("time (days)")
        cbar = fig.colorbar(mesh1, ax=ax, location="right", shrink = 0.5, label = "count")