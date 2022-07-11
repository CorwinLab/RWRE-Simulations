import sys
import os

# Need to link to diffusionPDF library (PyBind11 code)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DiffusionCDF")
sys.path.append(path)

import diffusionCDF
import numpy as np
import npquad
import csv
import fileIO
import json
import time


class DiffusionTimeCDF(diffusionCDF.DiffusionTimeCDF):
    """
    Class to simulate cumulative distribution function of random walks
    in random environments. This is a wrapper with helper functions
    for the C++ class.

    Parameters
    ----------
    beta : float
        Value of beta for the beta distribution to draw random probabilities
        from.

    tMax : int
        Maximum time that will be iterated to.

    Attributes
    ----------
    time : int
        Current time of the system

    beta : float
        Value of beta for the beta distribution to draw random probabilities
        from.

    CDF : numpy array (dtype of np.quad)
        The current recurrance vector Z_B(n, t) in the original Barraquand-Corwin paper.
        This relates to the CDF of the system through the relation:
        Z_B(n, t) = 1 - CDF(2*n + 2 - t, t).

    tMax : int
        Maximum time that can be iterated to. This sets the allocated size of the CDF.

    id : int
        System ID used to save the system state.

    save_dir : str
        Directory to save the system state to.

    Methods
    -------
    getxvals()
        Returns positions of CDF using the relation: x = 2*n + 2 - t.

    setBetaSeed(seed)
        Set the random seed of the beta distribution.

    iterateTimeStep()
        Evolve the system forward one step in time.

    evolveToTime(time)
        Evolve the system to a time.

    evolveTimesteps(num)
        Evolve the system forwared a number of timesteps.

    findQuantile(quantile)
        Find the corresponding quantile position.

    findQuantiles(quantiles, descending=False)
        Find the corresponding quantiles.

    getGumbelVariance(nParticles)
        Get the gumbel variance from the CDF.

    getProbandV(quantile)
        Get the probability and velocity of a quantile.

    saveState()
        Saves the current state of the system to a file.

    fromFiles(cdf_file, scalars_file)
        Load a DiffusionTimeCDF object from saved files.

    evolveAndGetVariance(times, nParticles, file)
        Get the gumbel variance at specific times and save to file.

    evolveAndSaveQuantile(times, quantiles, file)
        Evolve the system to specific times and save the quantiles at those times
        to a file.

    evolveAndGetProbAndV(quantile, time, save_file)
        Measure the probability and velocity of a quantile at different times.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_saved_time = time.process_time()  # seconds
        self._save_interval = 3600 * 6  # Set to save occupancy every 6 hours.
        self.id = None
        self.save_dir = "."

    def __str__(self):
        return f"DiffusionTimeCDF(beta={self.beta}, time={self.time})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, DiffusionTimeCDF):
            raise TypeError(
                f"Comparison must be between same object types, but other of type {type(other)}"
            )

        if (
            self.id == other.id
            and self.save_dir == other.save_dir
            and self.beta == other.beta
            and self.time == other.time
            and np.all(self.CDF == other.CDF)
            and self.tMax == other.tMax
        ):
            return True
        return False

    @property
    def time(self):
        return self.getTime()

    @time.setter
    def time(self, time):
        self.setTime(time)

    @property
    def beta(self):
        return self.getBeta()

    @beta.setter
    def beta(self, beta):
        self.setBeta(beta)

    @property
    def CDF(self):
        return self.getCDF()

    @CDF.setter
    def CDF(self, CDF):
        self.setCDF(CDF)

    @property
    def tMax(self):
        return self.gettMax()

    @tMax.setter
    def tMax(self, tMax):
        self.settMax(tMax)

    def xvals(self):
        """
        Returns positions of CDF using the relation: x = 2*n + 2 - t.

        Returns
        -------
        list :
            Positions of the CDF.
        """
        return super().getxvals()

    def setBetaSeed(self, seed):
        """
        Set the random seed of the beta distribution. Primarily
        for reproducability of results.

        Parameters
        ----------
        seed : int
            Random seed to use.
        """

        super().setBetaSeed(seed)

    def iterateTimeStep(self):
        """
        Evolve the recurrance relation forward one step in time. Note this method
        checks if the object has been save recently. If not, the state of the object
        is saved in case the process is terminated early.

        Raises
        ------
        ValueError
            If trying to evolve the system to a time greater than the allocated
            time. This would normally Core Dump since trying to allocate memory
            outside array.
        """
        if (time.process_time() - self._last_saved_time) > self._save_interval:
            self.saveState()
            self._last_saved_time = time.process_time()

        if self.time >= self.tMax:
            raise ValueError(f"Cannot evolve to time greater than tMax: {self.tMax}")

        super().iterateTimeStep()

    def evolveToTime(self, time):
        """
        Evolve the system to a time t.

        Parameters
        ----------
        time : int
            Time to iterate the system forward to.
        """

        while self.time < time:
            self.iterateTimeStep()

    def evolveTimesteps(self, num):
        """
        Evolve the system forward a number of timesteps.

        Parameters
        ----------
        num : int
            Number of timesteps to evolve the system
        """

        for _ in range(num):
            self.iterateTimeStep()

    def findQuantile(self, quantile):
        """
        Find the corresponding quantile position.

        Parameters
        ----------
        quantile : np.quad
            Quantile to measure. Should be > 1.

        Returns
        -------
        float
            Position of the quantile
        """

        return super().findQuantile(quantile)

    def findQuantiles(self, quantiles, descending=False):
        """
        Find the corresponding quantiles. Should be faster than a list compression
        over findQuntile b/c it does it in one loop in C++.

        Parameters
        ----------
        quantiles : numpy array (dtype np.quad)
            Quantiles to measure. Should all be > 1.

        descending : bool
            Whether or not the incoming quantiles are in descending or ascending order.
            If they are not in descending order we flip the output quantiles.

        Returns
        -------
        numpy array (dtype ints)
            Position of the quantiles.
        """

        if descending:
            return np.array(super().findQuantiles(quantiles))
        else:
            returnVals = super().findQuantiles(quantiles)
            returnVals.reverse()
            return np.array(returnVals)

    def getGumbelVariance(self, nParticles):
        """
        Get the gumbel variance from the CDF by sampling the CDF.

        Parameters
        ----------
        nParticles : float, np.quad or list
            Number of particles to get the gumbel variance for.

        Returns
        -------
        variance : np.quad
            Sampling variance from the CDF.
        """

        return super().getGumbelVariance(nParticles)

    def getProbandV(self, quantile):
        """
        Get the probability and velocity of a quantile.

        Parameters
        ----------
        quantile : float or np.quad
            Quantile to measure the velocity and probability for.

        Returns
        -------
        prob : np.quad
            Probability at the position

        v : float
            Velocity at the position. Should be between 0 and 1.
        """

        return super().getProbandV(quantile)

    def saveState(self):
        """
        Save all the simulation constants to a scalars file and the occupancy
        to a seperate file.

        Note
        ----
        Must have defined the ID attribute for this to work properly.
        The scalars are saved to a file Scalars{id}.json and the occupancy
        is saved to Occupancy{id}.txt.
        """

        cdf_file = os.path.join(self.save_dir, f"CDF{self.id}.txt")
        scalars_file = os.path.join(self.save_dir, f"Scalars{self.id}.json")

        fileIO.saveArrayQuad(cdf_file, self.getSaveCDF())

        vars = {
            "time": self.time,
            "beta": self.beta,
            "tMax": self.tMax,
            "id": self.id,
            "save_dir": self.save_dir,
        }

        with open(scalars_file, "w+") as f:
            json.dump(vars, f)

    @classmethod
    def fromFiles(cls, cdf_file, scalars_file):
        """
        Load a DiffusionTimeCDF object from saved files.

        Parameters
        ----------
        cdf_file : str
            File that contains the CDF of the system

        scalars_file : str
            File that contains system parameters

        Returns
        -------
        DiffusionTimeCDF
            Object loaded from file. Should be equivalent to the saved object.
        """

        with open(scalars_file, "r") as file:
            vars = json.load(file)

        load_cdf = fileIO.loadArrayQuad(cdf_file)
        cdf = np.zeros(vars["tMax"] + 1, dtype=np.quad)
        cdf[: vars["time"] + 1] = load_cdf

        d = DiffusionTimeCDF(beta=vars["beta"], tMax=vars["tMax"])
        d.time = vars["time"]
        d.id = vars["id"]
        d.save_dir = vars["save_dir"]
        d.CDF = cdf
        return d

    def evolveAndGetVariance(self, times, nParticles, file, append=False):
        """
        Get the gumbel variance at specific times and save to file.

        Parameters
        ----------
        times : numpy array or list
            Times to evolve the system to and save quantiles at.

        nParticles : list
            Number of particles to record quantile and variance for.

        file : str
            Destination to save the data to.

        append : bool (False)
            Whether or not to append to a file. Primarily whether to write the
            header or not.
        """
        f = open(file, "a")
        writer = csv.writer(f)

        if not append:
            header = (
                ["time"]
                + [str(N) for N in nParticles]
                + ["var" + str(N) for N in nParticles]
            )
            writer.writerow(header)
            f.flush()

        for t in times:
            self.evolveToTime(t)
            discrete = self.getGumbelVariance(nParticles)
            quantiles = self.findQuantiles(nParticles)
            row = [self.time] + list(quantiles) + discrete
            writer.writerow(row)
            f.flush()
        f.close()

    def evolveAndSaveQuantile(self, time, quantiles, file, append=False):
        """
        Evolve the system to specific times and save the quantiles at those times
        to a file.

        Parameters
        ----------
        time : numpy array or list
            Times to evolve the system to and save quantiles at

        quantiles : numpy array (dtype np.quad)
            Quantiles to save at each time

        file : str
            File to save the quantiles to.

        append : bool (False)
            Whether or not to append to a file. Primarily whether to write the
            header or not.
        """

        f = open(file, "a")
        writer = csv.writer(f)

        if not append:
            header = ["time"] + [str(q) for q in quantiles]
            writer.writerow(header)

        for t in time:
            self.evolveToTime(t)

            quantiles = list(np.array(quantiles))
            quantiles.sort()  # Need to get the quantiles in descending order
            quantiles.reverse()
            NthQuantiles = self.findQuantiles(quantiles)

            row = [self.time] + list(NthQuantiles)
            writer.writerow(row)
        f.close()

    def evolveAndGetProbAndV(self, quantile, time, save_file):
        """
        Measure the probability and velocity of a quantile at different times.

        Parameters
        ----------
        quantile : float or np.quad
            Quantile to measure the velocity and probability for. The algorithm
            looks for the position where the probability is greater than 1/quantile.

        time : list
            Times to measure the quantile at

        file : str
            File to save the data to
        """

        assert quantile >= 1

        f = open(save_file, "a")
        writer = csv.writer(f)

        header = ["time", "prob", "v"]
        writer.writerow(header)

        for t in time:
            self.evolveToTime(t)

            prob, v = self.getProbandV(quantile)
            row = [self.time, prob, v]
            writer.writerow(row)
            f.flush()

        f.close()
