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
    Create a class that iterates through the time of the CDF. Can also be used to
    get the discrete variance.

    Parameters
    ----------
    tMax : int
        Maximum time that will be iterated to.

    beta : float
        Value of beta used in the recurrance relation.

    Attributes
    ----------
    CDF : numpy array (dtype of np.quad)
        The current recurrance vector Z_B(n, t). This is actually
        Z_B(n, t) = 1 - CDF(n, t).

    time : int
        Current time of the system

    id : int
        System ID used to save the system state.

    save_dir : str
        Directory to save the system state to.

    Methods
    -------
    setBetaSeed(seed)
        Set the random seed of the beta distribution.

    iterateTimeStep()
        Evolve the system forward one step in time.

    evolveToTime(time)
        Evolve the system to a time.

    saveState()
        Saves the current state of the system to a file.

    evolveTimesteps(num)
        Evolve the system forward a number of timesteps.

    findQuantile(quantile)
        Find the corresponding quantile position.

    findQuantiles(quantiles, descending=False)
        Find the corresponding quantiles.

    getGumbelVariance(nParticles)
        Get the gumbel variance from the CDF.

    getGumbelVariancePDF(nParticles)
        Get gumbel variance from the CDF by first calculating the PDF.

    evolveAndGetVariance(times, nParticles, file)
        Get the gumbel variance at specific times and save to file.

    evolveAndSaveQuantile(times, quantiles, file)
        Evolve the system to specific times and save the quantiles at those times
        to a file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_saved_time = time.process_time()  # seconds
        self._save_interval = 3600 * 6  # Set to save occupancy every XX hours.
        self.id = None  # Need to also get SLURM ID
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

    def setBetaSeed(self, seed):
        """
        Set the random seed of the beta distribution.

        Parameters
        ----------
        seed : int(?)
            Random seed to use.
        """

        super().setBetaSeed(seed)

    def iterateTimeStep(self):
        """
        Evolve the recurrance relation zB forward one step in time.

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
            Time to iterate the system forward to
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
            Quantile to measure

        Returns
        -------
        float
            Position of the quantile
        """

        return super().findQuantile(quantile)

    def findQuantiles(self, quantiles, descending=False):
        """
        Find the corresponding quantiles. Should be faster than a list compression
        over findQuntile b/c it does it in one loop.

        Parameters
        ----------
        quantiles : numpy array (dtype np.quad)
            Nth quantiles to measure

        descending : bool
            Whether or not the incoming quantiles are in descending or ascending order.
            If they are not in descending order we flip the output quantiles.

        Returns
        -------
        numpy array (dtype ints)
            Position of Nth quantiles
        """

        if descending:
            return np.array(super().findQuantiles(quantiles))
        else:
            returnVals = super().findQuantiles(quantiles)
            returnVals.reverse()
            return np.array(returnVals)

    def findLowerQuantile(self, quantile):
        """
        Find a quantile in the lower part of the distribution.
        Opposite the findQuantile method
        """

        return super().findLowerQuantile(quantile)

    def getGumbelVariance(self, nParticles):
        """
        Get the gumbel variance from the CDF.

        Parameters
        ----------
        nParticles : float, np.quad or list
            Number of particles to get the gumbel variance for.

        Returns
        -------
        variance : np.quad
            Variance for the number of particles
        """

        return super().getGumbelVariance(nParticles)

    def getProbandV(self, quantile):
        """
        Get the probability and velocity of a quantile.

        Parameters
        ----------
        quantile : float or np.quad
            Quantile to measure the velocity and probability for. The algorithm
            looks for the position where the probability is greater than 1/quantile.

        Returns
        -------
        prob : np.quad
            Probability at the position

        v : float
            Velocity at the position. Should be between 0 and 1
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
        """
        # Need to make sure changing to "a" doesn't break when writing to an
        # empty or non-existant file
        f = open(file, "a")
        writer = csv.writer(f)

        if not append:
            header = ["time"] + [str(N) for N in nParticles] + ['var' + str(N) for N in nParticles]
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

        Examples
        --------
        >>> r = Recurrance(beta=np.inf)
        >>> r.evolveAndSaveQuantile([1, 5, 50], [5, 10], 'Data.txt')
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
        Measure the probability of a quantile at different times.

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

        f = open(save_file, 'a')
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

    def evolveAndSaveFirstPassage(self, quantile, distances, save_file):
        """
        Measure the first passage time of a quantile at various distances.
        """

        f = open(save_file, 'a')
        writer = csv.writer(f)

        header = ['Distance', 'Time']
        writer.writerow(header)
        f.flush()

        idx = 0
        while idx < len(distances):
            self.iterateTimeStep()
            upper_dist = self.findQuantile(quantile)
            lower_dist = abs(self.findLowerQuantile(quantile))

            if upper_dist >= distances[idx] or lower_dist >= distances[idx]:
                row = [distances[idx], self.time]
                writer.writerow(row)
                f.flush()
            idx += 1


class DiffusionPositionCDF(diffusionCDF.DiffusionPositionCDF):
    """
    Class to iterate through the position of the CDF.
    """

    def __str__(self):
        return f"DiffusionCDF(beta={self.beta}, time={self.position})"

    def __repr__(self):
        return self.__str__()

    @property
    def beta(self):
        return self.getBeta()

    @property
    def CDF(self):
        return self.getCDF()

    @property
    def tMax(self):
        return self.gettMax()

    @property
    def setBetaSeed(self, seed):
        super().setBetaSeed(seed)

    @property
    def position(self):
        return self.getPosition()

    @property
    def quantilePositions(self):
        return self.getQuantilesMeasurement()

    @property
    def quantiles(self):
        return self.getQuantiles()

    def stepPosition(self):
        """
        Move the system forward one step in the position.
        """

        if self.position == self.tMax:
            raise ValueError("Cannot evolve the system to a position larger than tMax")

        super().stepPosition()

    def evolveToPosition(self, n):
        """
        Move the system forward to a specified position.

        Parameters
        ----------
        n : int
            Position to move the system forward to.
        """

        while self.position < n:
            self.stepPosition()

    def evolvePositions(self, num_positions):
        """
        Move the system forward a fixed number of positions.
        """

        for _ in range(num_positions):
            self.stepPosition()

if __name__ == '__main__':
    r = DiffusionTimeCDF(1, tMax=10000)
    r.evolveAndGetProbAndV(100, [10, 100, 1000], 'data.txt')
