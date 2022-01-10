"""Agent-base pandemic simulator in Python.

Based on an original idea by Georg Wohlfahrt.
"""

__version__ = '0.2'

import time
from collections import Counter
import numpy as np
import xarray as xr
from numpy.random import default_rng
import matplotlib.pyplot as plt


class GameMaster:
    """The GameMaster class handles all things related to luck, fate and politics.

    A little bit like God, if you like. There is only one instance per game.
    """

    pos_when_out = (-1, -1)

    def __init__(self, t0=0, seed=None, npix=100):
        """Instantiate the object.

        Parameters
        ----------
        t0 : int
            initial game time (should be 0 in most case)
        seed : int
            if you want to repeat random simulation results, you'll want
            to set a seed.
        npix : int
            domain size.
        """
        self.t = t0
        self.rng = default_rng(seed=seed)
        self.npix = npix

    def get_random_pos(self):
        """Get a random (x, y) position in the simulation domain.

        Returns
        -------
        tuple of (x, y) positions.
        """
        # npix is excluded, so the range of values is [0, npix[
        return tuple(self.rng.integers(low=0, high=self.npix, size=2))

    def get_random_move(self, movemax):
        """Get a random agent move in the x and y direction.

        The agent is in charge of not walking out of the domain.

        Parameters
        ----------
        movemax : int
            the maximum number of pixels that the agent wants to travel.

        Returns
        -------
        [mx, my] a random move.
        """
        return self.rng.integers(low=-movemax, high=movemax, size=2, endpoint=True)

    def get_random_integer(self, low, high=None, size=None):
        """Get random integer(s) between bounds.

        Parameters
        ----------
        low : int
            Lowest (signed) integers to be drawn from the distribution
            (unless high=None, in which case this parameter is 0 and this value
            is used for high).
        high : int
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if high=None).
            If array-like, must contain integer values
        size : int or tuple of ints
            Output shape. If the given shape is, e.g., (m, n, k), then
            m * n * k samples are drawn. Default is None, in which case a
            single value is returned.

        Returns
        -------
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
        """
        return self.rng.integers(low, high=high, size=size)

    def get_random_float(self, size=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Parameters
        ----------
        size : int or tuple of ints
            Output shape. If the given shape is, e.g., (m, n, k), then
            m * n * k samples are drawn. Default is None, in which case a
            single value is returned.

        Returns
        -------
        Array of random floats of shape size (unless size=None, in which case
        a single float is returned).
        """
        return self.rng.random(size=size)


class NaiveAgent:
    """The Agent moves around and eventually gets sick or immune.

    It all depends on the GameMaster.
    """

    def __init__(self,
                 gamemaster,
                 movemax=5,
                 time_until_recovery=10,
                 time_hospitalised=10,
                 hospitalisation_rate=0.1,
                 mortality_rate=0.1,
                 probability_of_infection=0.5,
                 ):

        self.gamemaster = gamemaster
        self.movemax = movemax
        self.immune = False
        self.infected = False
        self.deceased = False
        self.quarantined = False
        self.hospitalised = False
        self.time_of_infection = None
        self.probability_of_infection = probability_of_infection
        self.time_until_recovery = time_until_recovery
        self.time_hospitalised = time_hospitalised
        self.hospitalisation_rate = hospitalisation_rate
        self.mortality_rate = mortality_rate
        self.pos = self.gamemaster.get_random_pos()
        self.agent_time = gamemaster.t

    def move(self):
        """We move in the domain

        Returns
        -------
        the (x, y) position.
        """

        if self.deceased or self.quarantined or self.hospitalised:
            # Dead agents are out
            return self.gamemaster.pos_when_out

        # Tell me where to go
        mx, my = self.gamemaster.get_random_move(self.movemax)
        # Abs is to reflect on the lower bound
        new_x = abs(self.pos[0] + mx)
        new_y = abs(self.pos[1] + my)
        # Reflect also on the higher bound
        if new_x >= self.gamemaster.npix:
            new_x = 2 * self.gamemaster.npix - new_x - 2
        if new_y >= self.gamemaster.npix:
            new_y = 2 * self.gamemaster.npix - new_y - 2
        self.pos = (new_x, new_y)
        return self.pos

    def contact(self):
        """Agent in contact with an infectious person. Decide what happens."""

        if self.immune:
            # Nothing happens
            return

        # Let fate decide
        v = self.gamemaster.get_random_float()
        if v < self.probability_of_infection:
            self.infected = True
            self.immune = True
            self.time_of_infection = self.gamemaster.t

    def _decide_quarantine(self):
        """We are naive we dont quarantine"""
        pass

    def _decide_hospitalisation_or_death(self):

        if not self.infected:
            # Nothing to do here
            return

        # Maybe bad things happen
        tor = self.time_of_infection + self.time_until_recovery
        if self.agent_time >= tor and not self.hospitalised:
            # Decide on hospitalisation or recovery
            v = self.gamemaster.get_random_float()
            if v < self.hospitalisation_rate:
                self.hospitalised = True
            else:
                # We recovered
                self.infected = False
                self.hospitalised = False
                self.quarantined = False

        torh = tor + self.time_hospitalised
        if self.agent_time >= torh and self.hospitalised:
            # Decide on death or recovery
            v = self.gamemaster.get_random_float()
            if v < self.mortality_rate:
                self.deceased = True
                self.infected = False
                self.hospitalised = False
                self.quarantined = False
            else:
                self.infected = False
                self.hospitalised = False
                self.quarantined = False

    def _decide_status(self):
        """Make the agent decide on its statut.

        This method needs to be called at least once per simulation day.
        """

        if self.agent_time == self.gamemaster.t:
            # We already decided what our state is
            return

        # New time
        self.agent_time = self.gamemaster.t

        self._decide_quarantine()
        self._decide_hospitalisation_or_death()

    @property
    def contagious(self):
        """Is the agent contagious? This has to be called once per turn!
        """

        self._decide_status()

        if not self.infected or self.deceased or self.quarantined or self.hospitalised:
            # Nope
            return False

        return self.infected


def game(seed=None,  # Repeat random results?
         n_agents=10000,  # Number of agents in the domain
         agent_class=NaiveAgent,  # The type of agent to use
         npix=100,  # Domain size
         nt=100,  # number time steps (days) in the simulation
         stop_when_extinct=False,  # stop the simulation if the virus is extinct
         n_initial_infected=5,  # Number of initially infected agents
         movemax=5,  # Agents max movements
         time_until_recovery=9,  # Agents days until recovery after infection
         time_hospitalised=9,  # Agents days until recovery after hospitalisation
         hospitalisation_rate=0.1,  # Hospitalisation rate at the end of the infection
         mortality_rate=0.1,  # Mortality rate at the end of the hospitalisation
         probability_of_infection=0.5,  # Probability of being infected if one other agent is infected
         log=True,  # Print simulation log on screen - if False, only log last step
         ):
    """Run a pandemic simulation.

    Returns
    -------
    the simulation results as an xarray dataset
    """
    # Timer
    start_time = time.time()

    # Prepare the players
    gamemaster = GameMaster(seed=seed, npix=npix)
    agents = []
    for _ in range(n_agents):
        agents.append(agent_class(gamemaster,
                                  movemax=movemax,
                                  time_until_recovery=time_until_recovery,
                                  time_hospitalised=time_hospitalised,
                                  hospitalisation_rate=hospitalisation_rate,
                                  mortality_rate=mortality_rate,
                                  probability_of_infection=probability_of_infection)
                      )

    # Fate: who is infected
    for idx in gamemaster.get_random_integer(0, n_agents-1, n_initial_infected):
        agents[idx].infected = True
        agents[idx].immune = True
        agents[idx].time_of_infection = gamemaster.t

    # Data containers (its faster to create them only once before the game)
    n_infected = np.zeros((nt + 1,), dtype=int)
    n_deceased = np.zeros((nt + 1,), dtype=int)
    n_immune = np.zeros((nt + 1,), dtype=int)
    n_vulnerable = np.zeros((nt + 1,), dtype=int)
    n_hospitalised = np.zeros((nt + 1,), dtype=int)
    n_quarantined = np.zeros((nt + 1,), dtype=int)

    is_infected = np.zeros((nt + 1, n_agents), dtype=bool)
    is_deceased = np.zeros((nt + 1, n_agents), dtype=bool)
    is_immune = np.zeros((nt + 1, n_agents), dtype=bool)
    is_hospitalised = np.zeros((nt + 1, n_agents), dtype=bool)
    is_quarantined = np.zeros((nt + 1, n_agents), dtype=bool)

    agent_density = np.zeros((nt + 1, npix, npix), dtype=np.int32)
    contagious_density = np.zeros((nt + 1, npix, npix), dtype=np.int32)
    immune_density = np.zeros((nt + 1, npix, npix), dtype=np.int32)

    # Add the data containers to an xarray dataset
    ds = xr.Dataset()
    ds['time'] = (('time', ), np.arange(nt + 1, dtype=int))
    ds['agents'] = (('agents', ), np.arange(n_agents, dtype=int))
    ds['x'] = (('x', ), np.arange(npix, dtype=int))
    ds['y'] = (('y', ), np.arange(npix, dtype=int))

    ds['n_infected'] = (('time', ), n_infected)
    ds['n_immune'] = (('time', ), n_immune)
    ds['n_vulnerable'] = (('time', ), n_vulnerable)
    ds['n_hospitalised'] = (('time', ), n_hospitalised)
    ds['n_quarantined'] = (('time', ), n_quarantined)
    ds['n_deceased'] = (('time', ), n_deceased)

    ds['is_infected'] = (('time', 'agents'), is_infected)
    ds['is_deceased'] = (('time', 'agents'), is_deceased)
    ds['is_immune'] = (('time', 'agents'), is_immune)
    ds['is_hospitalised'] = (('time', 'agents'), is_hospitalised)
    ds['is_quarantined'] = (('time', 'agents'), is_quarantined)

    ds['agent_density'] = (('time', 'y', 'x'), agent_density)
    ds['contagious_density'] = (('time', 'y', 'x'), contagious_density)
    ds['immune_density'] = (('time', 'y', 'x'), immune_density)

    # Lets go
    while gamemaster.t <= nt:

        pos_contagious = []
        for k, a in enumerate(agents):
            p = a.move()
            cont, imm = a.contagious, a.immune
            out = a.quarantined or a.hospitalised or a.deceased
            if cont:
                pos_contagious.append(p)
                contagious_density[gamemaster.t, p[1], p[0]] += 1
            if not out:
                agent_density[gamemaster.t, p[1], p[0]] += 1
                if imm:
                    immune_density[gamemaster.t, p[1], p[0]] += 1

            is_infected[gamemaster.t, k] = a.infected
            is_immune[gamemaster.t, k] = a.immune
            is_quarantined[gamemaster.t, k] = a.quarantined
            is_hospitalised[gamemaster.t, k] = a.hospitalised
            is_deceased[gamemaster.t, k] = a.deceased

        pos_contagious = Counter(pos_contagious)
        for a in agents:
            if a.pos in pos_contagious:
                # The probability to get infected is higher if there are more
                # agents infected in one cell
                for _ in range(pos_contagious[a.pos]):
                    a.contact()

        n_i = is_immune[gamemaster.t, :].sum()
        n_d = is_deceased[gamemaster.t, :].sum()
        n_infected[gamemaster.t] = is_infected[gamemaster.t, :].sum()
        n_immune[gamemaster.t] = n_i
        n_vulnerable[gamemaster.t] = n_agents - n_i - n_d
        n_quarantined[gamemaster.t] = is_quarantined[gamemaster.t, :].sum()
        n_hospitalised[gamemaster.t] = is_hospitalised[gamemaster.t, :].sum()
        n_deceased[gamemaster.t] = n_d

        if log:
            print(f"Day {gamemaster.t:3d}. "
                  f"N infected: {n_infected[gamemaster.t]:5d}. "
                  f"N vulnerable: {n_vulnerable[gamemaster.t]:5d}. "
                  f"N immune: {n_immune[gamemaster.t]:5d}. "
                  f"N quarantined: {n_quarantined[gamemaster.t]:5d}. "
                  f"N hospitalised: {n_hospitalised[gamemaster.t]:5d}. "
                  f"N deceased: {n_deceased[gamemaster.t]:5d}. ")

        if stop_when_extinct and n_infected[gamemaster.t] == 0:
            # Nothing left to compute
            break
        gamemaster.t += 1

    total_time = time.time() - start_time
    print(f"Simulation done. Total time: {total_time:.1f}s")

    return ds


if __name__ == '__main__':

    # Demo game without measures and with measures
    gm1 = game(seed=0)
    gm2 = game(seed=0, movemax=2)

    plt.plot(gm1.n_infected, color='C0', label='Infected')
    plt.plot(gm2.n_infected, color='C0', linestyle='--')
    plt.plot(gm1.n_immune, color='C1', label='Immune')
    plt.plot(gm2.n_immune, color='C1', linestyle='--')
    plt.plot(gm1.n_hospitalised, color='C2', label='Hospitalised')
    plt.plot(gm2.n_hospitalised, color='C2', linestyle='--')
    plt.plot(gm1.n_deceased, color='C3', label='Deceased (cum)')
    plt.plot(gm2.n_deceased, color='C3', linestyle='--')
    plt.legend()
    plt.show()
