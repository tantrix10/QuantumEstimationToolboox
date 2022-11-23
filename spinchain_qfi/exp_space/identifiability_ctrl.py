# -*- coding: utf-8 -*-
"""
@author: Jukka Kiukas
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

Classes for the identifiability of an unknown parameter in the dynamics
generator.

"""

import numpy as np
import scipy.linalg as la

import timeit

from qutip import sigmax, sigmay, sigmaz
# QuTiP control modules
# import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
# import qutip.control.termcond as termcond
# import qutip.control.optimizer as optimizer
# import qutip.control.stats as stats
import qutip.control.errors as errors
import qutip.control.tslotcomp as tslotcomp
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.logging_utils as logging
logger = logging.get_logger()

# pauli = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
pauli = [sigmax(), sigmay(), sigmaz()]


def dot(bl1, bl2):
    """
    Computes the scalar product of two bloch vectors
    """
    res = 0
    for l in range(3):
        res += bl1[l] * bl2[l]
    return res


def cutoff(num):
    if num < 1e15:
        return num
    else:
        print("value cutoff!")
        return 0
# import qutip.control.pulsegen as pulsegen


class DynamicsSysIdUnitary(dynamics.Dynamics):
    """
    This is the subclass to use for systems with dynamics described by
    unitary matrices, E.g. closed systems with Hermitian Hamiltonians.
    In addition to the drift Hamiltonian and controls, the class contains an
    additional Hamiltonian describing a direction in the set of Hermitian
    operators.

    Attributes
    ----------

    drift_dir : Qobj
        This is the direction Hamiltonian

    _drift_dir : ndArray
        This is the direction Hamiltonian in internal type

    _evo_interval : array[num_tslots x num_tslots] of Qobj
        Evolution from timeslot k to timeslot m

    _prop_der : array[num_tslots] of Qobj
        Propagator derivative in the direction drift_dir. Array of matrices
        that give the directional derivative in direction drift_dir of the
        propagators. Note this attribute is only created when the propagator
        computer is of exact gradient type.

    _prop_hessian : array[num_tslots, num_ctrls]  of Qobj
        Propagator derivative with respect to both the direction and the
        control at a given time slot.
    """

    def reset(self):
        dynamics.Dynamics.reset(self)
        self.id_text = 'UNIT_DIR'
        self._evo_interval = None
        self.drift_dir = None
        self._drift_dir = None
        self._prop_der = None
        self._prop_hessian = None
        self.fid_scale = None
        self.mixed_qfi = False
        self._dyn_gen_phase = -1j

    def _init_evo(self):
        """
        Create the container lists / arrays for the
        dynamics generations, propagators, and evolutions etc
        Set the time slices and cumulative time
        """
        dynamics.Dynamics._init_evo(self)
        # Change data type for drift direction
        self._drift_dir = self.drift_dir.full()
        # Create containers for the new derivatives
        shp = self.dyn_shape
        # set to be just empty float arrays with the the shape of the generator
        self._prop_der = [
            np.empty(shp, dtype=complex) for x in range(self.num_tslots)
            ]
        if self.prop_computer.grad_exact:
            self._prop_hessian = np.empty(
                [self.num_tslots, self.num_ctrls], dtype=np.ndarray
                )
            self._evo_interval = np.empty(
                [self.num_tslots, self.num_tslots+1], dtype=np.ndarray
                )

    def _create_computers(self):
        """
        Create the appropriate timeslot, fidelity and propagator computers
        """
        self.tslot_computer = TSlotCompSysId(self)
        self.fid_computer = FidCompQubitQFI(self)
        self.prop_computer = PropCompSysId(self)

    def _get_phased_dir_gen(self):
        """
        Get the direction Hamiltonian
        including the -i factor
        """
        return self._apply_phase(self._drift_dir)

    def _apply_phase(self, dg):
        """
        Include the -i factor
        """
        return -1j * dg


class TSlotCompSysId(tslotcomp.TSlotCompUpdateAll):
    """
    This is the timeslot computer to be used with Dynamics containing the
    direction Hamiltonian.
    """

    def reset(self):
        tslotcomp.TSlotCompUpdateAll.reset(self)
        self.id_text = 'TSLOTCOMP_DIR'

    def log_dyn_obj(self, k, j=None, prop=None, prop_der=None, prop_grad=None,
                    prop_hessian=None):
        if self.log_level <= logging.DEBUG_INTENSE:
            if prop is not None:
                logger.log(
                    logging.DEBUG_INTENSE,
                    "propagator at tslot {}:\n{}".format(k, prop)
                    )
            if prop_der is not None:
                logger.log(
                    logging.DEBUG_INTENSE,
                    "prop directional der at {}:\n{}".format(k, prop_der)
                )
            if prop_grad is not None:
                logger.log(
                    logging.DEBUG_INTENSE,
                    "prop derivative at {} for ctrl {}:\n{}".format(
                        k, j, prop_grad)
                )
            if prop_hessian is not None:
                logger.log(
                    logging.DEBUG_INTENSE,
                    "prop hessian at {} for control {}:\n{}".format(
                        k, j, prop_hessian)
                )

    def recompute_evolution(self):
        """
        Recalculates the evolution operators.
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary.
        """
        dyn = self.parent
        prop_comp = dyn.prop_computer
        n_ts = dyn.num_tslots
        n_ctrls = dyn.num_ctrls

        # Clear the public lists
        # These are only set if (external) users access them
        dyn._dyn_gen_qobj = None
        dyn._prop_qobj = None
        dyn._prop_grad_qobj = None
        dyn._fwd_evo_qobj = None
        dyn._onwd_evo_qobj = None
        dyn._onto_evo_qobj = None

        if dyn.stats is not None:
            dyn.stats.num_tslot_recompute += 1
            if self.log_level <= logging.DEBUG:
                logger.log(logging.DEBUG, "recomputing evolution {} ".format(
                               dyn.stats.num_tslot_recompute))
        # calculate the Hamiltonians
        time_start = timeit.default_timer()
        for k in range(n_ts):
            dyn._combine_dyn_gen(k)
        if dyn.stats is not None:
            dyn.stats.wall_time_dyn_gen_compute += \
                timeit.default_timer() - time_start
        # calculate the propagators, propagator gradients, derivatives and
        # hessians
        time_start = timeit.default_timer()
        for k in range(n_ts):
            if prop_comp.grad_exact:  # only compute the intervals with exact
                for j in range(n_ctrls):
                    if j == 0:  # first ctrl compute all j-independent ders
                        p, g, d, h = prop_comp._compute_prop_2nd_der(
                            k, 0,
                            compute_prop=True, compute_drift_frechet=True
                            )
                        dyn._prop[k] = p
                        dyn._prop_der[k] = d
                        dyn._prop_grad[k, 0] = g
                        dyn._prop_hessian[k, 0] = h
                        self.log_dyn_obj(
                            k, j=0, prop=p, prop_der=d,
                            prop_grad=g, prop_hessian=h
                            )
                    else:
                        # compute here all ctrl derivatives
                        g, h = prop_comp._compute_prop_2nd_der(
                            k, j=j, compute_prop=False,
                            compute_drift_frechet=False
                            )
                        dyn._prop_grad[k, j] = g
                        dyn._prop_hessian[k, j] = h
                        self.log_dyn_obj(k, j=j, prop_grad=g, prop_hessian=h)
            else:  # approx grad in use
                p, d = prop_comp._compute_prop_der(k, compute_prop=True)
                dyn._prop[k] = p
                dyn._prop_der[k] = d
                self.log_dyn_obj(k, prop=p, prop_der=d)
        if dyn.stats is not None:
            dyn.stats.wall_time_prop_compute += \
                timeit.default_timer() - time_start
        # compute the forward propagation and interval if exact gradients
        time_start = timeit.default_timer()
        if prop_comp.grad_exact:
            for k in range(n_ts):  # starting tslot index, inclusive
                dyn._evo_interval[k, k] = np.identity(dyn.get_drift_dim())
                for m in range(k+1, n_ts+1):  # endpoint tslot index, exclusive
                    dyn._evo_interval[k, m] = dyn._prop[m-1].dot(
                        dyn._evo_interval[k, m-1])
                    if k == 0:
                        dyn._fwd_evo[m] = dyn._prop[m-1].dot(
                                dyn._fwd_evo[m-1]  # endtslot excl init state
                            )
                dyn._onwd_evo[k] = dyn._evo_interval[k, n_ts]
        else:
            for m in range(1, n_ts+1):  # end tslot excl (1 - n_ts) init state
                dyn._fwd_evo[m] = dyn._prop[m-1].dot(dyn._fwd_evo[m-1])
            dyn._onwd_evo[n_ts - 1] = dyn._prop[n_ts - 1]
            for k in range(n_ts-2, -1, -1):  # starting tslot incl (0 - n_ts-1)
                dyn._onwd_evo[k] = dyn._onwd_evo[k+1].dot(dyn._prop[k])
        if dyn.stats is not None:
            dyn.stats.wall_time_fwd_prop_compute += \
                timeit.default_timer() - time_start


class PropCompSysId(propcomp.PropagatorComputer):
    """
    Class for computing propagators and relevant derivatives, including
    additional methods for the double derivative with respect to both
    a control and the identifiable parameter direction.
    """

    def reset(self):
        self.id_text = 'SysIdPropComp'
        self.log_level = self.parent.log_level
        self.apply_params()
        self.grad_exact = True

    def _compute_propagator(self, k):
        """
        calculate the progator between X(k) and X(k+1) using matrix exponential
        Assumes that the dyn_gen have been been calculated, i.e. drift and
        ctrls combined. Returns the propagator
        """
        dyn = self.parent
        dgt = dyn._get_phased_dyn_gen(k)*dyn.tau[k]
        prop = la.expm(dgt)
        return prop

    def _compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot.
        """
        dyn = self.parent
        A = dyn._get_phased_dyn_gen(k)*dyn.tau[k]
        try:
            E1 = dyn._get_phased_ctrl_dyn_gen(k, j)*dyn.tau[k]
        except TypeError:
            E1 = dyn._get_phased_ctrl_dyn_gen(j) * dyn.tau[k]
        if compute_prop:
            prop, propGrad = la.expm_frechet(A, E)
            return prop, propGrad
        else:
            propGrad = la.expm_frechet(A, E, compute_expm=False)
            return propGrad

    def _get_aug_mat_dir(self, k):
        """
        Generate the matrix [[A, E], [0, A]] where
            A is the overall dynamics generator at timeslot k
            E is the derivative direction in which the derivate is taken
        For a given timeslot returns this augmented matrix
        """
        dyn = self.parent
        A = dyn._get_phased_dyn_gen(k)*dyn.tau[k]
        E = dyn._get_phased_dir_gen()*dyn.tau[k]
        l = np.concatenate((A, np.zeros(self.parent.dyn_shape)))
        r = np.concatenate((E, A))
        aug = np.concatenate((l, r), 1)
        return aug

    def _get_large_aug_mat(self, k, j):
        """
        Generate the matrix [[A, E1, E2, 0], [0, A, 0, E2], [0, 0, A, E1],
        [0, 0, 0, A]] where
            A is the overall dynamics generator at timeslot k
            E1 is the j:th control dynamics generator at timeslot k
            E2 is the drift direction Hamiltonian (independent of timeslot)
        for a given timeslot and control
        returns this augmented matrix
        """
        dyn = self.parent
        A = dyn._get_phased_dyn_gen(k)*dyn.tau[k]
        try:
            E1 = dyn._get_phased_ctrl_dyn_gen(k, j)*dyn.tau[k]
        except TypeError:
            E1 = dyn._get_phased_ctrl_dyn_gen(j) * dyn.tau[k]

        E2 = dyn._get_phased_dir_gen()*dyn.tau[k]
        Z = np.zeros(self.parent.dyn_shape)
        col1 = np.concatenate((A, Z, Z, Z))
        col2 = np.concatenate((E1, A, Z, Z))
        col3 = np.concatenate((E2, Z, A, Z))
        col4 = np.concatenate((Z, E2, E1, A))
        aug = np.concatenate((col1, col2, col3, col4), 1)
        return aug

    def _compute_prop_der(self, k, compute_prop=False):
        shp = self.parent.dyn_shape
        aug = self._get_aug_mat_dir(k)
        aug_exp = la.expm(aug)
        prop_der = aug_exp[:shp[0], shp[1]:]
        if compute_prop:
            prop = aug_exp[:shp[0], :shp[1]]
            return prop, prop_der
        else:
            return prop_der

    def _compute_prop_2nd_der(self, k, j, compute_prop=False,
                              compute_drift_frechet=False
                              ):
        shp = self.parent.dyn_shape
        aug = self._get_large_aug_mat(k, j)
        aug_exp = la.expm(aug)
        hessian = aug_exp[:shp[0], 3*shp[1]:4*shp[1]]
        frechet_control = aug_exp[:shp[0], shp[1]:2*shp[1]]
        if compute_prop and compute_drift_frechet:
            prop = aug_exp[:shp[0], :shp[1]]
            frechet_drift = aug_exp[:shp[0], 2*shp[1]:3*shp[1]]
            return prop, frechet_control, frechet_drift, hessian
        elif compute_drift_frechet:
            frechet_drift = aug_exp[:shp[0], 2*shp[1]:3*shp[1]]
            return frechet_control, frechet_drift, hessian
        elif compute_prop:
            prop = aug_exp[:shp[0], :shp[1]]
            return prop, frechet_control, hessian
        else:
            return frechet_control, hessian


class FidCompQubitQFI(fidcomp.FidelityComputer):
    """
    Attributes
    ----------
    evo_der : Qobj
        Directional derivative of the evolution map

    bloch : array [3]
        Contains the current Bloch vector of the observable qubit

    d_bloch: array [3]
        Directional derivative of the current Bloch vector

    fisher_info : float
        Fisher information of the direction. This determines the fidelity
    """

    def reset(self):
        fidcomp.FidelityComputer.reset(self)
        self.id_text = 'FID_COMP_SYSID'
        # self.fid_norm_func = lambda x: np.tanh(x)
        # self.grad_norm_func = lambda x: 1 / np.cosh(x)**2
        self.uses_onwd_evo = True
        self.evo_der = None
        self.bloch = None
        self.d_bloch = None
        self.mixed = True

    def init_comp(self):
        """
        initialises the computer based on the configuration of the Dynamics
        """
        # optionally implemented in subclass
        self.fid_norm_func = lambda x: 1 + x * self.parent.fid_scale
        self.grad_norm_func = lambda x: self.parent.fid_scale
        self.mixed = self.parent.mixed_qfi

    def reset_bloch(self):
        self.bloch = np.empty(3, dtype=float)  # initial bloch vector
        self.d_bloch = np.empty(3, dtype=float)

    def real_exp(self, l, state1, state2):
        """
        Returns the real part of the sandwich of an observable qubit operator
        between two vectors of the large system.
        """
        vecshape = (self.parent.get_drift_dim(), 1)
        if not (state1.shape == vecshape and state2.shape == vecshape):
            raise errors.Error("wrong array dim")
        large_sysop = self.parent.embedding(l).full()
        return np.real(state1.conj().T.dot(large_sysop.dot(state2)).trace())

    def computeBloch(self):
        """
        Computes the Bloch vector and its directional derivative.
        In the process updates self.evo_der used in the gradient computation
        """
        dyn = self.parent
        n_ts = dyn.num_tslots
        evo_final = dyn._fwd_evo[n_ts]
        #print(len(dyn.fwd_evo))
        #print("start", dyn.prop[-1], "end")
        #uni_final = dyn.prop[-1]

        # find the parameter derivative of the final state
        self.evo_der = np.zeros([self.parent.get_drift_dim(), 1])
        # print('######################')
        # print(dyn._prop_der)
        # print('######################')
        for k in range(n_ts):
            derk = dyn._prop_der[k].dot(dyn._fwd_evo[k])
            #print(dyn._prop_der[k])
            if k+1 < n_ts:
                derk = dyn._onwd_evo[k+1].dot(derk)
                #print(derk)
            self.evo_der = self.evo_der + derk
            # Compute the Bloch vector and its derivative for the obs qubit
        self.reset_bloch()
        for l in range(3):
            self.bloch[l] = self.real_exp(l, evo_final, evo_final)
            #print(evo_final)
            self.d_bloch[l] = 2 * self.real_exp(
                        l, self.evo_der, evo_final
                    )
        return self.bloch, self.d_bloch

    def computeBloch_dergrad(self, k0, j0):
        """
        Computes the derivative and second derivative of the Bloch vector
        with respect to a given control j0 at tslot k0. For exact gradient.
        Assumes evo_der is current.
        """
        dyn = self.parent
        n_ts = dyn.num_tslots
        # Final state
        evo_final = dyn._fwd_evo[n_ts]
        # The derivative of the final state with respect to the control
        evo_grad = None
        derk = dyn._prop_grad[k0, j0].dot(dyn._fwd_evo[k0])
        if k0+1 < n_ts:
            evo_grad = dyn._onwd_evo[k0+1].dot(derk)
        else:
            evo_grad = derk
        # The mixed second derivative
        dergrad = np.zeros([self.parent.get_drift_dim(), 1])
        for k in range(k0):  # index of the propagator to be differentiated
            derk = dyn._prop_der[k].dot(dyn._fwd_evo[k])
            derk = dyn._evo_interval[k+1, k0].dot(derk)
            derk = dyn._prop_grad[k0, j0].dot(derk)
            if k0+1 < n_ts:
                derk = dyn._onwd_evo[k0+1].dot(derk)
            dergrad = dergrad + derk
        for k in range(k0+1, n_ts):
            derk = dyn._prop_grad[k0, j0].dot(dyn._fwd_evo[k0])
            derk = dyn._evo_interval[k0+1, k].dot(derk)
            derk = dyn._prop_der[k].dot(derk)
            if k+1 < n_ts:
                derk = dyn._onwd_evo[k+1].dot(derk)
            dergrad = dergrad + derk
        derk = dyn._prop_hessian[k0, j0].dot(dyn._fwd_evo[k0])
        if k0+1 < n_ts:
            derk = dyn._onwd_evo[k0+1].dot(derk)
        dergrad = dergrad + derk
        # Compute the Bloch vector derivatives
        bloch_grad = np.empty(3, dtype=float)
        bloch_dergrad = np.empty(3, dtype=float)
        for l in range(3):
            bloch_grad[l] = 2 * self.real_exp(l, evo_grad, evo_final)
            bloch_dergrad[l] = 2 * (self.real_exp(
                    l, dergrad, evo_final
                ) + self.real_exp(l, self.evo_der, evo_grad))
        return bloch_grad, bloch_dergrad

    def get_fid_err(self):
        """
        Gets the absolute error in the fidelity
        """
        if not self.fidelity_current:
            dyn = self.parent
            dyn.compute_evolution()
            self.computeBloch()
            self.compute_fidelity()
            if dyn.stats is not None:
                dyn.stats.num_fidelity_computes += 1
            if self.log_level <= logging.INFO:
                c = dyn.stats.num_fidelity_computes
                logger.info("Fidelity error {}: {}".format(c, self.fid_err))
            #
            print("Fidelity error {}: {}".format(c, self.fid_err))
        return self.fid_err

    def compute_fidelity(self):
        """
        Computes the Fisher information
        """
        b = self.bloch
        db = self.d_bloch
        aux = 0
        if self.mixed:
            aux = cutoff(dot(b, db)**2 / (1 - dot(b, b)))
        self.fisher_info = dot(db, db) + aux
        self.fidelity = self.fid_norm_func(self.fisher_info)
        self.fid_err = 1 - self.fidelity
        if np.isnan(self.fid_err):
            self.fid_err = np.Inf
        self.fidelity_current = True
		
		
    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates
        (although this is not typically found to happen)
        """
        # Make sure that the fidelity and evo_der is current
        self.get_fid_err()
        # Start computing the gradient
        if not self.fid_err_grad_current:
            dyn = self.parent
            dyn.compute_evolution()
            self.compute_fid_err_grad()
            if dyn.stats is not None:
                dyn.stats.num_grad_computes += 1
            if self.log_level <= logging.DEBUG:
                logger.log(logging.DEBUG, "fidelity error gradients:\n"
                           "{}".format(self.fid_err_grad)
                           )
            if self.log_level <= logging.INFO:
                gc = dyn.stats.num_grad_computes
                logger.info("Gradient norm {}: "
                            "{} ".format(gc, self.grad_norm)
                            )
        return self.fid_err_grad

    def compute_fid_err_grad(self):
        """
        Calculate exact gradient of the fidelity error function
        wrt to each timeslot control amplitudes.
        These are returned as a (nTimeslots x n_ctrls) array. Assumes evo_der
        fisher_info, bloch and d_bloch are current.
        """
        dyn = self.parent
        n_ctrls = dyn.num_ctrls
        n_ts = dyn.num_tslots
        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls])
        time_st = timeit.default_timer()
        b = self.bloch
        db = self.d_bloch
        for j in range(n_ctrls):
            for k in range(n_ts):
                gb, dgb = self.computeBloch_dergrad(k, j)
                aux = 0
                if self.mixed:
                    aux = cutoff(dot(b, db) / (1 - dot(b, b)))
                grad[k, j] = - 2 * self.grad_norm_func(self.fisher_info) * \
                    (
                    dot(dgb, db) +
                    aux * (dot(dgb, b) + dot(db, gb) + aux * dot(gb, b))
                    )
        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st
        self.fid_err_grad = grad
        self.grad_norm = np.sqrt(np.sum(self.fid_err_grad**2))
        self.fid_err_grad_current = True
        return grad
