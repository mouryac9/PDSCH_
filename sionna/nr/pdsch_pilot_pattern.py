#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PDSCH pilot pattern for the nr (5G) sub-package of the Sionna library.
"""
import warnings
from collections.abc import Sequence
import tensorflow as tf
import numpy as np
from sionna.ofdm import PilotPattern
from .PDSCH_config import PDSCHConfig

class PDSCHPilotPattern(PilotPattern):
    # pylint: disable=line-too-long
    r"""Class defining a pilot pattern for NR PDSCH.

    This class defines a :class:`~sionna.ofdm.PilotPattern`
    that is used to configure an OFDM :class:`~sionna.ofdm.ResourceGrid`.

    For every transmitter, a separte :class:`~sionna.nr.PDSCHConfig`
    needs to be provided from which the pilot pattern will be created.

    Parameters
    ----------
    PDSCH_configs : instance or list of :class:`~sionna.nr.PDSCHConfig`
        PDSCH Configurations according to which the pilot pattern
        will created. One configuration is needed for each transmitter.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self,
                 PDSCH_configs,
                 dtype=tf.complex64):

        # Check correct type of PDSCH_configs
        if isinstance(PDSCH_configs, PDSCHConfig):
            PDSCH_configs = [PDSCH_configs]
        elif isinstance(PDSCH_configs, Sequence):
            for c in PDSCH_configs:
                assert isinstance(c, PDSCHConfig), \
                    "Each element of PDSCH_configs must be a valide PDSCHConfig"
        else:
            raise ValueError("Invalid value for PDSCH_configs")

        # Check validity of provided PDSCH_configs
        num_tx = len(PDSCH_configs)
        num_streams_per_tx = PDSCH_configs[0].num_layers
        dmrs_grid = PDSCH_configs[0].dmrs_grid
        num_subcarriers = dmrs_grid[0].shape[0]
        num_ofdm_symbols = PDSCH_configs[0].l_d
        precoding = PDSCH_configs[0].precoding
        dmrs_ports = []
        num_pilots = np.sum(PDSCH_configs[0].dmrs_mask)
        for PDSCH_config in PDSCH_configs:
            assert PDSCH_config.num_layers==num_streams_per_tx, \
                "All PDSCH_configs must have the same number of layers"
            assert PDSCH_config.dmrs_grid[0].shape[0]==num_subcarriers, \
                "All PDSCH_configs must have the same number of subcarriers"
            assert PDSCH_config.l_d==num_ofdm_symbols, \
                "All PDSCH_configs must have the same number of OFDM symbols"
            assert PDSCH_config.precoding==precoding, \
                "All PDSCH_configs must have a the same precoding method"
            assert np.sum(PDSCH_config.dmrs_mask)==num_pilots, \
                "All PDSCH_configs must have a the same number of masked REs"
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                for port in PDSCH_config.dmrs.dmrs_port_set:
                    if port in dmrs_ports:
                        msg = f"DMRS port {port} used by multiple transmitters"
                        warnings.warn(msg)
            dmrs_ports += PDSCH_config.dmrs.dmrs_port_set

        # Create mask and pilots tensors
        mask = np.zeros([num_tx,
                         num_streams_per_tx,
                         num_ofdm_symbols,
                         num_subcarriers], bool)
        num_pilots = np.sum(PDSCH_configs[0].dmrs_mask)
        pilots = np.zeros([num_tx, num_streams_per_tx, num_pilots], complex)
        for i, PDSCH_config in enumerate(PDSCH_configs):
            for j in range(num_streams_per_tx):
                ind0, ind1 = PDSCH_config.symbol_allocation
                mask[i,j] = np.transpose(
                                PDSCH_config.dmrs_mask[:, ind0:ind0+ind1])
                dmrs_grid = np.transpose(
                                PDSCH_config.dmrs_grid[j, :, ind0:ind0+ind1])
                pilots[i,j] = dmrs_grid[np.where(mask[i,j])]

        # Init PilotPattern class
        super().__init__(mask, pilots,
                         trainable=False,
                         normalize=False,
                         dtype=dtype)
