"""
A Python module for radar simulation

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

::

    ██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝
    ██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝ 
    ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗ 
    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

"""


import numpy as np

# C infrastructure no longer required
def simpy(radar, targets, noise=True):

    """
    simpy(radar, targets, noise=True)

    Radar simulator with python engine

    :param Radar radar:
        Radar model
    :param list[dict] targets:
        Ideal point target list

        [{

        - **location** (*numpy.1darray*) --
            Location of the target (m), [x, y, z]
        - **rcs** (*float*) --
            Target RCS (dBsm)
        - **speed** (*numpy.1darray*) --
            Speed of the target (m/s), [vx, vy, vz]. ``default
            [0, 0, 0]``
        - **phase** (*float*) --
            Target phase (deg). ``default 0``

        }]

        *Note*: Target's parameters can be specified with
        ``Radar.timestamp`` to customize the time varying property.
        Example: ``location=(1e-3*np.sin(2*np.pi*1*radar.timestamp), 0, 0)``
    :param bool noise:
        Flag to enable noise calculation. ``default True``

    :return:
        {

        - **baseband** (*numpy.3darray*) --
            Time domain complex (I/Q) baseband data.
            ``[channes/frames, pulses, samples]``

            *Channel/frame order in baseband*

            *[0]* ``Frame[0] -- Tx[0] -- Rx[0]``

            *[1]* ``Frame[0] -- Tx[0] -- Rx[1]``

            ...

            *[N]* ``Frame[0] -- Tx[1] -- Rx[0]``

            *[N+1]* ``Frame[0] -- Tx[1] -- Rx[1]``

            ...

            *[M]* ``Frame[1] -- Tx[0] -- Rx[0]``

            *[M+1]* ``Frame[1] -- Tx[0] -- Rx[1]``

        - **timestamp** (*numpy.3darray*) --
            Refer to Radar.timestamp

        }
    :rtype: dict
    """

    # Data size
    frames = radar.time_prop["frame_size"] # Number of frames to simulate
    channels = radar.array_prop["size"]   # Spatial channel count
    pulses = radar.radar_prop["transmitter"].waveform_prop["pulses"] # Number of pulses per frame
    samples = radar.sample_prop["samples_per_pulse"] # A/D samples per pulse

    # Initialize return to zero
    baseband = np.zeros(
                    frames*channels,
                    pulses,
                    samples)

    # ?? shape of the timestamp of the samples    
    #ts_shape = np.shape(radar.time_prop["timestamp"])
    # time stamps of the samples
    ts  = np.shape(radar.time_prop["timestamp"])

    # Get shorter reference to radar components
    rx           = radar.radar_prop["receiver"]     # receiver parameters
    tx           = radar.radar_prop["transmitter"]  # transmitter parameters
    ego_loc      = radar.radar_prop["location"]     # radar location
    ego_spd      = radar.radar_prop["speed"]        # radar speed
    ego_rot      = radar.radar_prop["rotation"]     # radar rotation
    ego_rot_rate = radar.radar_prop["rot_rate"]     # radar rotation rate
    
    # get shorter reference to target components
    tgt_loc = tgt["location"]
    tgt_spd = tgt.get("speed", (0, 0, 0))
    tgt_rcs = tgt["rcs"]
    tgt_phs = tgt.get("phase", 0)
 
    # get relative location
    rel_loc = tgt_loc - ego_loc
    rel_spd = tgt_spd - ego_spd

    # get relative location at each point in time to radar center
    r = rel_loc + ts*rel_spd

    # get offset to each transmitter, receiver channel
    
    # Transmitter Channels
    #for idx_c in range(0, radar.radar_prop["transmitter"].txchannel_prop["size"]):
    #    tx_c.AddChannel(cp_TxChannel(radar.radar_prop["transmitter"], idx_c))

    # Receiver
    #for idx_c in range(0, radar.radar_prop["receiver"].rxchannel_prop["size"]):
    #    rx_c.AddChannel(cp_RxChannel(radar.radar_prop["receiver"], idx_c))

    # rotate to range, az, el coordinates
     
    # get amplitude of return
        #rx_c = Receiver[float_t](
    #    <float_t> radar.radar_prop["receiver"].bb_prop["fs"],
    #    <float_t> radar.radar_prop["receiver"].rf_prop["rf_gain"],
    #    <float_t> radar.radar_prop["receiver"].bb_prop["load_resistor"],
    #    <float_t> radar.radar_prop["receiver"].bb_prop["baseband_gain"]


#    if len(np.shape(radar.radar_prop["location"])) == 4:
#        locx_mv = radar.radar_prop["location"][:,:,:,0].astype(np_float)
 #       locy_mv = radar.radar_prop["location"][:,:,:,1].astype(np_float)
 #       locz_mv = radar.radar_prop["location"][:,:,:,2].astype(np_float)
 
 #   else:
 #       loc_mv = radar.radar_prop["location"].astype(np_float)
 #       loc_vt.push_back(Vec3[float_t](&loc_mv[0]))
    
    # Generate Gaussian background noise, if requested
    if noise:
        baseband = baseband +\
            radar.sample_prop["noise"]*(
                np.random.randn(
                    frames*channels,
                    pulses,
                    samples,
                ) + \
                1j * np.random.randn(
                    frames*channels,
                    pulses,
                    samples,
                )
            )

    if False: #radar.radar_prop["interf"] is not None:
        interf_radar_prop = radar.radar_prop["interf"].radar_prop
        """
        Transmitter
        """
        interf_tx_c = cp_Transmitter(radar.radar_prop["interf"])

        """
        Transmitter Channels
        """
        for idx_c in range(0, interf_radar_prop["transmitter"].txchannel_prop["size"]):
            interf_tx_c.AddChannel(cp_TxChannel(interf_radar_prop["transmitter"], idx_c))

        """
        Receiver
        """
        interf_rx_c = Receiver[float_t](
            <float_t> interf_radar_prop["receiver"].bb_prop["fs"],
            <float_t> interf_radar_prop["receiver"].rf_prop["rf_gain"],
            <float_t> interf_radar_prop["receiver"].bb_prop["load_resistor"],
            <float_t> interf_radar_prop["receiver"].bb_prop["baseband_gain"]
        )

        for idx_c in range(0, interf_radar_prop["receiver"].rxchannel_prop["size"]):
            interf_rx_c.AddChannel(cp_RxChannel(interf_radar_prop["receiver"], idx_c))

        interf_radar_c = Radar[float_t](interf_tx_c, interf_rx_c)

        if len(np.shape(interf_radar_prop["location"])) == 4:
            locx_mv = interf_radar_prop["location"][:,:,:,0].astype(np_float)
            locy_mv = interf_radar_prop["location"][:,:,:,1].astype(np_float)
            locz_mv = interf_radar_prop["location"][:,:,:,2].astype(np_float)
            spdx_mv = interf_radar_prop["speed"][:,:,:,0].astype(np_float)
            spdy_mv = interf_radar_prop["speed"][:,:,:,1].astype(np_float)
            spdz_mv = interf_radar_prop["speed"][:,:,:,2].astype(np_float)
            rotx_mv = interf_radar_prop["rotation"][:,:,:,0].astype(np_float)
            roty_mv = interf_radar_prop["rotation"][:,:,:,1].astype(np_float)
            rotz_mv = interf_radar_prop["rotation"][:,:,:,2].astype(np_float)
            rrtx_mv = interf_radar_prop["rotation_rate"][:,:,:,0].astype(np_float)
            rrty_mv = interf_radar_prop["rotation_rate"][:,:,:,1].astype(np_float)
            rrtz_mv = interf_radar_prop["rotation_rate"][:,:,:,2].astype(np_float)

            Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, interf_loc_vt)
            Mem_Copy_Vec3(&spdx_mv[0,0,0], &spdy_mv[0,0,0], &spdz_mv[0,0,0], bbsize_c, interf_spd_vt)
            Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, interf_rot_vt)
            Mem_Copy_Vec3(&rrtx_mv[0,0,0], &rrty_mv[0,0,0], &rrtz_mv[0,0,0], bbsize_c, interf_rrt_vt)
                                                                           
        else:
            loc_mv = interf_radar_prop["location"].astype(np_float)
            interf_loc_vt.push_back(Vec3[float_t](&loc_mv[0]))

            spd_mv = interf_radar_prop["speed"].astype(np_float)
            interf_spd_vt.push_back(Vec3[float_t](&spd_mv[0]))

            rot_mv = interf_radar_prop["rotation"].astype(np_float)
            interf_rot_vt.push_back(Vec3[float_t](&rot_mv[0]))

            rrt_mv = interf_radar_prop["rotation_rate"].astype(np_float)
            interf_rrt_vt.push_back(Vec3[float_t](&rrt_mv[0]))

        interf_radar_c.SetMotion(interf_loc_vt,
                        interf_spd_vt,
                        interf_rot_vt,
                        interf_rrt_vt)

        sim_c.Interference(radar_c, interf_radar_c, bb_real, bb_imag)

        interference = np.zeros((frames_c*channles_c, pulses_c, samples_c), dtype=complex)

        for ch_idx in range(0, frames_c*channles_c):
            for p_idx in range(0, pulses_c):
                for s_idx in range(0, samples_c):
                    bb_idx = ch_idx * chstride_c + p_idx * psstride_c + s_idx
                    interference[ch_idx, p_idx, s_idx] = bb_real[bb_idx] +  1j*bb_imag[bb_idx]

    else:
        interference = None

    return {"baseband": baseband,
            "timestamp": radar.time_prop["timestamp"],
            "interference": interference}
