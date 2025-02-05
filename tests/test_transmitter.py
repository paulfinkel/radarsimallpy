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

import scipy.constants as const
import numpy as np
import numpy.testing as npt

from radarsimpy import Transmitter


def cw_tx():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    return Transmitter(f=24e9, t=10, tx_power=10, pulses=2)


def test_cw_tx():
    """_summary_
    """
    print("#### CW transmitter ####")
    cw = cw_tx()

    print("# CW transmitter parameters #")
    # assert np.array_equal(cw.fc_vect, np.ones(2)*24e9)
    # assert cw.pulse_length == 10
    # assert cw.bandwidth == 0
    assert cw.rf_prop["tx_power"] == 10
    assert cw.waveform_prop["prp"][0] == 10
    assert cw.waveform_prop["pulses"] == 2

    print("# CW transmitter channel #")
    assert cw.txchannel_prop["size"] == 1
    assert np.array_equal(cw.txchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(cw.txchannel_prop["az_angles"], [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.txchannel_prop["az_patterns"], [np.zeros(2)])
    assert np.array_equal(cw.txchannel_prop["el_angles"], [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.txchannel_prop["el_patterns"], [np.zeros(2)])

    print("# CW transmitter modulation #")
    assert np.array_equal(
        cw.txchannel_prop["pulse_mod"], [np.ones(cw.waveform_prop["pulses"])]
    )


def fmcw_tx():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6

    tx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
    }

    return Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=256,
        channels=[tx_channel],
    )


def test_fmcw_tx():
    """_summary_
    """
    print("#### FMCW transmitter ####")
    fmcw = fmcw_tx()

    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6
    pattern = pattern - np.max(pattern)

    print("# FMCW transmitter parameters #")
    # assert np.array_equal(fmcw.fc_vect, np.ones(256) * 24.125e9)
    assert fmcw.waveform_prop["pulse_length"] == 80e-6
    assert fmcw.waveform_prop["bandwidth"] == 100e6
    assert fmcw.rf_prop["tx_power"] == 10
    assert fmcw.waveform_prop["prp"][0] == 100e-6
    assert fmcw.waveform_prop["pulses"] == 256

    print("# FMCW transmitter channel #")
    assert fmcw.txchannel_prop["size"] == 1
    assert np.array_equal(fmcw.txchannel_prop["locations"], np.array([[0, 0, 0]]))
    assert np.array_equal(fmcw.txchannel_prop["az_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.txchannel_prop["az_patterns"], [pattern])
    assert np.array_equal(fmcw.txchannel_prop["el_angles"], [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.txchannel_prop["el_patterns"], [pattern])

    print("# FMCW transmitter modulation #")
    assert np.array_equal(
        fmcw.txchannel_prop["pulse_mod"], [np.ones(fmcw.waveform_prop["pulses"])]
    )


def tdm_fmcw_tx():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    wavelength = const.c / 24.125e9

    tx_channel_1 = {"location": (0, -4 * wavelength, 0), "delay": 0}
    tx_channel_2 = {"location": (0, 0, 0), "delay": 100e-6}

    return Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=20,
        prp=200e-6,
        pulses=2,
        channels=[tx_channel_1, tx_channel_2],
    )


def test_tdm_fmcw_tx():
    """_summary_
    """
    print("#### TDM FMCW transmitter ####")
    tdm = tdm_fmcw_tx()

    print("# TDM FMCW transmitter parameters #")
    # assert np.array_equal(tdm.fc_vect, np.ones(2) * 24.125e9)
    assert tdm.waveform_prop["pulse_length"] == 80e-6
    assert tdm.waveform_prop["bandwidth"] == 100e6
    assert tdm.rf_prop["tx_power"] == 20
    assert tdm.waveform_prop["prp"][0] == 200e-6
    assert tdm.waveform_prop["pulses"] == 2

    print("# TDM FMCW transmitter channel #")
    assert tdm.txchannel_prop["size"] == 2
    assert np.array_equal(
        tdm.txchannel_prop["locations"],
        np.array([[0, -4 * const.c / 24.125e9, 0], [0, 0, 0]]),
    )
    assert np.array_equal(
        tdm.txchannel_prop["az_angles"],
        [np.arange(-90, 91, 180), np.arange(-90, 91, 180)],
    )
    assert np.array_equal(tdm.txchannel_prop["az_patterns"], [np.zeros(2), np.zeros(2)])
    assert np.array_equal(
        tdm.txchannel_prop["el_angles"],
        [np.arange(-90, 91, 180), np.arange(-90, 91, 180)],
    )
    assert np.array_equal(tdm.txchannel_prop["el_patterns"], [np.zeros(2), np.zeros(2)])

    print("# TDM FMCW transmitter modulation #")
    assert np.array_equal(
        tdm.txchannel_prop["pulse_mod"],
        [np.ones(tdm.waveform_prop["pulses"]), np.ones(tdm.waveform_prop["pulses"])],
    )


def pmcw_tx(code1, code2):
    """_summary_

    :param code1: _description_
    :type code1: _type_
    :param code2: _description_
    :type code2: _type_
    :return: _description_
    :rtype: _type_
    """
    angle = np.arange(-90, 91, 1)
    pattern = np.ones(181) * 12

    pulse_phs1 = np.zeros(np.shape(code1))
    pulse_phs2 = np.zeros(np.shape(code2))
    pulse_phs1[np.where(code1 == 1)] = 0
    pulse_phs1[np.where(code1 == -1)] = 180
    pulse_phs2[np.where(code2 == 1)] = 0
    pulse_phs2[np.where(code2 == -1)] = 180

    mod_t1 = np.arange(0, len(code1)) * 4e-9
    mod_t2 = np.arange(0, len(code2)) * 4e-9

    tx_channel_1 = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
        "mod_t": mod_t1,
        "phs": pulse_phs1,
    }

    tx_channel_2 = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
        "mod_t": mod_t2,
        "phs": pulse_phs2,
    }

    return Transmitter(
        f=24.125e9,
        t=2.1e-6,
        tx_power=20,
        pulses=256,
        channels=[tx_channel_1, tx_channel_2],
    )


def test_pmcw_tx():
    """_summary_
    """
    code1 = np.array(
        [
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
        ]
    )
    code2 = np.array(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
        ]
    )

    print("#### PMCW transmitter ####")
    pmcw = pmcw_tx(code1, code2)

    print("# PMCW transmitter parameters #")
    # assert np.array_equal(pmcw.fc_vect, np.ones(256) * 24.125e9)
    assert pmcw.waveform_prop["pulse_length"] == 2.1e-6
    assert pmcw.waveform_prop["bandwidth"] == 0
    assert pmcw.rf_prop["tx_power"] == 20
    assert pmcw.waveform_prop["prp"][0] == 2.1e-6
    assert pmcw.waveform_prop["pulses"] == 256

    print("# PMCW transmitter channel #")
    assert pmcw.txchannel_prop["size"] == 2
    assert np.array_equal(
        pmcw.txchannel_prop["locations"], np.array([[0, 0, 0], [0, 0, 0]])
    )
    assert np.array_equal(
        pmcw.txchannel_prop["az_angles"], [np.arange(-90, 91, 1), np.arange(-90, 91, 1)]
    )
    assert np.array_equal(
        pmcw.txchannel_prop["az_patterns"], [np.zeros(181), np.zeros(181)]
    )
    assert np.array_equal(
        pmcw.txchannel_prop["el_angles"], [np.arange(-90, 91, 1), np.arange(-90, 91, 1)]
    )
    assert np.array_equal(
        pmcw.txchannel_prop["el_patterns"], [np.zeros(181), np.zeros(181)]
    )

    print("# PMCW transmitter modulation #")
    npt.assert_almost_equal(pmcw.txchannel_prop["waveform_mod"][0]["var"], code1)
    npt.assert_almost_equal(pmcw.txchannel_prop["waveform_mod"][1]["var"], code2)

    npt.assert_almost_equal(
        pmcw.txchannel_prop["waveform_mod"][0]["t"], np.arange(0, len(code1)) * 4e-9
    )
    npt.assert_almost_equal(
        pmcw.txchannel_prop["waveform_mod"][1]["t"], np.arange(0, len(code2)) * 4e-9
    )

    # assert np.array_equal(pmcw.chip_length, [4e-9, 4e-9])


def test_fsk_tx():
    """_summary_
    """
    print("#### FSK transmitter ####")


def test_bpm_fmcw_tx():
    """_summary_
    """
    print("#### BPM FMCW transmitter ####")
