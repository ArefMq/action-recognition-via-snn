#!/usr/bin/python
# -*- coding: utf8 -*
#####################
# read_td_events.py #
#####################
# Feb 2017 - Jean-Matthieu Maro
# Email: jean-matthieu dot maro, hosted at inserm, which is located in FRance.
# Thanks to Germain Haessig and Laurent Dardelet.

from struct import unpack, pack
import numpy as np
import sys


def peek(f, length=1):
    pos = f.tell()
    data = f.read(length)
    f.seek(pos)
    return data

def readATIS_tddat(file_name, orig_at_zero = True, drop_negative_dt = True, verbose = True, events_restriction = [0, np.inf]):

    """
    reads ATIS td events in .dat format

    input:
    filename: string, path to the .dat file
    orig_at_zero: bool, if True, timestamps will start at 0
    drop_negative_dt: bool, if True, events with a timestamp greater than the previous event are dismissed
    verbose: bool, if True, verbose mode.
    events_restriction: list [min ts, max ts], will return only events with ts in the defined boundaries

    output:
    timestamps: numpy array of length (number of events), timestamps
    coords: numpy array of size (number of events, 2), spatial coordinates: col 0 is x, col 1 is y.
    polarities: numpy array of length (number of events), polarities
    removed_events: integer, number of removed events (negative delta-ts)

    """

    polmask = 0x0002000000000000
    xmask = 0x000001FF00000000
    ymask = 0x0001FE0000000000
    polpadding = 49
    ypadding = 41
    xpadding = 32

    # This one read _td.dat files generated by kAER
    if verbose:
        print('Reading _td dat file... (' + file_name + ')')
    file = open(file_name,'rb')

    header = False
    while peek(file) == b'%':
        file.readline()
        header = True
    if header:
        ev_type = unpack('B',file.read(1))[0]
        ev_size = unpack('B',file.read(1))[0]
        if verbose:
            print('> Header exists. Event type is ' + str(ev_type) + ', event size is ' + str(ev_size))
        if ev_size != 8:
            print('Wrong event size. Aborting.')
            return -1, -1, -1, -1
    else: # set default ev type and size
        if verbose:
            print('> No header. Setting default event type and size.')
        ev_size = 8
        ev_type = 0

    # Compute number of events in the file
    start = file.tell()
    file.seek(0,2)
    stop = file.tell()
    file.seek(start)

    Nevents = int( (stop-start)/ev_size )
    dNEvents = Nevents/100
    if verbose:
        print("> The file contains %d events." %Nevents)

    # store read data
    timestamps = np.zeros(Nevents, dtype = int)
    polarities = np.zeros(Nevents, dtype = int)
    coords = np.zeros((Nevents, 2), dtype = int)

    ActualEvents = 0
    for i in np.arange(0, int(Nevents)):

        event = unpack('Q',file.read(8))
        ts = event[0] & 0x00000000FFFFFFFF
        # padding = event[0] & 0xFFFC000000000000
        pol = (event[0] & polmask) >> polpadding
        y = (event[0] & ymask) >> ypadding
        x = (event[0] & xmask) >> xpadding
        if i >= events_restriction[0] and ts>=timestamps[max(0,i-1)]:
            ActualEvents += 1
            timestamps[i] = ts
            polarities[i] = pol
            coords[i, 0] = x
            coords[i, 1] = y

        if verbose and i%dNEvents == 0:
            sys.stdout.write("> "+str(i/dNEvents)+"% \r")
            sys.stdout.flush()
        if i > events_restriction[1]:
            break
    file.close()
    if verbose:
        print ("> After loading events, actually found {0} events.".format(ActualEvents))

    timestamps = timestamps[:ActualEvents]
    coords = coords[:ActualEvents, :]
    polarities = polarities[:ActualEvents]

    #check for negative timestamps
    for ts in timestamps:
        if ts < 0:
            print('Found a negative timestamp.')

    if orig_at_zero:
        timestamps = timestamps - timestamps[0]

    drop_sum = 0
    if drop_negative_dt:
        if verbose:
            print('> Looking for negative dts...')
        # first check if negative TS differences
        just_dropped = True
        nPasses = 0
        while just_dropped:
            nPasses += 1
            index_neg = []
            just_dropped = False
            ii = 0
            while ii < (timestamps.size - 1):
                dt = timestamps[ii+1] - timestamps[ii]
                if dt < 0:  # alors ts en ii+1 plus petit que ii
                    index_neg += [ii+1]
                    ii += 1
                    just_dropped = True
                if verbose and ii%dNEvents == 0:
                    sys.stdout.write("> "+str(ii/dNEvents)+"% (pass "+str(nPasses)+") \r")
                    sys.stdout.flush()
                ii += 1
            if len(index_neg) > 0:
                drop_sum += len(index_neg)
                index_neg = np.array(index_neg)
                timestamps = np.delete(timestamps, index_neg)
                polarities = np.delete(polarities, index_neg)
                coords = np.delete(coords, index_neg, axis = 0)
                if verbose:
                    print('> Removed {0} events in {1} passes.'.format(drop_sum, nPasses))
        removed_events = drop_sum
    else:
        removed_events = -1
    if verbose:
        print("> Sequence duration: {0:.2f}s, ts[0] = {1}, ts[{2}] = {3}.".format(float(timestamps[-1] - timestamps[0]) / 1e6, timestamps[0], len(timestamps)-1, timestamps[-1]))


    return timestamps, coords, polarities, removed_events
