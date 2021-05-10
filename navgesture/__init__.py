import glob
import numpy as np
import torch
from navgesture.read_td_events import readATIS_tddat
from itertools import groupby

classes = ['do', 'ho', 'le', 'ri', 'se', 'up']
num_of_classes = len(classes)


def get_action_from_filename(filename):
    for idx, action in enumerate(classes):
        if '_%s_' % action in filename:
            return idx
    return -1


def accumulate_coordinations(a, b):
    new_a = []
    new_b = []

    for k,g in groupby(zip(a,b), lambda x: x[0][0] * 10000 + x[0][1]):
        g = list(g)
        new_a.append(g[0][0])
        new_b.append(sum([i[1] for i in g]))

    new_a = np.array(new_a).reshape((-1, 2))
    new_b = np.array(new_b)

    return new_a, new_b

def reduce_duplicated(coords, poolars):
    pos_idx = poolars >= 1
    neg_idx = poolars < 1
    
    pos_coords = coords[pos_idx]
    neg_coords = coords[neg_idx]
    pos_poolars = np.ones_like(poolars[pos_idx])
    neg_poolars = np.ones_like(poolars[neg_idx])
    
    pos_coords, pos_poolars = accumulate_coordinations(pos_coords, pos_poolars)
    neg_coords, neg_poolars = accumulate_coordinations(neg_coords, neg_poolars)
    
    pos_coords = np.concatenate((np.ones ((pos_coords.shape[0], 1)), pos_coords), axis=1)
    neg_coords = np.concatenate((np.zeros((neg_coords.shape[0], 1)), neg_coords), axis=1)
    
    return np.concatenate((pos_coords, neg_coords), axis=0), np.concatenate((pos_poolars, neg_poolars), axis=0)


def serialize_events(ev_coords, ev_times, ev_polarities, image_size=(128, 128), frame_step=1, coord_scale=(1,1), polarity_mode=None):
    frame = 0
    max_time = np.max(ev_times)
    current_time = np.min(ev_times)
    
    for i in [0, 1]:
        ev_coords[:, i] = np.clip(ev_coords[:, i] * coord_scale[i], 0, image_size[i] - 1)
    ev_coords[:, 1] = image_size[1] - ev_coords[:, 1] - 1
    
    while current_time < max_time:
        ev_lb = ev_times >= current_time
        ev_ub = ev_times < (current_time + frame_step)

        event_coord = ev_coords[ev_lb & ev_ub, :]
        event_polar = ev_polarities[ev_lb & ev_ub]
        
        if polarity_mode == 'accumulative':
            event_coord, event_polar = reduce_duplicated(event_coord, event_polar)
            event_coord = torch.from_numpy(event_coord).t()[[0,2,1], :]
            event_polar = torch.from_numpy(event_polar)
            retina = {'indicies': event_coord, 'values': event_polar, 'image_size': (2, *image_size)}
            
        elif polarity_mode == 'twolayer':
            event_polar = np.reshape(event_polar, (-1, 1))
            event_coord = np.concatenate((event_polar, event_coord), axis=1)            
            event_coord = torch.from_numpy(event_coord).t()[[0,2,1], :]
            event_polar = torch.ones(event_coord.shape[1])
            retina = {'indicies': event_coord, 'values': event_polar, 'image_size': (2, *image_size)}
            
        elif polarity_mode == 'onelayer':
            event_coord = torch.from_numpy(event_coord).t()[[1,0], :]
            event_polar = torch.from_numpy(2 * event_polar - 1)
            retina = {'indicies': event_coord, 'values': event_polar, 'image_size': image_size}
            
        elif polarity_mode == 'ignore' or polarity_mode is None:
            event_coord = torch.from_numpy(event_coord).t()[[1,0], :]
            event_polar = torch.from_numpy(event_polar)
            retina = {'indicies': event_coord, 'values': event_polar, 'image_size': image_size}
        
        current_time += frame_step
        frame += 1

        yield retina, frame

        
def frame_collector(file_name, **kwargs):
    frames = kwargs.get('frames')
    timestamps, coords, polarities, removed_events = readATIS_tddat(
        file_name,
        orig_at_zero=True,
        drop_negative_dt=True,
        verbose=False,
        events_restriction=[0, np.inf])
    
    retina_generator = serialize_events(
        coords,
        timestamps,
        polarities,
        image_size=kwargs.get('image_size'),
        frame_step=kwargs.get('frame_len'),
        coord_scale=kwargs.get('image_scale'),
        polarity_mode=kwargs.get('polarity_mode')
    )
    
    file_class_id = get_action_from_filename(file_name)
    
    x_train = []
    y_train = [file_class_id]
    for retina, f in retina_generator:
        if len(x_train) == frames:
            yield x_train, y_train
            x_train = []
            y_train = [file_class_id]
        
        x_train.append(retina)

        
def get_file_lists(**kwargs):
    path = kwargs.get('data_path')
    max_read_file = kwargs.get('max_read_file', None)
    counter = 0
    for file_name in glob.glob(path):
        yield file_name
        counter += 1
        
        if max_read_file is not None and counter >= max_read_file:
            break

    if counter == 0:
        raise Exception('no data file found at "%s"' % path)


def load_all_files(**kwargs):
    print('')
    counter1 = 0
    for file_name in get_file_lists(**kwargs):
        counter1 += 1
        counter2 = 0
        for x_chunk, y_chunk in frame_collector(file_name, **kwargs):
            counter2 += 1
            if not kwargs.get('silent', False):
                print('\r%d) reading:' % counter1, '...%s' % file_name[-30:], '(chunk: %d)' % counter2, end='')
            yield x_chunk, y_chunk


def load_all(**kwargs):
    batch_id = 0
    frames = kwargs.get('frames', 100)
    for x_chunk, y_chunk in load_all_files(**kwargs):
        image_size = x_chunk[0]['image_size']
        
        data = None
        for frame_id, x_data in enumerate(x_chunk):
            i = x_data['indicies']
            v = x_data['values']

            onex = np.ones((1, i.shape[1]))
            i = np.vstack((onex*frame_id, i))

            i = torch.as_tensor(i, dtype=torch.long)
            v = torch.as_tensor(v, dtype=torch.float)
            d = torch.sparse.FloatTensor(i, v, (frames, *image_size))
            if data is None:
                data = d
            else:
                data += d
        yield data, y_chunk[0]

