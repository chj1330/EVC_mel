from dtw import dtw
import numpy as np
import os
from numpy.linalg import norm
from os.path import join
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# frame, fft
hop_size = 256
sample_rate = 22050

def process_dtw(MEL_DIR, SAVE_DIR, target_spec, source_mel, target_mel):

    t_spec = np.load(join(MEL_DIR, target_spec))
    s_mel = np.load(join(MEL_DIR, source_mel))
    t_mel = np.load(join(MEL_DIR, target_mel))
    dist, cost, acc, path = dtw(s_mel, t_mel,
                                dist=lambda s_mel, t_mel: norm(s_mel - t_mel, ord=1))

    dtw_s_mel = np.asarray([s_mel[path[0][i]] for i in range(path[0].size)])
    dtw_t_mel = np.asarray([t_mel[path[1][i]] for i in range(path[1].size)])
    dtw_t_spec = np.asarray([t_spec[path[1][i]] for i in range(path[1].size)])

    frame_len = dtw_s_mel.shape[0]
    np.save(join(SAVE_DIR, target_spec), dtw_t_spec)
    np.save(join(SAVE_DIR, source_mel), dtw_s_mel)
    np.save(join(SAVE_DIR, target_mel), dtw_t_mel)

    return (target_spec, source_mel, target_mel, frame_len)


if __name__ == '__main__':

    DATA_ROOT = '../data'
    MEL_DIR = '../data/emotion'
    TARGET = ['Happy', 'Sad', 'Angry']

    for target in TARGET:

        SAVE_DIR = join(DATA_ROOT, 'dtw_{}'.format(target))

        if not os.path.exists(SAVE_DIR):
            print("mkdirs: {}".format(SAVE_DIR))
            os.makedirs(SAVE_DIR)

        executor = ProcessPoolExecutor(max_workers=8)
        futures = []
        with open(join(MEL_DIR, 'train_{}.txt'.format(target))) as f:
            for line in f:
                parts = line.strip().split('|')
                target_spec = parts[0]
                source_mel = parts[1]
                target_mel = parts[2]
                futures.append(executor.submit(partial(process_dtw, MEL_DIR, SAVE_DIR, target_spec, source_mel, target_mel)))

        metadata = [future.result() for future in tqdm(futures)]

        with open(os.path.join(SAVE_DIR, 'train.txt'), 'w', encoding='utf-8') as f:
            for m in metadata:
                f.write('|'.join([str(x) for x in m]) + '\n')
        frames = sum([m[3] for m in metadata])
        frame_shift_ms = hop_size / sample_rate * 1000
        hours = frames * frame_shift_ms / (3600 * 1000)
        print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
        print('Max length:  %d' % max(m[3] for m in metadata))











