from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from hparams import hparams


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    with open(os.path.join(in_dir, 'train.txt'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            emotion_id = parts[0]
            #source_wav_path = os.path.join(in_dir, parts[1])
            #target_wav_path = os.path.join(in_dir, parts[2])
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, in_dir, parts[1], parts[2], emotion_id)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, in_dir, source_wav_name, target_wav_name, emotion_id):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    source_wav = audio.load_wav(os.path.join(in_dir, source_wav_name))
    target_wav = audio.load_wav(os.path.join(in_dir, target_wav_name))

    if hparams.rescaling:
        source_wav = source_wav / np.abs(source_wav).max() * hparams.rescaling_max
        target_wav = target_wav / np.abs(target_wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    #s_spectrogram = audio.spectrogram(source_wav).astype(np.float32)
    t_spectrogram = audio.spectrogram(target_wav).astype(np.float32)

    # Compute a mel-scale spectrogram from the wav:
    smel_spectrogram = audio.melspectrogram(source_wav).astype(np.float32)
    tmel_spectrogram = audio.melspectrogram(target_wav).astype(np.float32)
    s_n_frames = smel_spectrogram.shape[1]
    t_n_frames = tmel_spectrogram.shape[1]

    # Write the spectrograms to disk:
    #s_spectrogram_filename = 'source-spec-{}.npy'.format(source_wav_name)
    t_spectrogram_filename = 'target-spec-{}.npy'.format(target_wav_name.replace('.wav', ''))
    smel_filename = 'source-mel-{}.npy'.format(source_wav_name.replace('.wav', ''))
    tmel_filename = 'target-mel-{}.npy'.format(target_wav_name.replace('.wav', ''))
    #np.save(os.path.join(out_dir, s_spectrogram_filename), s_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, t_spectrogram_filename), t_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, smel_filename), smel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, tmel_filename), tmel_spectrogram.T, allow_pickle=False)


    # Return a tuple describing this training example:
    return (emotion_id, t_spectrogram_filename, smel_filename, tmel_filename, s_n_frames, t_n_frames)
