# Modified from https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd
import os
import csv
import torch
from tqdm import tqdm

GENRE_MAP_NAME_ID = {
    'Rock'         : 12,
    'Electronic'   : 15,
    'Experimental' : 38,
    'Hip-Hop'      : 211,
    'Folk'         : 17,
    'Instrumental' : 1235,
    'Pop'          : 10,
    'Classical'    : 5
}

GENRE_MAP_ID_NAME = {
    12      :   'Rock',
    15      :   'Electronic',
    38      :   'Experimental',
    211     :   'Hip-Hop',
    17      :   'Folk',
    1235    :   'Instrumental',
    10      :   'Pop',
    5       :   'Classical'
}

if __name__=="__main__":

    fma_path = 'fma_medium'
    song_info = 'mp3_titles_and_genres_medium.csv'

    # Convert the spectrogram back to audio
    tensor_list = []
    genre_list = []
    error_list = []
    # dir: data/fma_medium/000, ...
    
    with open(song_info, 'r') as file:
        reader = csv.DictReader(file)
    
        for row in reader:
            mp3_path = fma_path + '/' + row['path']
            genre_id = int(row['genre_id'])
            
            # Only process file if the genre id is in GENRE_MAP_ID_NAME
            if(genre_id not in GENRE_MAP_ID_NAME):
                print(f"INFO: Not processing song {mp3_path}, id not in genre map")
            else:
                print(f"  INFO: Processing song {mp3_path}. ", flush=True, end='')
                
                if(os.path.isfile(mp3_path)):
                    # Catch librosa errors
                    try:
                        mel_db = get_mel_db(mp3_path, hop_length=512)
                        if(mel_db.shape[1] >= 1290):
                            print(f"Shape: {mel_db.shape}")
                            mel_db = mel_db[0:, 0:1290]
                            # print("Spectrogram Shape: " + str(get_spectrogram_db(mp3_path).shape))
                            spec_tensor = torch.from_numpy(mel_db)
                            
                            tensor_list.append(spec_tensor)
                            genre_list.append(genre_id)
                        else:
                            print(f"Not adding, shape: {mel_db.shape}")
                    except:
                        print()
                        print(f"  ERROR: Error opening {mp3_path}")
                        error_list.append(mp3_path)
    
    spectrogram_tensors = torch.stack(tensor_list)
    genre_tensors = torch.stack(genre_list)
    # spectrogram_tensors = spectrogram_tensors[0:12000].clone()
    print(f"INFO: Tensor shape: {spectrogram_tensors.shape}")
    print(f"INFO: Number of errors: {len(error_list)}")
    torch.save(spectrogram_tensors, "spec_tensors_512_hop_x.pt")
    torch.save(genre_tensors, "spec_tensors_512_hop_y.pt")