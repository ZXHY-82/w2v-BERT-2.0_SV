import os


def init_spk2utt(dataset_dir, subset, spk2utt):
    cache_path = os.path.join(dataset_dir, '{}.spk2utt'.format(subset))
    if not os.path.exists(cache_path):
        print(f'No_{dataset_dir}_{subset}')
    
    with open(cache_path, 'r') as f:
        for line in f.readlines():
            spk_id, utt_path = line.strip('\n').split('\t')
            if spk_id not in spk2utt:
                spk2utt[spk_id] = []
            spk2utt[spk_id].append(utt_path)
    return 


def load_musan_dict(dataset_dir):
    "Load musan noises without vocals."
    path_dict = dict(noise=[], music=[], babb=[])

    # noise part
    for cls in ['noise/free-sound', 'noise/sound-bible']:
        cls_dir = os.path.join(dataset_dir, cls)
        for file in os.listdir(cls_dir):
            audio = os.path.join(dataset_dir, cls_dir, file)
            if os.path.exists(audio) and audio.endswith('.wav'):
                path_dict['noise'].append(audio)

    # music part
    for cls in ['music/fma', 'music/fma-western-art', 'music/hd-classical', 'music/jamendo', 'music/rfm']:
        anno_path = os.path.join(dataset_dir, cls, 'ANNOTATIONS')
        with open(anno_path, 'r') as f:
            annos = f.readlines()
        for d in annos:
            vocal = d.split(' ')[2]
            audio = os.path.join(dataset_dir, cls, d.split(' ')[0]+'.wav')
            if vocal=='N' and os.path.exists(audio):
                path_dict['music'].append(audio)

    # babb part         
    for cls in ['speech/librivox', 'speech/us-gov']:
        cls_dir = os.path.join(dataset_dir, cls)
        for file in os.listdir(cls_dir):
            audio = os.path.join(dataset_dir, cls_dir, file)
            if os.path.exists(audio) and audio.endswith('.wav'):
                path_dict['babb'].append(audio)
                
    return path_dict


def load_rirs(dataset_dir):
    
    path_list = []
    for d in ['simulated_rirs/mediumroom','simulated_rirs/smallroom']:
        sub_dir = os.path.join(dataset_dir, d)
        if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
            for r in os.listdir(sub_dir):
                room_dir = os.path.join(sub_dir, r)
                if not os.path.isdir(room_dir):   
                    continue
                for file in os.listdir(room_dir):
                    audio = os.path.join(room_dir, file)
                    if os.path.exists(audio) and audio.endswith('.wav'):
                        path_list.append(audio)
                        
    return path_list


def load_audio_corpus(dataset_dir,
                    subsets=['audio']):
    spk2utt = {}
    for subset in subsets:
        init_spk2utt(dataset_dir, subset, spk2utt)

    return spk2utt

    