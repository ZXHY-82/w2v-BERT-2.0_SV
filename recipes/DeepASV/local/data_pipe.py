import os, sys, traceback
sys.path.append(os.path.split(__file__)[0])
from deeplab.utils.fileio import load_scp, load_trial


def prepare_scp_and_trial_list(scp_path, trial_path, group_id=None):
    
    # used_utt_ids = set()
    
    trial_list = []
    for trial_data in load_trial(trial_path):
        if group_id is not None:
            trial_data['utt1'] = '{}-{}'.format(group_id, trial_data['utt1'])
            trial_data['utt2'] = '{}-{}'.format(group_id, trial_data['utt2'])
        trial_list.append(trial_data)
        # used_utt_ids.add(trial_data['utt1'])
        # used_utt_ids.add(trial_data['utt2'])
        
    scp_list = []
    for scp_data in load_scp(scp_path):
        if group_id is not None:
            scp_data['reco'] = '{}-{}'.format(group_id, scp_data['reco'])
        # if scp_data['reco'] in used_utt_ids:
        scp_list.append(scp_data)
            
    return scp_list, trial_list