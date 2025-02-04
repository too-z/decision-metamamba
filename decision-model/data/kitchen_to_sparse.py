import h5py
import pickle
import collections
import numpy as np

from tqdm import tqdm

keys = ['observations', 'actions', 'rewards', 'terminals']

datasets = ['mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
            'kitchen_microwave_kettle_light_slider-v0.hdf5',
            'kitchen_microwave_kettle_bottomburner_light-v0.hdf5']

names = ['mini_kitchen_microwave_kettle_light_slider-v0',
            'kitchen_microwave_kettle_light_slider-v0',
            'kitchen_microwave_kettle_bottomburner_light-v0']

def check_data(dataset, name):
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            raise RuntimeError('All datasets should have timeouts')
            # final_timestep = (episode_step == 1000-1)
        for k in keys:
            data_[k].append(dataset[k][i])
        end_of_episode = done_bool or final_timestep
        if end_of_episode:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(paths, f)


for i in tqdm(range(len(datasets))):
    file_name = datasets[i]
    print(file_name)
    check_data(dataset=h5py.File(file_name, 'r'), name=names[i])

for i in range(len(names)):
    with open(names[i] + '.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    for j in range(len(trajectories)):
        trajectories[j]['rewards'] = np.diff(np.insert(trajectories[j]['rewards'], 0, 0))
    with open(f'{names[i]}-sparse.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

# check
with open(names[0]+'-sparse.pkl', 'rb') as f:
    trajectories = pickle.load(f)

