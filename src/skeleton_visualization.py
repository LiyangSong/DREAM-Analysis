import os
import json
import utils


def skeleton_visualization(dataset_root_path, skeleton_keys, animation_frames, animation_interval):
    users = [user for user in sorted(os.listdir(dataset_root_path)) if user != '.DS_Store']

    for user in users:
        sessions_path = os.path.join(dataset_root_path, user)

        sessions = sorted(os.listdir(sessions_path))
        sessions = [session for session in sessions if
                    ('initial diagnosis' in session.lower()
                     or 'diagnosis abilities' in session.lower()
                     and os.path.splitext(session)[1] == '.json')]

        for session in sessions:
            session_path = os.path.join(sessions_path, session)

            with open(session_path, 'r') as file:
                data = json.load(file)

            print(user, session, data['task'])
            utils.visualize(data, skeleton_keys, animation_frames, animation_interval, session_path)


if __name__ == '__main__':
    skeleton_visualization(
        dataset_root_path='dataset',
        skeleton_keys=['elbow_left', 'elbow_right', 'hand_left', 'hand_right', 'head','sholder_center', 'sholder_left', 'sholder_right', 'wrist_left', 'wrist_right'],
        animation_frames=2000,
        animation_interval=200
    )
