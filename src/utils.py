import json
import math
import os

import pandas as pd
import seaborn as sns
import numpy as np
from celluloid import Camera
from matplotlib import pyplot as plt
from scipy.stats import linregress
from collections import defaultdict


def get_num_frames(data):
    num_frames = min(
        len(data['skeleton']['elbow_left']['x']),
        len(data['head_gaze']['rx']),
        len(data['eye_gaze']['rx']))
    frames = list(range(num_frames))
    return num_frames, frames


def extract_joints(data):
    """ Extract skeleton data """

    num_frames, frames = get_num_frames(data)
    joints = np.zeros((len(data['skeleton'].keys()), num_frames, 4))

    for j, joint in enumerate(data['skeleton'].keys()):
        for f in frames:
            c = data['skeleton'][joint]['confidence'][f]
            x = data['skeleton'][joint]['x'][f]
            y = data['skeleton'][joint]['y'][f]
            z = data['skeleton'][joint]['z'][f]

            joints[j, f, 0] = c if c is not None else 0
            joints[j, f, 1] = x if x is not None else 0
            joints[j, f, 2] = y if y is not None else 0
            joints[j, f, 3] = z if z is not None else 0

    return joints


def extract_gaze(data):
    num_frames, frames = get_num_frames(data)

    # Extract eye gaze data
    eye_gaze = np.zeros((num_frames, 3))
    for f in frames:
        eye_gaze[f, 0] = data['eye_gaze']['rx'][f]
        eye_gaze[f, 1] = data['eye_gaze']['ry'][f]
        eye_gaze[f, 2] = data['eye_gaze']['rz'][f]

    # Extract head gaze data
    head_gaze = np.zeros((num_frames, 3))
    for f in frames:
        head_gaze[f, 0] = data['head_gaze']['rx'][f]
        head_gaze[f, 1] = data['head_gaze']['ry'][f]
        head_gaze[f, 2] = data['head_gaze']['rz'][f]

    return eye_gaze, head_gaze


def calculate_angle(vector1, vector2):
    """Calculate the angle (in degrees) between two vectors in 3D space."""

    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate the angle using the dot product formula
    cos_theta = dot_product / (magnitude1 * magnitude2)

    # To avoid floating-point errors that may cause issues with arc cos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def calculate_joint_angles(skeleton_keys, joints, frame):
    """Calculate angles between body parts in 3D skeleton for a specific frame."""

    # Get the coordinates for relevant joints
    sholder_left = joints[skeleton_keys.index('sholder_left'), frame, 1:4]
    sholder_right = joints[skeleton_keys.index('sholder_right'), frame, 1:4]
    elbow_left = joints[skeleton_keys.index('elbow_left'), frame, 1:4]
    elbow_right = joints[skeleton_keys.index('elbow_right'), frame, 1:4]
    wrist_left = joints[skeleton_keys.index('wrist_left'), frame, 1:4]
    wrist_right = joints[skeleton_keys.index('wrist_right'), frame, 1:4]

    # Create vectors between the joints
    sholder_to_sholder = sholder_right - sholder_left
    sholder_to_elbow_left = elbow_left - sholder_left
    sholder_to_elbow_right = elbow_right - sholder_right
    elbow_to_wrist_left = wrist_left - elbow_left
    elbow_to_wrist_right = wrist_right - elbow_right

    # Calculate angles
    angle_body_upper_arm_left = calculate_angle(sholder_to_sholder, sholder_to_elbow_left)
    angle_body_upper_arm_right = calculate_angle(sholder_to_sholder, sholder_to_elbow_right)
    angle_upper_lower_arm_left = calculate_angle(sholder_to_elbow_left, elbow_to_wrist_left)
    angle_upper_lower_arm_right = calculate_angle(sholder_to_elbow_right, elbow_to_wrist_right)

    return sholder_left, sholder_right, elbow_left, elbow_right, wrist_left, wrist_right, angle_body_upper_arm_left, angle_body_upper_arm_right, angle_upper_lower_arm_left, angle_upper_lower_arm_right


def plot_angle_lines(ax, joint_start, joint_mid, joint_end, angle, color):
    """Plot lines approximating the angle in 3D space."""

    # Plot the lines that define the angle
    ax.plot([joint_mid[0], joint_start[0]],
            [joint_mid[2], joint_start[2]],
            [joint_mid[1], joint_start[1]],
            color=color, linewidth=0.5)
    ax.plot([joint_mid[0], joint_end[0]],
            [joint_mid[2], joint_end[2]],
            [joint_mid[1], joint_end[1]],
            color=color, linewidth=0.5)

    # Calculate the mid-point of the angle for annotation
    midpoint_x = (joint_mid[0] + joint_start[0] + joint_end[0]) / 3
    midpoint_y = (joint_mid[1] + joint_start[1] + joint_end[1]) / 3
    midpoint_z = (joint_mid[2] + joint_start[2] + joint_end[2]) / 3

    # Annotate the angle value in 3D space
    ax.text(midpoint_x, midpoint_z, midpoint_y,
            f"{angle:.1f}°", color=color, fontsize=9, ha='center')


def line_seg(x_coords, y_coords, z_coords, idx_1, idx_2):
    return [x_coords[idx_1], x_coords[idx_2]], \
        [z_coords[idx_1], z_coords[idx_2]], \
        [y_coords[idx_1], y_coords[idx_2]]  # y and z swapped for orientation


def plot_skeleton(ax, x_coords, y_coords, z_coords, **kwargs):
    ax.scatter(x_coords[4], z_coords[4], y_coords[4], c='black', s=500)
    ax.plot(*line_seg(x_coords, y_coords, z_coords, 4, 5), **kwargs)
    ax.plot(*line_seg(x_coords, y_coords, z_coords, 6, 7), **kwargs)
    ax.plot(*line_seg(x_coords, y_coords, z_coords, 6, 0), **kwargs)
    ax.plot(*line_seg(x_coords, y_coords, z_coords, 0, 8), **kwargs)
    ax.plot(*line_seg(x_coords, y_coords, z_coords, 7, 1), **kwargs)
    ax.plot(*line_seg(x_coords, y_coords, z_coords, 1, 9), **kwargs)


def visualize(data, skeleton_keys, animation_frame_num, animation_interval, file_path):
    num_frames, frames = get_num_frames(data)

    joints = extract_joints(data)
    eye_gaze, head_gaze = extract_gaze(data)

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_subplot(projection='3d')  # Slow

    camera = Camera(fig)

    frame_step = max(1, math.floor(num_frames/animation_frame_num))
    for frame in range(0, num_frames, frame_step):

        # Plot skeleton direction
        plot_skeleton(ax, joints[:, frame, 1], joints[:, frame, 2], joints[:, frame, 3],
                      linewidth=3, c='black')

        # Get the coordinates of the head joint
        head_index = skeleton_keys.index('head')
        head_x = joints[head_index, frame, 1]
        head_y = joints[head_index, frame, 2]
        head_z = joints[head_index, frame, 3]

        eye_gaze_scaled = 500 * eye_gaze[frame, :]
        head_gaze_scaled = 500 * head_gaze[frame, :]

        # Plot head gaze direction
        ax.quiver(head_x, head_z, head_y,
                  head_gaze_scaled[0], head_gaze_scaled[2], head_gaze_scaled[1],
                  color='black', length=200, normalize=True, label='Head Gaze')

        # Plot eye gaze direction
        ax.quiver(head_x, head_z, head_y,
                  eye_gaze_scaled[0], eye_gaze_scaled[2], eye_gaze_scaled[1],
                  color='blue', length=200, normalize=True, label='Eye Gaze')

        # Calculate the angles for the current frame
        sholder_left, sholder_right, elbow_left, elbow_right, wrist_left, wrist_right, angle_body_upper_arm_left, angle_body_upper_arm_right, angle_upper_lower_arm_left, angle_upper_lower_arm_right = (
            calculate_joint_angles(skeleton_keys, joints, frame))

        # Calculate differences and determine colors
        diff_body_upper_arm = abs(angle_body_upper_arm_left - angle_body_upper_arm_right)
        diff_upper_lower_arm = abs(angle_upper_lower_arm_left - angle_upper_lower_arm_right)
        color_body_upper_arm = 'green' if diff_body_upper_arm <= 30 else 'red'
        color_upper_lower_arm = 'green' if diff_upper_lower_arm <= 30 else 'red'

        # Plot lines and labels for angles
        plot_angle_lines(ax, sholder_left, sholder_right, elbow_right, angle_body_upper_arm_left, color_body_upper_arm)
        plot_angle_lines(ax, sholder_right, sholder_left, elbow_left, angle_body_upper_arm_right, color_body_upper_arm)
        plot_angle_lines(ax, sholder_left, elbow_left, wrist_left, angle_upper_lower_arm_left, color_upper_lower_arm)
        plot_angle_lines(ax, sholder_right, elbow_right, wrist_right, angle_upper_lower_arm_right, color_upper_lower_arm)

        camera.snap()

    animation = camera.animate(interval=animation_interval)
    animation.save(os.path.splitext(file_path)[0] + '_vis.mp4')


def calculate_symmetry_differences(data, skeleton_keys, threshold):
    """Calculate differences between left and right angles across all frames."""

    num_frames, frames = get_num_frames(data)
    joints = extract_joints(data)

    diff_body_upper_arm_list = []
    diff_upper_lower_arm_list = []
    large_diff_body_upper_arm_count = 0
    large_diff_upper_lower_arm_count = 0

    for frame in range(num_frames):
        # Get the angles for the current frame
        _, _, _, _, _, _, angle_body_upper_arm_left, angle_body_upper_arm_right, angle_upper_lower_arm_left, angle_upper_lower_arm_right = (
            calculate_joint_angles(skeleton_keys, joints, frame))

        # Calculate the absolute differences between left and right angles
        diff_body_upper_arm = abs(angle_body_upper_arm_left - angle_body_upper_arm_right)
        diff_upper_lower_arm = abs(angle_upper_lower_arm_left - angle_upper_lower_arm_right)

        # Store the differences
        diff_body_upper_arm_list.append(diff_body_upper_arm)
        diff_upper_lower_arm_list.append(diff_upper_lower_arm)

        # Count frames with Large differences
        if diff_body_upper_arm > threshold:
            large_diff_body_upper_arm_count += 1
        if diff_upper_lower_arm > threshold:
            large_diff_upper_lower_arm_count += 1

    # Calculate the percentage of frames with Large differences
    large_diff_body_upper_arm_percentage = (large_diff_body_upper_arm_count / num_frames) * 100
    large_diff_upper_lower_arm_percentage = (large_diff_upper_lower_arm_count / num_frames) * 100

    return np.array(diff_body_upper_arm_list), np.array(diff_upper_lower_arm_list), large_diff_body_upper_arm_percentage, large_diff_upper_lower_arm_percentage


def plot_symmetry_distribution(diff_body_upper_arm_array, diff_upper_lower_arm_array, large_diff_body_upper_arm_percentage, large_diff_upper_lower_arm_percentage, threshold, user_id, ados_score):
    """Plot the distribution of symmetry differences across all frames."""

    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Symmetry Distribution of {user_id} (ADOS Score: {ados_score})', fontsize=14)

    # Plot distribution of body-upper arm differences
    plt.subplot(1, 2, 1)
    sns.histplot(diff_body_upper_arm_array, kde=True, color='blue', bins=30)
    plt.title(f'Distribution of Body-Upper Arm Angle Differences\n'
              f'{np.mean(diff_body_upper_arm_array):.2f} +/- {np.std(diff_body_upper_arm_array):.2f}\n'
              f'Large Percentage (threshold={threshold}): {large_diff_body_upper_arm_percentage:.1f}')
    plt.xlabel('Angle Difference (degrees)')
    plt.ylabel('Frequency')

    # Plot distribution of upper-lower arm differences
    plt.subplot(1, 2, 2)
    sns.histplot(diff_upper_lower_arm_array, kde=True, color='green', bins=30)
    plt.title(f'Distribution of Upper-Lower Arm Angle Differences\n'
              f'{np.mean(diff_upper_lower_arm_array):.2f} +/- {np.std(diff_upper_lower_arm_array):.2f}\n'
              f'Large Percentage (threshold={threshold}): {large_diff_upper_lower_arm_percentage:.1f}')
    plt.xlabel('Angle Difference (degrees)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def prepare_data_for_symmetry_analysis(dataset_root_path, skeleton_keys, distribution_visualize):
    users = [user for user in sorted(os.listdir(dataset_root_path)) if user.startswith('User ')]

    user_list = []
    session_list = []
    ados_score_list = []
    diff_body_upper_arm_mean_list = []
    diff_body_upper_arm_std_list = []
    diff_upper_lower_arm_mean_list = []
    diff_upper_lower_arm_std_list = []
    large_diff_body_upper_arm_percentage_list_threshold_30 = []
    large_diff_upper_lower_arm_percentage_list_threshold_30 = []
    large_diff_body_upper_arm_percentage_list_threshold_45 = []
    large_diff_upper_lower_arm_percentage_list_threshold_45 = []
    large_diff_body_upper_arm_percentage_list_threshold_60 = []
    large_diff_upper_lower_arm_percentage_list_threshold_60 = []
    task_ability_list = []
    task_difficulty_level_list = []

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

                user_list.append(user)
                session_list.append(session)

                diff_body_upper_arm_array, diff_upper_lower_arm_array, large_diff_body_upper_arm_percentage_threshold_30, large_diff_upper_lower_arm_percentage_threshold_30 = calculate_symmetry_differences(data, skeleton_keys, threshold=30)
                _, _, large_diff_body_upper_arm_percentage_threshold_45, large_diff_upper_lower_arm_percentage_threshold_45 = calculate_symmetry_differences(data, skeleton_keys, threshold=45)
                _, _, large_diff_body_upper_arm_percentage_threshold_60, large_diff_upper_lower_arm_percentage_threshold_60 = calculate_symmetry_differences(data, skeleton_keys, threshold=60)

                ados_score = data['ados']['preTest']['total']
                ados_score_list.append(ados_score)

                diff_body_upper_arm_mean_list.append(np.mean(diff_body_upper_arm_array))
                diff_body_upper_arm_std_list.append(np.std(diff_body_upper_arm_array))
                diff_upper_lower_arm_mean_list.append(np.mean(diff_upper_lower_arm_array))
                diff_upper_lower_arm_std_list.append(np.std(diff_upper_lower_arm_array))

                large_diff_body_upper_arm_percentage_list_threshold_30.append(large_diff_body_upper_arm_percentage_threshold_30)
                large_diff_upper_lower_arm_percentage_list_threshold_30.append(large_diff_upper_lower_arm_percentage_threshold_30)
                large_diff_body_upper_arm_percentage_list_threshold_45.append(large_diff_body_upper_arm_percentage_threshold_45)
                large_diff_upper_lower_arm_percentage_list_threshold_45.append(large_diff_upper_lower_arm_percentage_threshold_45)
                large_diff_body_upper_arm_percentage_list_threshold_60.append(large_diff_body_upper_arm_percentage_threshold_60)
                large_diff_upper_lower_arm_percentage_list_threshold_60.append(large_diff_upper_lower_arm_percentage_threshold_60)

                task_ability_list.append(data['task']['ability'])
                task_difficulty_level_list.append(data['task']['difficultyLevel'])

                if distribution_visualize:
                    plot_symmetry_distribution(
                        diff_body_upper_arm_array,
                        diff_upper_lower_arm_array,
                        large_diff_body_upper_arm_percentage_threshold_30,
                        large_diff_upper_lower_arm_percentage_threshold_30,
                        30,
                        user,
                        ados_score
                    )

                    plot_symmetry_distribution(
                        diff_body_upper_arm_array,
                        diff_upper_lower_arm_array,
                        large_diff_body_upper_arm_percentage_threshold_45,
                        large_diff_upper_lower_arm_percentage_threshold_45,
                        45,
                        user,
                        ados_score
                    )

                    plot_symmetry_distribution(
                        diff_body_upper_arm_array,
                        diff_upper_lower_arm_array,
                        large_diff_body_upper_arm_percentage_threshold_60,
                        large_diff_upper_lower_arm_percentage_threshold_60,
                        60,
                        user,
                        ados_score
                    )

    return pd.DataFrame({
        'user': user_list,
        'session': session_list,
        'task_ability': task_ability_list,
        'task_difficultyLevel': task_difficulty_level_list,
        'ados_score': ados_score_list,
        'diff_body_upper_arm_mean': diff_body_upper_arm_mean_list,
        'diff_body_upper_arm_std': diff_body_upper_arm_std_list,
        'diff_upper_lower_arm_mean': diff_upper_lower_arm_mean_list,
        'diff_upper_lower_arm_std': diff_upper_lower_arm_std_list,
        'large_diff_body_upper_arm_percentage_threshold_30': large_diff_body_upper_arm_percentage_list_threshold_30,
        'large_diff_upper_lower_arm_percentage_threshold_30': large_diff_upper_lower_arm_percentage_list_threshold_30,
        'large_diff_body_upper_arm_percentage_threshold_45': large_diff_body_upper_arm_percentage_list_threshold_45,
        'large_diff_upper_lower_arm_percentage_threshold_45': large_diff_upper_lower_arm_percentage_list_threshold_45,
        'large_diff_body_upper_arm_percentage_threshold_60': large_diff_body_upper_arm_percentage_list_threshold_60,
        'large_diff_upper_lower_arm_percentage_threshold_60': large_diff_upper_lower_arm_percentage_list_threshold_60,
    })


def symmetry_analysis(data, skeleton_keys, threshold, user_id):
    diff_body_upper_arm_array, diff_upper_lower_arm_array, large_diff_body_upper_arm_percentage, large_diff_upper_lower_arm_percentage = (
        calculate_symmetry_differences(data, skeleton_keys, threshold))

    ados_score = data['ados']['preTest']['total']
    plot_symmetry_distribution(diff_body_upper_arm_array, diff_upper_lower_arm_array, large_diff_body_upper_arm_percentage, large_diff_upper_lower_arm_percentage, threshold, user_id, ados_score)


def plot_symmetry_vs_ados(df, plt):
    # Plot for Body-Upper Arm Mean Difference vs ADOS Score
    plt.subplot(2, 2, 1)
    slope, intercept, r_value, p_value, std_err = linregress(df['ados_score'], df['diff_body_upper_arm_mean'])
    plt.errorbar(df['ados_score'], df['diff_body_upper_arm_mean'], yerr=df['diff_body_upper_arm_std'], fmt='o', color='blue', ecolor='lightblue', capsize=3)
    sns.regplot(data=df, x='ados_score', y='diff_body_upper_arm_mean', scatter=False, color='blue')
    plt.title(f'Mean Body-Upper Arm Angle Difference vs ADOS Score\np-value: {p_value:.3e}')
    plt.xlabel('ADOS Score')
    plt.ylabel('Mean Angle Difference (degrees)')

    # Plot for Upper-Lower Arm Mean Difference vs ADOS Score
    plt.subplot(2, 2, 2)
    slope, intercept, r_value, p_value, std_err = linregress(df['ados_score'], df['diff_upper_lower_arm_mean'])
    plt.errorbar(df['ados_score'], df['diff_upper_lower_arm_mean'], yerr=df['diff_upper_lower_arm_std'], fmt='o', color='green', ecolor='lightgreen', capsize=3)
    sns.regplot(data=df, x='ados_score', y='diff_upper_lower_arm_mean', scatter=False, color='green')
    plt.title(f'Mean Upper-Lower Arm Angle Difference vs ADOS Score\np-value: {p_value:.3e}')
    plt.xlabel('ADOS Score')
    plt.ylabel('Mean Angle Difference (degrees)')

    # Plot for Large Body-Upper Arm Percentage vs ADOS Score
    plt.subplot(2, 2, 3)
    slope, intercept, r_value, p_value, std_err = linregress(df['ados_score'], df['large_diff_body_upper_arm_percentage'])
    plt.errorbar(df['ados_score'], df['large_diff_body_upper_arm_percentage'], fmt='o', color='blue', ecolor='lightblue', capsize=3)
    sns.regplot(data=df, x='ados_score', y='large_diff_body_upper_arm_percentage', scatter=False, color='blue')
    plt.title(f'Large Body-Upper Arm Percentage vs ADOS Score\np-value: {p_value:.3e}')
    plt.xlabel('ADOS Score')
    plt.ylabel('Percentage of Frames with Large Angle Difference')

    # Plot for Large Upper-Lower Arm Percentage vs ADOS Score
    plt.subplot(2, 2, 4)
    slope, intercept, r_value, p_value, std_err = linregress(df['ados_score'], df['large_diff_upper_lower_arm_percentage'])
    plt.errorbar(df['ados_score'], df['large_diff_upper_lower_arm_percentage'], fmt='o', color='green', ecolor='lightgreen', capsize=3)
    sns.regplot(data=df, x='ados_score', y='large_diff_upper_lower_arm_percentage', scatter=False, color='green')
    plt.title(f'Large Upper-Lower Arm Percentage vs ADOS Score\np-value: {p_value:.3e}')
    plt.xlabel('ADOS Score')
    plt.ylabel('Percentage of Frames with Large Angle Difference')

    plt.tight_layout()
    plt.show()


def extract_user_metadata(dataset_root_path):
    users = [user for user in sorted(os.listdir(dataset_root_path)) if user.startswith('User ')]
    user_data_list = []

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
                filtered_data = {k: v for k, v in data.items() if k not in ['$id', '$schema', 'eye_gaze', 'frame_rate', 'head_gaze', 'skeleton', 'time']}
                flattened_data = flatten_nested_dict(filtered_data)

                # Add user and session as an identifier
                flattened_data['user'] = user
                flattened_data['session'] = session

                user_data_list.append(flattened_data)

    return pd.DataFrame(user_data_list)


def flatten_nested_dict(d, parent_key='', sep='_'):
    """
    Recursively flatten a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, defaultdict) or isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, list(v) if isinstance(v, set) else v))
    return dict(items)
