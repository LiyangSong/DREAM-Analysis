import pandas as pd
import utils


def symmetry_analysis(dataset_root_path, skeleton_keys, output_path):
    print("Extracting user metadata...")
    user_metadata_df = utils.extract_user_metadata(dataset_root_path)

    print("Preparing symmetry analysis data...")
    symmetry_df = utils.prepare_data_for_symmetry_analysis(
        dataset_root_path,
        skeleton_keys,
        distribution_visualize=False
    )
    symmetry_df.drop(['task_ability', 'task_difficultyLevel', 'ados_score'], axis=1, inplace=True)

    print("Merging metadata and symmetry data...")
    df = pd.merge(user_metadata_df, symmetry_df, how='left', on=['user', 'session'])

    print(f"Saving combined data to {output_path}...")
    df.to_csv(output_path, index=False)

    print("Done!")


if __name__ == '__main__':
    symmetry_analysis(
        dataset_root_path='../dataset',
        skeleton_keys=['elbow_left', 'elbow_right', 'hand_left', 'hand_right', 'head', 'sholder_center', 'sholder_left', 'sholder_right', 'wrist_left', 'wrist_right'],
        output_path='../results/analysis/symmetry_analysis.csv'
    )
