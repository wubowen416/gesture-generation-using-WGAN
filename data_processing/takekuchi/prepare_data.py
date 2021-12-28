import os
import shutil
import pandas

NUM_OF_TEST = 90
FIRST_DATA_ID = 20
LAST_DATA_ID = 1182

AUGMENT = False

# data_root = "/media/wu/database/speech-to-gesture-takekuchi-2017/"
data_root = "./data/takekuchi/source"


def _split_and_format_data(data_dir):

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    _download_datasets(data_dir)


def _download_datasets(data_dir):

    _create_dir(data_dir)

    # prepare training data (including validation data)
    for i in range(FIRST_DATA_ID, LAST_DATA_ID - NUM_OF_TEST):
        filename = "audio" + str(i) + ".wav"
        original_file_path = os.path.join(f"{data_root}/speech/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = os.path.join(
                data_dir + "/train/inputs/" + filename)
            print(target_file_path)
            shutil.copyfile(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")
        filename = "gesture" + str(i) + ".bvh"
        original_file_path = os.path.join(f"{data_root}/motion/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = os.path.join(
                data_dir + "/train/labels/" + filename)
            print(target_file_path)
            shutil.copyfile(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")

    # prepare test data
    for i in range(LAST_DATA_ID - NUM_OF_TEST, LAST_DATA_ID + 1, 2):
        filename = "audio" + str(i) + ".wav"
        original_file_path = os.path.join(f"{data_root}/speech/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = os.path.join(
                data_dir + "/test/inputs/" + filename)
            print(target_file_path)
            shutil.copyfile(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")
        filename = "gesture" + str(i) + ".bvh"
        original_file_path = os.path.join(f"{data_root}/motion/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = os.path.join(
                data_dir + "/test/labels/" + filename)
            print(target_file_path)
            shutil.copyfile(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")

    # prepare dev data (does not affect results of training at all)
    for i in range(LAST_DATA_ID - NUM_OF_TEST + 1, LAST_DATA_ID + 1, 2):
        filename = "audio" + str(i) + ".wav"
        original_file_path = os.path.join(f"{data_root}/speech/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = os.path.join(
                data_dir + "/dev/inputs/" + filename)
            print(target_file_path)
            shutil.copyfile(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")
        filename = "gesture" + str(i) + ".bvh"
        original_file_path = os.path.join(f"{data_root}/motion/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = os.path.join(
                data_dir + "/dev/labels/" + filename)
            print(target_file_path)
            shutil.copyfile(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")

    # data augmentation
    # if AUGMENT:
    #     os.system('./data_processing/add_noisy_data.sh {0} {1} {2} {3}'.format(
    #         "train", FIRST_DATA_ID, LAST_DATA_ID-NUM_OF_TEST, data_dir))

    extracted_dir = os.path.join(data_dir)

    dev_files, train_files, test_files = _format_datasets(extracted_dir)

    dev_files.to_csv(os.path.join(extracted_dir, "gg-dev.csv"), index=False)
    train_files.to_csv(os.path.join(
        extracted_dir, "gg-train.csv"), index=False)
    test_files.to_csv(os.path.join(extracted_dir, "gg-test.csv"), index=False)


def _create_dir(data_dir):

    dir_names = ["train", "test", "dev"]
    sub_dir_names = ["inputs", "labels"]

    # create ../data_dir/[train, test, dev]/[inputs, labels]
    for dir_name in dir_names:
        dir_path = os.path.join(data_dir, dir_name)
        print(dir_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)  # ../data/train

        for sub_dir_name in sub_dir_names:
            dir_path = os.path.join(data_dir, dir_name, sub_dir_name)
            print(dir_path)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)


def _format_datasets(extracted_dir):
    train_files = _files_to_pandas_dataframe(
        extracted_dir, "train", range(FIRST_DATA_ID, LAST_DATA_ID - NUM_OF_TEST))
    test_files = _files_to_pandas_dataframe(extracted_dir, "test", range(
        LAST_DATA_ID - NUM_OF_TEST, LAST_DATA_ID + 1, 2))
    dev_files = _files_to_pandas_dataframe(extracted_dir, "dev", range(
        LAST_DATA_ID - NUM_OF_TEST+1, LAST_DATA_ID + 1, 2))

    return dev_files, train_files, test_files


def _files_to_pandas_dataframe(extracted_dir, set_name, idx_range):
    files = []
    for idx in idx_range:
        # original files
        try:
            input_file = os.path.abspath(
                os.path.join(extracted_dir, set_name, "inputs", "audio" + str(idx) + ".wav"))
        except OSError:
            continue
        try:
            label_file = os.path.abspath(
                os.path.join(extracted_dir, set_name, "labels", "gesture" + str(idx) + ".bvh"))
        except OSError:
            continue
        try:
            wav_size = os.path.getsize(input_file)
        except OSError:
            continue

        files.append((input_file, wav_size, label_file))

        # noisy files
        try:
            noisy_input_file = os.path.abspath(
                os.path.join(extracted_dir, set_name, "inputs", "naudio" + str(idx) + ".wav"))
        except OSError:
            continue
        try:
            noisy_wav_size = os.path.getsize(noisy_input_file)
        except OSError:
            continue
        print(str(idx))

        files.append((noisy_input_file, noisy_wav_size, label_file))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "bvh_filename"])


if __name__ == "__main__":
    split_dir = os.path.join(data_root, 'split')
    _split_and_format_data(split_dir)
