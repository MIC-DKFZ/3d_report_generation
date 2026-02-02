from vlm.utils.load_and_save_files import load_json, save_json

def main():
    dataset_json_paths = [
        "/home/j385i/Data/vlm/ctrate_for_cluster/ctrate_train.json",
        "/home/j385i/Data/vlm/ctrate_for_cluster/ctrate_train_one_recon.json",
        "/home/j385i/Data/vlm/ctrate_for_cluster/final_ctrate_train.json",
    ]

    for dataset_json_path in dataset_json_paths:
        dataset_json = load_json(dataset_json_path)
        print(len(dataset_json))
        dataset_json = [d for d in dataset_json if d["image_id"] != "train_9510_a_1"]
        print(len(dataset_json))

        save_json(dataset_json, dataset_json_path)



if __name__ == "__main__":
    main()
