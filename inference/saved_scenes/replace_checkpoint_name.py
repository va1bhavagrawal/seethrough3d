import pickle 
import os 

pkl_paths = os.listdir(".") 
pkl_paths = [_ for _ in pkl_paths if _.find(".pkl") != -1] 
for i, pkl_path in enumerate(pkl_paths):
    # with open(pkl_path, "rb") as f:
    #     pkl_data = pickle.load(f)
    # pkl_data["checkpoint"] = "seethrough3d_release/seethrough3d_release"
    # with open(pkl_path, "wb") as f:

    #     pickle.dump(pkl_data, f) 
    os.rename(pkl_path, f"example{i}.pkl") 
    os.rename(pkl_path.replace("pkl", "webp"), f"example{i}.webp") 