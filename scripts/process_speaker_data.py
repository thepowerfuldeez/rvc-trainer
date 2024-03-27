import json
import pandas as pd
from pathlib import Path

vctk_n2id = json.loads(Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/vctk_name2id.json").read_text())
vctk_n2id = {k.replace("/home/george", "/mnt/harddrive"): v for k, v in vctk_n2id.items()}
vctk_speakers = {x.name for x in Path("/mnt/harddrive/datasets/speech/vctk/wav48_silence_trimmed").iterdir() if
                 x.is_dir()}
vctk_spk2id = {s: i for i, s in enumerate(sorted(vctk_speakers))}
n2spk = {}
for n, id_ in vctk_n2id.items():
    n2spk[f"{id_}"] = vctk_spk2id[Path(n).stem.split("_")[0]]

opensinger_n2id = json.loads(
    Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/opensinger_name2id.json").read_text())
opensinger_n2id = {k.replace("/home/george", "/mnt/harddrive"): v for k, v in opensinger_n2id.items()}
keys = [Path(x).relative_to("/mnt/harddrive/datasets/singing_voice/OpenSinger/") for x in opensinger_n2id]
opensinger_path_mapping = {}
opensinger_path2spkid = {}
cur_spk_id = max(n2spk.values())
for k in sorted(keys):
    parent_f = k.parent
    dict_k = str(parent_f.parent / parent_f.name.split("_")[0])
    if dict_k not in opensinger_path_mapping:
        cur_spk_id += 1
        opensinger_path_mapping[dict_k] = cur_spk_id
    opensinger_path2spkid[k] = opensinger_path_mapping[dict_k]
for n, id_ in opensinger_n2id.items():
    k = Path(n).relative_to("/mnt/harddrive/datasets/singing_voice/OpenSinger/")
    n2spk[f"{id_}"] = opensinger_path2spkid[k]

acappella_n2id = json.loads(
    Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/acappella_name2id.json").read_text())
acappella_n2id = {k.replace("/home/george", "/mnt/harddrive"): v for k, v in acappella_n2id.items()}
cur_spk_id = max(n2spk.values())
for n, id_ in acappella_n2id.items():
    n2spk[f"{id_}"] = cur_spk_id + int(Path(n).stem.split("_", maxsplit=1)[0])

dsd100_n2id = json.loads(
    Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/dsd100_name2id.json").read_text())
dsd100_n2id = {k.replace("/home/george", "/mnt/harddrive"): v for k, v in dsd100_n2id.items()}
dsd100_speakers = {Path(x).stem.split("__")[0] for x in dsd100_n2id}
dsd100_spk2id = {s: i for i, s in enumerate(sorted(dsd100_speakers))}
cur_spk_id = max(n2spk.values())
for n, id_ in dsd100_n2id.items():
    n2spk[f"{id_}"] = cur_spk_id + dsd100_spk2id[Path(n).stem.split("__", maxsplit=1)[0]]

mixing_secrets_n2id = json.loads(
    Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/mixing_secrets_name2id.json").read_text())
mixing_secrets_n2id = {k.replace("/home/george", "/mnt/harddrive"): v for k, v in mixing_secrets_n2id.items()}
mixing_secrets_speakers = {Path(x).parent.parent.name for x in mixing_secrets_n2id}
mixing_secrets_spk2id = {s: i for i, s in enumerate(sorted(mixing_secrets_speakers))}
cur_spk_id = max(n2spk.values())
for n, id_ in mixing_secrets_n2id.items():
    n2spk[f"{id_}"] = cur_spk_id + mixing_secrets_spk2id[Path(n).parent.parent.name]

csd_n2id = json.loads(Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/csd_name2id.json").read_text())
csd_n2id = {k.replace("/home/george", "/mnt/harddrive"): v for k, v in csd_n2id.items()}
for n, id_ in csd_n2id.items():
    n2spk[f"{id_}"] = max(n2spk.values()) + 1

yt_n2id = json.loads(Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/youtube_artists.json").read_text())
# video_id to speaker id
filtered_df = pd.read_csv("/mnt/harddrive/datasets/singing_voice/downloaded_youtube_audios/labels.csv")
artist_to_id = {k: i for i, k in enumerate(sorted(filtered_df.artist.unique()))}
print(artist_to_id)
yt_singers_path2spkid = {}
for _, row in filtered_df.iterrows():
    yt_singers_path2spkid[row['video_id']] = artist_to_id[row['artist']]
cur_spk_id = max(n2spk.values())
for n, id_ in yt_n2id.items():
    k = Path(n).stem
    n2spk[f"{id_}"] = cur_spk_id + yt_singers_path2spkid[k]

print("total speakers", max(n2spk.values()) + 1)

Path("/home/george/ai-covers-backend/src/rvc/logs/pretrain_1228/speaker_mapping.json").write_text(json.dumps(n2spk))
