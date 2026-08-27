# gun_detection

Weapon detection for Philippine CCTV and webcam feeds, built on YOLOv8 and OpenCV.

Public weapon datasets cover modern firearms and generic knives. None of them cover a bolo, a
balisong, a paltik, a sumpak, or a carbine conversion kit — the things actually carried here. This
project exists to close that gap with a locally sourced, volunteer-contributed dataset, and to ship
a detector trained on it.

## Scope

Intended use is CCTV safety alerting with a human in the loop. Explicitly out of scope: face
recognition, identity tracking, and autonomous response. The model reports what it sees in a frame;
deciding what that means is a person's job.

## Run it

```bash
pip install ultralytics opencv-contrib-python
python detect.py                                    # default webcam
python detect.py 1                                  # second webcam
python detect.py rtsp://user:pass@host:554/stream1  # CCTV
python detect.py rtsp://... 0.55                    # raise the confidence threshold
python detect.py --selftest
```

RTSP is forced to TCP; UDP drops packets and smears frames on most CCTV hardware.

## Classes

`classes.yaml` is the single source of truth. IDs are **append-only, forever** — volunteers' label
files store the integer, so renumbering invalidates donated work. The table below is generated
(`python scripts/remap_classes.py --write-readme`); edit `classes.yaml`, not this section.

Naming is `snake_case` English, with Filipino terms only where English has no accurate word
(`bolo`, `balisong`, `paltik`, `sumpak`, `baston`, `indian_pana`). Anything the camera cannot
resolve gets an honest `*_unknown` class rather than a guess.

<!-- taxonomy:start -->
| id | class | group | Filipino | notes |
|---:|---|---|---|---|
| 0 | `person` | context |  | Import only from datasets that label every person; partial person labels teach the model that people are background. |
| 1 | `pistol` | firearm |  |  |
| 2 | `revolver` | firearm |  |  |
| 3 | `rifle_ar15` | firearm |  | AR-15/M4 pattern. |
| 4 | `rifle_ak` | firearm |  | AK pattern. |
| 5 | `rifle_bolt` | firearm |  |  |
| 6 | `shotgun` | firearm |  |  |
| 7 | `smg` | firearm |  |  |
| 8 | `carbine_conversion_kit` | firearm |  | Pistol in a carbine chassis. Common locally, absent from every public dataset. |
| 9 | `paltik` | firearm | paltik |  |
| 10 | `sumpak` | firearm | sumpak |  |
| 11 | `airsoft_replica` | firearm |  | Only when an orange tip or replica marking is actually visible. Otherwise label what it looks like. |
| 12 | `unknown_firearm` | firearm |  | Firearm-shaped, subtype not determinable from the frame. The honest CCTV label. |
| 13 | `bolo` | blade | itak | A farm tool in most Philippine frames. Still labeled -- threat tiering happens downstream. |
| 14 | `balisong` | blade | balisong |  |
| 15 | `kitchen_knife` | blade |  |  |
| 16 | `folding_knife` | blade |  |  |
| 17 | `karambit` | blade |  |  |
| 18 | `ice_pick` | blade |  |  |
| 19 | `scissors` | blade |  |  |
| 20 | `box_cutter` | blade |  |  |
| 21 | `sword` | blade |  | `spear` is mapped here for want of a polearm class; revisit if spear volume ever justifies its own id. |
| 22 | `axe` | blade |  |  |
| 23 | `knife_unknown` | blade |  | Blade visible, type not determinable. Most imported knife boxes land here. |
| 24 | `baseball_bat` | blunt |  |  |
| 25 | `pipe` | blunt |  |  |
| 26 | `crowbar` | blunt |  |  |
| 27 | `hammer` | blunt |  |  |
| 28 | `screwdriver` | blunt |  |  |
| 29 | `brass_knuckles` | blunt |  |  |
| 30 | `baston` | blunt | baston |  |
| 31 | `indian_pana` | improvised | indian pana |  |
| 32 | `grenade` | other |  |  |
| 33 | `stun_gun` | other |  |  |
| 34 | `blunt_unknown` | blunt |  | Blunt object, type not determinable. Import target. |
| 35 | `rifle_unknown` | firearm |  | Long gun, pattern not determinable. Keeps the long-gun/handgun distinction that `unknown_firearm` would throw away. |
<!-- taxonomy:end -->

Hard negatives — phones, wallets, umbrellas, water guns, power drills — belong in the dataset as
**unlabeled background images**, not as classes.

## Coarse views

Label fine, generate coarse. Fine → coarse is a free, repeatable transform; coarse → fine means
relabeling every image by hand.

```bash
python scripts/remap_classes.py datasets/ph-fine --to tier --apply    # 6 group classes
python scripts/remap_classes.py datasets/ph-fine --to binary --apply  # person / weapon
```

The binary view merges `screwdriver`, `hammer`, and `scissors` into `weapon`, which teaches the
model that every workbench is a threat. Use it as a tripwire, not as an alerting model — the tier
view is the one that should drive alerts.

## Contributing data

See [docs/LABELING.md](docs/LABELING.md) for the box rules and [docs/DATASET_SOURCES.md](docs/DATASET_SOURCES.md)
for imported dataset provenance.

Contributed images must be yours to share, and everyone identifiable in them must have consented.
Uploading an image asserts that. If you cannot assert it, do not upload it.

## Licence

AGPL-3.0, matching ultralytics. The dataset is licensed separately on Roboflow; a licence over code
says nothing about image rights.
