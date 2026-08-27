# Imported dataset provenance

Every imported dataset gets a row here. AGPL over this repo's code says nothing about image rights,
so the licence is recorded before the data is used.

**Where the licence actually is:** the SDK's `project.license` is empty, but every export ships a
`README.dataset.txt` at its root carrying `License: ...`. Read it there rather than hunting the
project page.

Import with:

```bash
python scripts/import_dataset.py <workspace>/<project> --version N   # dry run
python scripts/import_dataset.py <workspace>/<project> --version N --apply
python scripts/dedupe.py datasets/ph-fine datasets/*                 # before any split
```

## Two tiers, kept apart

- **import-coarse** — bulk public data, mostly `unknown_firearm` / `knife_unknown` after mapping.
  Pretraining only.
- **ph-fine** — contributed and curated Philippine-context images with the fine taxonomy. The
  fine-tuning set, and the **only** evaluation benchmark.

Mixing them lets imported noise flatter the metrics of a model that has never seen a bolo.

## Candidates

Sizes marked *verified* were confirmed independently; the rest come from the submitted shortlist and
still need checking. The licence column is filled in from the project page at import time.

| Dataset | Images | Classes | Licence | Status |
|---|---:|---|---|---|
| [crime-detection-zbmr9/gun-knife-thesis](https://universe.roboflow.com/crime-detection-zbmr9/gun-knife-thesis) | 9,918 *verified* | gun 6,388 / knife 6,380 boxes | CC BY 4.0 | **v5** imported, 9,741 images / 12,538 boxes. v11 is augmented to 16,861 (duplicates baked in); v6 has no splits |
| [ai-0jtbr/knife-detection-hgvy2](https://universe.roboflow.com/ai-0jtbr/knife-detection-hgvy2) | 9,189 *verified* | v1 export merges all of them into one `Knife` class, 9,361 boxes | CC BY 4.0 | **v1** imported, 9,189 images. v2's split is broken (6,675 valid vs 2,514 train). Note the project's `Person`/`pisau`/`Pistol` classes do not survive into v1 -- it is a single-class knife set |
| [new-workspace-zhpxq/weapon-detection-djjj0](https://universe.roboflow.com/new-workspace-zhpxq/weapon-detection-djjj0) | ~9,657 *verified* | 29 classes | TBD | highest value, fine-grained; inspect the class list first |
| [test-7awfy/weapon-detection-f1lih](https://universe.roboflow.com/test-7awfy/weapon-detection-f1lih) | 9,633 | Grenade, Knife, Missile, Pistol, Rifle | TBD | candidate; Missile is out of scope |
| [nu-5eb9m/weapon-mkcq9](https://universe.roboflow.com/nu-5eb9m/weapon-mkcq9) | 8,820 | gun, Gun | TBD | candidate; every box lands in `unknown_firearm` |
| [weapondetection-mqgm5/weapon-detection-cabwp](https://universe.roboflow.com/weapondetection-mqgm5/weapon-detection-cabwp) | 8,855 | person, weapon, Handgun | TBD | candidate; `weapon` is refused and needs review |
| [image-model-urv6n/gun-knife-blunt_object-hywzs](https://universe.roboflow.com/image-model-urv6n/gun-knife-blunt_object-hywzs) | 7,959 | gun, knife, blunt_object | TBD | candidate; `blunt_object` maps to `blunt_unknown` |
| [shopsurveillance/weapon-detection-t2esr](https://universe.roboflow.com/shopsurveillance/weapon-detection-t2esr) | 7,043 | knife, pistol, retail objects | TBD | candidate; keep retail images unlabeled as hard negatives |
| [poke-mstry/gun-knife-ymlps](https://universe.roboflow.com/poke-mstry/gun-knife-ymlps) | 5,837 | knife, pistol | TBD | candidate; maps cleanly |
| [thuvaraga/weapon-detection-7twf4](https://universe.roboflow.com/thuvaraga/weapon-detection-7twf4) | 3,562 | handgun, knife, pistol, rifle | TBD | candidate |
| [comsci-rajamangala-isan/pistol-72vgk](https://universe.roboflow.com/comsci-rajamangala-isan/pistol-72vgk) | 3,364 | Pistol | TBD | candidate |
| [buildx/weapon-detection-7kro8](https://universe.roboflow.com/buildx/weapon-detection-7kro8) | 9,523 | firearms, knives, swords, cleavers, spear | TBD | review, broad taxonomy |
| [weapon-detection-yolov6/weapon-detection-sopxu](https://universe.roboflow.com/weapon-detection-yolov6/weapon-detection-sopxu) | 2,857 | duplicate spellings (rifle/Riffles, Grenade/grenade) | TBD | review; the alias map absorbs the spellings |
| [muhammadbugaje/weapon-detector-security](https://universe.roboflow.com/muhammadbugaje/weapon-detector-security) | 2,543 | person, Bike, Blade, Gun | TBD | review; keep Bike images unlabeled as hard negatives |
| [imam-maulana-b4xet/handgun-detection-jtvaj](https://universe.roboflow.com/imam-maulana-b4xet/handgun-detection-jtvaj) | 2,006 | handgun | TBD | candidate |
| [knife-detection-sjzqp/knife-detection-bstjz](https://universe.roboflow.com/knife-detection-sjzqp/knife-detection-bstjz) | 1,722 | knife | TBD | candidate |
| [pandit-deendayal-energy-university-ne0py/weapon-detection-jnutv](https://universe.roboflow.com/pandit-deendayal-energy-university-ne0py/weapon-detection-jnutv) | 1,118 | 23 classes | TBD | review |
| [capstone2025-mifho/military-base-object-detection](https://universe.roboflow.com/capstone2025-mifho/military-base-object-detection) | 12,098 | Hand-Gun, Pistol, Rifle, Knife | TBD | pretraining only; military imagery is the wrong domain for CCTV |
| cv-aquarium-dataset/balisong-sample | unknown | unknown | unknown | **unverified, likely does not exist.** Two independent searches surface no balisong dataset on Universe, and `cv-aquarium-dataset` is Roboflow's public aquarium sample workspace. Open it in a browser before trusting it. |

## Picking a version

Roboflow keeps every generated version, and the newest is often the worst choice:

- **Augmented versions are duplicates.** `gun-knife-thesis` v11 inflates 9,918 images to 16,861 by
  baking in flips and rotations. Ultralytics augments during training anyway, so this only adds
  near-identical images that leak across the train/val boundary.
- **Check the split ratio.** `knife-detection-hgvy2` v2 puts 6,675 images in valid against 2,514 in
  train. v1 is 8,172/559/458.
- **A version with no splits is unusable** here -- `train.py` needs `train/images` and
  `valid/images` to exist.

Query before downloading: `p.versions()` exposes `images`, `splits`, and `augmentation` per version.

## Rules applied to every import

1. **Person boxes are dropped** by default. Person labeling in these sets is partial, and partial
   labels teach the model that unlabeled people are background. Pass `--keep-person` only for a set
   that labels every person in every frame.
2. **Unmapped class names stop the import.** `weapon`, `dangerous object`, `missile` and friends are
   listed under `unmapped_review` in `classes.yaml` and are dropped deliberately; a name nobody has
   seen before fails loudly instead of being guessed at.
3. **Deduplicate before splitting.** These sets descend from the same handful of public corpora. Run
   `scripts/dedupe.py` over the union with the curated set listed first, so duplicates are deleted
   from the imports and never from ph-fine.
