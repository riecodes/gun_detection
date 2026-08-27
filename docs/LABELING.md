# Labeling rules

The class list lives in [`classes.yaml`](../classes.yaml). These are the rules for deciding which
class a box gets, and where the box goes.

## The one rule everything else follows from

**Label what the camera shows, not what you know.** If you photographed your own airsoft replica
and the frame shows no orange tip, it is not `airsoft_replica` — it is `unknown_firearm`. A model
trained on labels encoding knowledge the camera never had will confidently guess in deployment,
which is the exact failure that gets someone hurt.

## Boxes

- Tight on the object, not on the hand holding it. Include the whole weapon: barrel, grip, blade,
  handle.
- Skip the box if more than about half the object is occluded. A quarter of a grip is not a
  trainable example of a pistol.
- One class per box. Overlapping objects get overlapping boxes.
- Motion blur is fine and wanted — CCTV is blurry. Blur is not a reason to skip an image.

## Choosing the class

**When you cannot tell the subtype, say so.** `unknown_firearm`, `rifle_unknown`, `knife_unknown`
and `blunt_unknown` exist precisely for this. They are correct labels, not failures.

- Handgun shape, cannot tell semi-auto from revolver → `unknown_firearm`
- Clearly a long gun, cannot tell AR from AK → `rifle_unknown`
- Blade visible, type unclear → `knife_unknown`

**Dual-use tools are labeled whenever visible.** A bolo leaning against a fence is a `bolo`. A
hammer on a workbench is a `hammer`. Scissors on a desk are `scissors`. Threat depends on context —
held, raised, near a person — and that decision belongs in alert logic downstream, not in the
label. Making annotators judge intent produces inconsistent labels and a model that learns nothing
stable.

**Airsoft only when marked.** `airsoft_replica` requires a visible orange tip or replica marking in
that frame. Unmarked replicas get the class of the firearm they resemble, because that is what a
camera — and a responding person — actually sees.

## Filipino-specific classes

These have no English equivalent and are the reason this dataset exists:

| class | what it is |
|---|---|
| `bolo` | itak; single-edged machete-style blade, farm tool and weapon both |
| `balisong` | butterfly knife, Batangas knife |
| `paltik` | improvised/homemade firearm |
| `sumpak` | improvised shotgun |
| `baston` | arnis/eskrima stick |
| `indian_pana` | improvised dart or arrow launcher |
| `carbine_conversion_kit` | pistol mounted in a carbine chassis |

Photograph these in ordinary settings — a market, a yard, a jeepney, a sari-sari store — not only
posed against a wall. The model has to work on a CCTV frame, not a product shot.

## Background images

Images with **no** weapon in them are valuable and take zero annotation effort: phones held to ears,
wallets, umbrellas, power drills, brooms, water guns, cameras. Submit them unlabeled. They cut false
alarms more cheaply than any amount of extra weapon data.

## Privacy

Contributed images must be yours to share, and everyone identifiable must have consented. If you
cannot say that about an image, do not upload it. Faces of uninvolved bystanders should be blurred
before upload. A takedown request on any published image is honored.
