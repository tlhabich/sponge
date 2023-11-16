---
title: Cast Bellows
nav_order: 3
---

# Cast Bellows
The 3D-printed soft bellows are problematic for long-term use: the material ages over time and becomes porous. In addition, the layer-by-layer production causes the bellows to fail after just a few load cycles. The silicone casting process is an alternative, as the entire membrane cures homogeneously. The semi-modular robot is used as an example to show how casted bellows can be manufactured and replace the printed ones. Using similar approaches, cast bellows can also be realized for the modular SPONGE.
<p align="center">
<img src="../images/bellows_cast.png" width=500>
</p>

## Downloads

* 3D models
   * [Files for 3D-Printer (.stl)](/sponge/downloads/SPONGE_SemiModular_SiliconeBellow_CAD_STL.zip)
   * [CAD-Files created with Autodesk Inventor (.ipt,.iam)](/sponge/downloads/SPONGE_SemiModular_SiliconeBellow_CAD_inventor.zip)

## Components

|part no. | name | function | quantity <br> per bellows|
| :----: | :----: | :----: | :----: |
| **(1)** | lower mold | casting | 1 |
| **(2)** | core | casting | 1 |
| **(3)** | upper mold | casting | 1 |
| **(4)** | funnel | casting | 1 |
| **(5)** | plug | casting| 1 |
| **(6)** | screw M2x20 | mounting| 16 |
| **(7)** | nut M2 | mounting | 16 |
| **(8)** | upper platform | mounting <br> sealing | 1 |
| **(9)** | cast membrane | actuation | 1 |
| **(10)** | ring | prevent ballooning | 1 |
| **(11)** | lower plattform | mounting | 1 |

## Manufacturing and Assembly
This procedure was successfully tested with the silicone Dragon Skin 10 Slow (Smooth-On).
### Mold Preparation
1. Attach **(4)** to **(3)** using glue. Two separate parts are printed, resulting in smooth surfaces without support structures. Alternatively, **(3)** and **(4)** could also be printed in one step.
2. Position **(2)** in **(1)**. Make sure that **(2)** points in the right direction.
3. Mount **(3)**/**(4)** to **(1)**/**(2)** using **(6)**/**(7)**.

### Casting
Several vacuum/pressure cycles are used to eliminate trapped air bubbles in the membrane. The cycle time depends on the pot time of the used silicone (Dragon Skin 10 Slow: 45 minutes). It is recommended to use a silicone with a long pot time.
1. Mix both silicone parts (A+B) according to the manufacturer's instruction and place them in a vacuum pump.
2. Start the vacuum pump and leave the temperature controller at room temperature.
3. After eight minutes, pressurize the vacuum chamber for around 5 seconds and then continue the pump.
4. After another eight minutes, pressurize the vacuum chamber and take the cup out of the pump.
5. Pour the silicone into the mold, place it into the vacuum chamber and start the pump.
6. Continue again with vacuum/pressure cycles: eight minutes vacuum and five seconds pressurization. During the cycle, the absolute pressure inside the vacuum chamber varies between 7mbar and 512mbar. This depends on the vacuum pump used. After the third cycle, continue the vacuum for five minutes and then remove the mold from the vacuum chamber.
7. Attach **(5)** to the remaining mold parts **(1)**/**(2)**/**(3)**/**(4)**. Use a screw clamp in order to push out excess silicone.
8. Cure the silicone for the time specified by the manufacturer (Dragon Skin 10 Slow: 7 hours).

### Post-Processing
1. Take out **(9)** with **(2)** from the mold.
2. Remove carefully **(2)**. This is possible due to the soft material.
3. Cut off excess cured silicone from **(9)**.
4. Place **(10)** between the folds of **(9)**.
5. Glue **(11)** to the bottom of **(9)** using silicone rubber adhesive (Sil-Poxy).
6. Glue **(8)** to the top of **(9)** using the same silicone glue.
7. Cure the glue for the time specified by the manufacturer (24 hours).

## 3D-printing

When you [downloaded](/sponge/downloads/SPONGE_SemiModular_SiliconeBellow_CAD_STL.zip) the .stl files, you need to pay attention to the preparation of the files for printing. The upper platform of the bellows, printed with Polyjet technology, need to be printed as an assembly due to the sealing. Use “add as assembly” to create the print job.

| part no. |material | files |
| :----: | :----: | :----: |
|**(1)**|PLA filament | lower_mold.stl |
|**(2)**|PLA filament | core.stl |
|**(3)**|PLA filament | upper_mold.stl|
|**(4)**|PLA filament | funnel.stl|
|**(5)**|PLA filament | plug.stl|
|**(8)**|Agilus30 | cast_bellow_upper_platform_seal.stl |
|**(8)**|VeroBlackPlus/VeroWhitePlus | cast_bellow_upper_platform.stl |
|**(10)**|VeroBlackPlus/VeroWhitePlus | ring.stl |
|**(11)**|VeroBlackPlus/VeroWhitePlus | cast_bellow_lower_platform.stl |
