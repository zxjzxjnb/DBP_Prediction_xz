# Dataset1 Formal Ablation SHAP Summary

## Formal 5-feature

### Random Forest
- BDCM: Bromide (0.4568), UV254 (0.2024), TOC (0.1617)
- DBCM: Bromide (0.3500), pH (0.1101), UV254 (0.0853)
- THM4: TOC (0.3386), Bromide (0.1472), UV254 (0.1443)

### XGBoost
- BDCM: Bromide (0.4518), UV254 (0.1875), TOC (0.1344)
- DBCM: Bromide (0.3675), Temperature (0.1087), UV254 (0.1065)
- THM4: TOC (0.3260), Bromide (0.1545), UV254 (0.1503)

### MLP
- BDCM: TOC (0.4960), Bromide (0.3271), UV254 (0.1352)
- DBCM: Bromide (0.3037), UV254 (0.1278), Temperature (0.0882)
- THM4: TOC (0.6468), Bromide (0.1385), pH (0.1223)

### KAN
- BDCM: Bromide (0.3795), TOC (0.2987), pH (0.0935)
- DBCM: Bromide (0.3301), UV254 (0.1226), Temperature (0.1209)
- THM4: TOC (0.4330), UV254 (0.2197), pH (0.0786)

## Formal 6-feature + Cl2

### Random Forest
- BDCM: Bromide (0.4600), Cl2 dose (0.2592), TOC (0.0977)
- DBCM: Bromide (0.3113), pH (0.1143), Cl2 dose (0.0827)
- THM4: Cl2 dose (0.2139), TOC (0.1892), Bromide (0.1207)

### XGBoost
- BDCM: Bromide (0.4569), Cl2 dose (0.2130), UV254 (0.1151)
- DBCM: Bromide (0.3712), Temperature (0.0949), UV254 (0.0779)
- THM4: Cl2 dose (0.2392), TOC (0.1593), Bromide (0.1318)

### MLP
- BDCM: Bromide (0.3530), Cl2 dose (0.2977), TOC (0.2916)
- DBCM: Bromide (0.3069), UV254 (0.1051), Temperature (0.0802)
- THM4: Cl2 dose (0.3581), TOC (0.1870), Bromide (0.0953)

### KAN
- BDCM: Bromide (0.3824), Cl2 dose (0.2202), TOC (0.1660)
- DBCM: Bromide (0.3055), UV254 (0.1069), Temperature (0.1010)
- THM4: Cl2 dose (0.2467), TOC (0.2071), UV254 (0.1726)
