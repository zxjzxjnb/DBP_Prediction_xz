# Input Structure Summary

Scope:
- Tai Lake / small dataset: 9 README inputs (`pH`, `COD`, `NH4-N`, `NO2-N`, `NO3-N`, `Bromide`, `TOC`, `UV254`, `Temperature`).
- Dataset1: 7-feature formal backbone (`pH`, `UV254`, `Temperature`, `TOC`, `Bromide`, `Cl2 dose`, `Contact time`).
- Bromide raw values sit on very different absolute scales across the two tables; treat the raw D1/TL bromide ratios as a structural flag first, and verify unit alignment before making a literal magnitude claim.

## Key Takeaways
- Among the 5 shared inputs, Dataset1 is narrowest relative to Tai Lake for Temperature (std ratio D1/TL = 0.413) and widest for Bromide (std ratio D1/TL = 7073.080).
- Dataset1 within-tsid variability is smallest for Contact time (median within/overall std ratio = 0.003) and largest for Temperature (ratio = 0.229).
- Dataset1-only inputs remain strongly right-skewed: Cl2 dose std = 4.860 mg/L and contact time std = 29.018 min.
- Tai Lake-only chemistry inputs are fully observed in this table; the widest spread among them is COD (std = 0.869).

## Shared Inputs

| feature     | dataset1_column | tailake_column | dataset1_mean | tailake_mean | dataset1_median | tailake_median | dataset1_std | tailake_std | std_ratio_dataset1_to_tailake | dataset1_iqr | tailake_iqr | iqr_ratio_dataset1_to_tailake |
| ----------- | --------------- | -------------- | ------------- | ------------ | --------------- | -------------- | ------------ | ----------- | ----------------------------- | ------------ | ----------- | ----------------------------- |
| pH          | ph_in_avg       | pH             | 7.658         | 7.683        | 7.770           | 7.700          | 0.979        | 0.159       | 6.153                         | 1.410        | 0.200       | 7.050                         |
| UV254       | uv_in_avg       | UV254_A_cm     | 0.113         | 0.112        | 0.065           | 0.097          | 0.120        | 0.047       | 2.584                         | 0.047        | 0.050       | 0.940                         |
| Temperature | temp_in_avg     | temp_C         | 21.102        | 22.211       | 21.500          | 24.000         | 3.444        | 8.331       | 0.413                         | 3.675        | 14.000      | 0.263                         |
| TOC         | toc_in_avg      | TOC_mg_L       | 4.482         | 4.299        | 3.365           | 4.161          | 3.098        | 0.804       | 3.852                         | 2.022        | 1.229       | 1.645                         |
| Bromide     | br_in_avg       | Br_mg_L        | 147.862       | 0.123        | 85.000          | 0.120          | 219.605      | 0.031       | 7073.080                      | 111.105      | 0.045       | 2441.868                      |

## Dataset1 Within-tsid Structure

| feature      | source_column | n_tsid_total | n_tsid_valid | within_std_mean | within_std_median | within_std_q25 | within_std_q75 | within_std_iqr | within_std_max | within_std_to_overall_ratio | share_below_half_overall_std |
| ------------ | ------------- | ------------ | ------------ | --------------- | ----------------- | -------------- | -------------- | -------------- | -------------- | --------------------------- | ---------------------------- |
| pH           | ph_in_avg     | 95           | 83           | 0.227           | 0.193             | 0.060          | 0.316          | 0.256          | 1.189          | 0.197                       | 0.904                        |
| UV254        | uv_in_avg     | 95           | 83           | 0.018           | 0.007             | 0.003          | 0.013          | 0.010          | 0.314          | 0.062                       | 0.952                        |
| Temperature  | temp_in_avg   | 95           | 82           | 1.318           | 0.790             | 0.328          | 1.446          | 1.118          | 8.260          | 0.229                       | 0.793                        |
| TOC          | toc_in_avg    | 95           | 83           | 0.491           | 0.331             | 0.112          | 0.524          | 0.412          | 6.753          | 0.107                       | 0.964                        |
| Bromide      | br_in_avg     | 95           | 82           | 27.407          | 9.955             | 2.143          | 22.172         | 20.029         | 462.033        | 0.045                       | 0.951                        |
| Cl2 dose     | cl2d_in_avg   | 95           | 83           | 0.788           | 0.450             | 0.206          | 0.841          | 0.635          | 12.488         | 0.093                       | 0.940                        |
| Contact time | time_sds_avg  | 95           | 83           | 2.145           | 0.097             | 0.000          | 0.504          | 0.504          | 30.131         | 0.003                       | 0.940                        |

## Plot Inventory
- `pH`: `ph.png`
- `UV254`: `uv254.png`
- `Temperature`: `temperature.png`
- `TOC`: `toc.png`
- `Bromide`: `bromide.png`
- `Cl2 dose`: `cl2_dose.png`
- `Contact time`: `contact_time.png`
- `COD`: `cod.png`
- `NH4-N`: `nh4_n.png`
- `NO2-N`: `no2_n.png`
- `NO3-N`: `no3_n.png`