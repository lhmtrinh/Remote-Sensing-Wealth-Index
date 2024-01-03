# `df210.csv` - Asset Index Data

## Description
This file contains asset index data for the years 2010, 2012, 2014, and 2016.

## Notable Exclusions
- **2018**: This year was excluded because it does not have the sampling weight.
- **2019**: This dataset does not include data for 2019. It's important to note that the questionnaire for 2019 differs from the other years. Specifically, it contains a subset of the questions from other years. As a result, the value returned from a Principal Component Analysis (PCA) for 2019 will also be different.

## Additional Information
Another version of the file may include data for 2019. Not yet included!

# `sum_WI_10-18.csv` - Asset Index Data  inlcuding 2018

## Description
This file contains asset index data for the years 2010, 2012, 2014, 2016, 2018.

## Notable Exclusions
- **2018**: This year does not have weighted average asset index.

# `district_data.csv` - Consistent District Code 

## Description
Code of district, name over years and the long, lat, area in square meters

# `ward_WI_unweigthedPCA.csv` - Asset index 

## Description
Asset index per ward for national survey in 2014 and 2019

# `ward_data.csv` - Ward data and location 

## Description
Divide VN into grid of 30m/px by 255 px. Each row is a cell id and a ward within it. If a cell has more than 1 ward within then there are additional rows for them
