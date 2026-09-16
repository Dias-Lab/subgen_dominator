# Input Data

Raw and intermediate feature-source files for the maize and *B. rapa* subgenome
dominance models are archived on Zenodo rather than stored in this repository,
due to file size.

**Zenodo DOI:** 10.5281/zenodo.22797527

## Setup
1. Download the archive from the DOI above.
2. Extract its contents into this `inputData/` folder, preserving the folder
   names below.

```text
## Expected structure
inputData/
├── maize_inputData/ # raw/intermediate feature files for maize
└── brapa_inputData/ # raw/intermediate feature files for B. rapa
```

Processed, model-ready feature sets (the files used to train the final
XGBoost models) are included directly in this repository under
`final_featuresets/` and do not require the Zenodo download.