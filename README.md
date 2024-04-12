The first Jupyter notebook is titled preprocessing_ZariDR3. The main objective is to homogenize the labels from SKIFF and retain only those labels that conform to the MK classification system.

To execute this code, it is necessary to have the catalog named ZariDR3_2arcsecSkiff.csv https://drive.google.com/file/d/1Vu9lyB-xSGzsQLR1f1jtC5gyHDY9Lttv/view?usp=drive_link, which was constructed using a cross-match between Zari DR3  (Zari with data from Gaia Data Release 3) https://drive.google.com/file/d/1etbSm_15a_nWZJkP5XPdzrgGVC7C4bFe/view?usp=drive_link and the complete SKIFF database https://drive.google.com/file/d/116K_U1-djnHWbtKFWw15YSTZcclVg2NC/view?usp=drive_link.

We use the preprocessed labels from ZariDR3_Prep_2arcscSkiff, available at this https://drive.google.com/file/d/1fDT3Fw0Rg3FzksvCyzD5OvUg2IoRXr2U/view?usp=drive_link, to train the balanced random forest. The training process is conducted using the Jupyter Notebook titled training_BRF. The results are saved for Gmag in the folder accessible at this https://drive.google.com/drive/folders/1PwqXYQs5sDm5UYA9OAtzLAY12jSaDm_8?usp=drive_link.

To set up the environment, use the command: conda env create -f RF.yml --name myenv
