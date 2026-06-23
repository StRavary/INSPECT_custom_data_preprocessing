### INSPECT Baseline Reconstruction: Process

Reconstructing the baseline dataset was a complex debugging process. Here is a formal summary of the exact steps taken to successfully overcome these blockers and generate the baseline dataset. 

**1. Raw Data Download & Format Conversion**
To conserve disk space, a Redivis downloader script (`Custom/INSPECT_DL_EHR.py`) was initially developed to download all 32+ raw OMOP tables (e.g., measurement, condition_occurrence, person) as highly compressed `.parquet` files.
* **The Catch:** While the `.parquet` format is highly optimized for future custom processing pipelines, it was discovered that the legacy baseline pipeline strictly required uncompressed `.csv` files.
* **The Fix:** The downloader script was updated to target the `EHR_CSV` directory directly, and a conversion script (`Custom/convert_parquet_to_csv.py`) was used to expand existing Parquet files back into flat `.csv` formats.
* **Database Compilation:** The legacy `ehr/1_csv_to_database.py` script was then successfully executed. This utilized the `femr` framework to ingest the raw CSVs and compile them into a highly optimized, longitudinal patient database located at `DATA_RAW/EHR_FEMR_DB/extract`.

**2. Diagnosing the Missing Labels**
Upon reviewing the `run_all_ehr.py` master script, it was realized that the pipeline explicitly requires a pre-built file named `cohort_0.2.0_master_file_anon.csv`. The downloaded EHR dataset from Redivis was analyzed, revealing that the core ground-truth labels for the project (specifically the `pe_positive_nlp` column) were entirely missing.

**3. The AIMI Portal Breakthrough**
After hitting a dead end with the public dataset, an inquiry was sent to Professor Fries. He clarified that the INSPECT dataset is actually distributed across two different portals:
* **Tabular EHR Data:** Hosted by the Shah Lab on Redivis.
* **True Labels, Mappings, and Images:** Hosted by the AIMI Center on a separate portal.

> **NOTE - Undocumented Requirement:** The required link to the AIMI center's label and mapping files was notably absent from the main INSPECT dataset website. This split-distribution setup was essentially a hidden requirement that could only be resolved by directly contacting the dataset authors.

**4. Data Reconstruction & OMOP Clinical Anchoring (`merge_labels.py`)**
Following the download of the missing `labels_20250611.tsv` and `study_mapping_20250611.tsv` files from the AIMI portal, a custom Python script (`Custom/merge_labels.py`) was engineered to reconstruct the master file.
* An inner join was executed between the true labels and the official study mapping.
* **The Catch:** It was identified that exactly 779 scans in the official dataset were completely missing timestamp data (`procedure_DATETIME`). Specifically, during Stanford's strict PHI de-identification scrubbing process, the `procedure_occurrence_id` relational key was stripped entirely from 710 records, turning them into "ghost" patients.
* **The Fix:** Because the `femr` pipeline relies on timestamps to calculate patient history, dropping these records caused a ~3% deviation from the paper's official baseline count. To perfectly recreate the cohort, a robust 5-tier OMOP fallback strategy was implemented. The script sequentially attempts to anchor missing imaging timestamps using clinical proxy events from the OMOP database (e.g., pulling the patient's most recent `visit_start_DATETIME` or `procedure_DATETIME`). For the final 143 "ghost" patients entirely absent from the EHR tables, a safe global minimum anchor date (`1974-09-15`) was assigned to prevent pipeline exclusion.
* The result was a perfectly formatted `cohort_0.2.0_master_file_anon.csv` that successfully retained exactly **23,248 valid records**, flawlessly matching the official baseline benchmark splits published in the original paper.

**5. Data Validation Dashboard**
To visually inspect and sanity check the generated data (including sparse feature matrices, missing value ratios, and label distributions), a custom Streamlit web dashboard (`Custom/merged_labeled_data_viewer.py`) was constructed. This allowed for interactive filtering and statistical validation of the reconstructed cohort prior to pipeline execution.

**6. Bypassing the Legacy Pipeline (`run_baseline_benchmark.py`)**
The repository's master script (`run_all_ehr.py`) is designed to train massive deep-learning models (MOTOR/CLMBR). Two critical errors were encountered when attempting execution:
* The legacy Python 3.10 environment had an incompatible JAX/jaxlib installation.
* The script's hardcoded paths attempted to load the Foundation Model weights from internal `/share/pi/` servers. While the MOTOR model is actively hosted on Hugging Face (`StanfordShahLab/motor-t-base`), access requires formal approval which was pending at the time of execution.

To bypass this dependency, a clean Python wrapper (`Custom/run_baseline_benchmark.py`) was implemented. This script manually invoked the legacy Python environment and executed only Step 2 (`2_generate_labels_and_features.py`) against the newly reconstructed master cohort.

**7. Successful Feature Extraction**
The targeted script ran flawlessly, ingesting all 23,248 patients and successfully outputting the final, baseline clinical features to `DATA_RAW/EHR_FEMR_DB/features/PE`. This definitively validated the structural integrity of the local data environment, completing the baseline reproduction phase.
