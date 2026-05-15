
## Task 1: OceanDepth 2050 — Global Sea Level Forecasting

# Overview

## Description
The year is 2050, and the effects of climate change have shifted from theoretical models to daily realities. The United Nations Global Climate Task Force (GCTF) has deployed a massive network of autonomous buoys, known as the "Sentinel Grid," across the world’s oceans. These buoys collect real-time data on water temperature, salinity, atmospheric pressure, and glacial melt rates.

To protect coastal cities and plan infrastructure for the next decade, we need precise predictions of sea-level fluctuations. While we have historical data dating back to the late 19th century, the non-linear acceleration of polar ice melt has made traditional linear models obsolete. Your challenge is to develop a regression model that can forecast mean sea-level rise for various coastal coordinates over the next 24 months.

Success in this task will allow governments to deploy flood defenses more effectively and save trillions of dollars in urban infrastructure.

## Evaluation

### Metric
Submissions are evaluated on the **Root Mean Squared Error (RMSE)** between the predicted sea-level change (in millimeters) and the actual observed change. 

### Submission Format
The submission should be a CSV file with the following format:
```
Station_Timestamp_Id,SeaLevelChange
NYC_2051_01,4.2
LON_2051_01,3.1
TOK_2051_01,5.8
etc.
```

# Dataset Description

You are provided with historical sensor data from 500 global stations.

## File and Data Field Descriptions

- **train.csv** - Monthly sensor readings from 2020 to 2049.
    - `Station_Id` - A unique identifier for the buoy station.
    - `Latitude / Longitude` - Geographic coordinates.
    - `Avg_Temp` - Average water temperature at 10m depth.
    - `Salinity` - Practical Salinity Units (PSU) measured at the surface.
    - `Ice_Melt_Index` - A composite score of nearby glacial runoff.
    - `Atmospheric_Pressure` - Mean sea-level pressure in hPa.
    - `SeaLevelRise` - The target variable (millimeter change from the 2020 baseline).
- **test.csv** - The same features for the years 2050–2051, excluding the target variable.
- **sample_submission.csv** - A submission file in the correct format.

-----

## Task 2: NeuroScan-X — Automated Brain Lesion Segmentation

# Overview

## Description
In the field of neurology, early detection of microscopic lesions is the key to halting degenerative diseases like Multiple Sclerosis. The "NeuroScan-X" initiative is a collaborative effort between leading research hospitals to standardize the analysis of high-resolution 3D MRI scans.

Manual segmentation of these lesions by radiologists is a time-consuming process and prone to inter-observer variability. We are challenging the community to build a computer vision model capable of performing automated image segmentation. You are provided with thousands of multi-modal MRI scans where expert neurologists have hand-labeled the "damaged" voxels. 

Your goal is to produce a model that generates a binary mask for each slice of an MRI, identifying the exact pixels where a lesion is present.

## Evaluation

### Metric
Submissions are evaluated based on the **Dice Coefficient**, which measures the overlap between the predicted segmentation mask and the ground truth.

### Submission Format
To reduce file size, submissions must use **Run-Length Encoding (RLE)** for the predicted masks. The CSV should contain:
```
ImageId,EncodedPixels
Scan001_Slice1,1 1 5 10
Scan001_Slice2,1 5 22 3
etc.
```

# Dataset Description

The dataset consists of NIfTI formatted 3D volumes, but for ease of use, they have been converted into 2D PNG slices.

## File and Data Field Descriptions

- **train_images/** - A folder of MRI slices (T1-weighted, T2-weighted, and FLAIR modalities).
- **train_masks/** - Binary PNG images where white pixels (255) represent a lesion and black (0) represents healthy tissue.
- **test_images/** - The slices for which you must predict the masks.
- **metadata.csv** - Supplemental data for each scan.
    - `Patient_ID` - Unique ID for the patient.
    - `Age` - Patient age at the time of scan.
    - `Manufacturer` - The brand of the MRI machine used (GE, Siemens, Philips).
    - `Field_Strength` - Magnetic field strength (1.5T or 3.0T).

-----

## Task 3: FinGuard — Fraudulent Transaction Network Analysis

# Overview

## Description
In the hyper-connected world of digital finance, money laundering has become a complex web of "smurfing" and "layering" across thousands of shell accounts. Traditional rule-based systems are failing to catch sophisticated syndicates that move small amounts of money through vast networks of seemingly unrelated individuals.

FinGuard is a large-scale graph dataset representing three months of transactions within a digital neo-bank. Your task is to perform **Node Classification**. Specifically, you must identify which "Accounts" (Nodes) are involved in a known money laundering ring. This is not just about the individual's behavior, but who they are connected to and how money flows through them.

Can you spot the "mules" and "architects" hidden in the noise of millions of legitimate transactions?

## Evaluation

### Metric
Submissions are evaluated using the **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**. This ensures that models are penalized for false positives while maintaining a high catch rate for rare fraudulent nodes.

### Submission Format
```
AccountId,FraudProbability
ACC_0921,0.98
ACC_4412,0.02
ACC_8819,0.45
etc.
```

# Dataset Description

The data is provided as a set of CSVs representing a directed graph.

## File and Data Field Descriptions

- **nodes.csv** - Details of the accounts.
    - `AccountId` - Unique node ID.
    - `AccountType` - Individual, Business, or Non-Profit.
    - `CreationDate` - Date the account was opened.
    - `VerifiedStatus` - Level of identity verification (KYC level).
- **edges.csv** - The transactions between nodes.
    - `SourceId` - The account sending money.
    - `TargetId` - The account receiving money.
    - `Amount` - Value in USD.
    - `Timestamp` - Exact time of transaction.
    - `Type` - Wire, P2P, or Internal Transfer.
- **train_labels.csv** - The target labels for the training set (1 for Fraud, 0 for Legitimate).

-----

## Task 4: RetailMind — Customer Persona Clustering

# Overview

## Description
Global retailer "OmniMart" has a problem: they have too much data and not enough insight. Their marketing department is currently sending the same generic coupons to 50 million customers, resulting in a very low conversion rate. To solve this, they want to move toward "Hyper-Personalization."

This is an **Unsupervised Learning** task. You are provided with a year's worth of anonymized customer behavior data, including purchase history, app engagement, and demographic hints. Your goal is to identify distinct, stable clusters of customers (Personas). 

Unlike other competitions, there is no "Target" column. You must prove that your clusters are meaningful. The winning solution will be used to tailor the OmniMart shopping experience for millions of people.

## Evaluation

### Metric
Since this is unsupervised, submissions will be evaluated using a combination of the **Silhouette Score** (for cluster cohesion/separation) and a **Stability Index** (measuring how consistent the clusters remain when the data is sub-sampled). 

*Note: For the leaderboard, we provide a "Validation Set" where we have hidden the labels of 5 predefined shopping segments (e.g., "Budget Parents," "Tech Enthusiasts") to check for alignment with known business categories.*

### Submission Format
```
CustomerId,ClusterId
CUST_001,0
CUST_002,4
CUST_003,1
etc.
```

# Dataset Description

## File and Data Field Descriptions

- **customer_profiles.csv** - Static data.
    - `CustomerId` - Unique ID.
    - `Region` - Geographic territory.
    - `Signup_Method` - Social media, Email, or In-store.
- **transaction_summary.csv** - Aggregated behavior.
    - `Total_Spend` - Total USD spent in 12 months.
    - `Category_Preference` - The department they visit most (e.g., Electronics, Grocery).
    - `Return_Rate` - Percentage of items returned.
- **app_logs.json** - Semi-structured data.
    - Contains clickstream data (average time on app, number of searches, night vs. day usage).

-----

## Task 5: VoxCritique — Multi-Language Sentiment & Intent Generation

# Overview

## Description
Customer support centers are overwhelmed by voice-to-text transcripts in dozens of different languages. To help human agents prioritize their work, we need an AI that can not only detect the "Sentiment" of a transcript but also generate a one-sentence "Executive Summary" of the customer's intent.

This is a **Natural Language Processing (NLP)** task that combines classification and text generation. You will be given thousands of raw text transcripts from international call centers. You must predict whether the customer is "Frustrated," "Neutral," or "Satisfied," and then generate a summary of their core problem (e.g., "Customer wants a refund for a broken toaster").

The challenge lies in the linguistic diversity; transcripts include English, Spanish, Mandarin, and Arabic, often mixed with local slang.

## Evaluation

### Metric
1. **Sentiment Accuracy (50%)**: Categorical accuracy of the sentiment label.
2. **Summary Quality (50%)**: Evaluated using the **ROUGE-L Score**, comparing your generated summary to a gold-standard summary written by human supervisors.

### Submission Format
```
TranscriptId,Sentiment,Summary
TR_990,Frustrated,Wants a refund for a late delivery.
TR_991,Neutral,Inquiring about store holiday hours.
etc.
```

# Dataset Description

## File and Data Field Descriptions

- **transcripts.csv** - The primary text data.
    - `TranscriptId` - Unique ID.
    - `Language` - The primary language detected.
    - `Raw_Text` - The transcribed text of the call.
    - `Duration` - Length of the call in seconds.
- **train_annotations.csv** - The training targets.
    - `Sentiment` - [Frustrated, Neutral, Satisfied].
    - `Summary` - A human-written summary of the intent.
- **test_transcripts.csv** - The text data for your predictions.
