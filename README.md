# Master’s project: 'Mind Reading' with MEG and Machine Learning

## Abstract
*Can we read the thoughts by looking at the neural signals produced by the brain?*

Reading complex thoughts is clearly 
still beyond the scope of our technology (for better or for worse). In this Master’s project our aim is much simpler:
we wish to predict whether the person is thinking of a noun (or object-like imagery) as opposed to verb 
(or action-like imagery).

To this end, we will use magnetoencephalography (MEG), a technology designed to read magnetic fields created 
by the current produced by neurons in real-time. The signal obtained will then be analysed with a Machine Learning 
algorithm to classify noun vs. verb condition.

The results show that **we can indeed 'read minds'**. In particular, we observe specific patterns of activation in 
what is known as *Episodic Memory Network* which is thought to process episodic memories.


## Spatio-temporal pattern of significant decoding
<img src="./data/figures/pilot/results/whole-brain.png" alt="pattern" width="600"/>

## Regions of Interest (ROI)
<img src="./data/figures/pilot/results/rois.png" alt="roi" width="400"/>

## Decoding performances in ROIs
<img src="./data/figures/pilot/results/multiple.png" alt="multiple" width="600"/>

# Processing workflow
## Preprocessing specific to pilot data
Pilot data is in BTI format and EEG data (including stimuli triggers) are
recorded separately. Thus, they must be combined into a single FIF file.

The original file structure:
```commandline
meg-data
    config
    data
    hs_file
eeg-data
    t_FVI_01_0001.eeg
    t_FVI_01_0001.vhdr
    t_FVI_01_0001.vhdr.display
    t_FVI_01_0001.vhdr.flt
    t_FVI_01_0001.vhdr.levels
    t_FVI_01_0001.vhdr.mtg
    t_FVI_01_0001.vmrk
```

Use `preprocess_pilot.py` script to convert it into FIF format.


## Top level workflow
* Preprocessing (per subject)
* Dataset generation (per cortical area + sensor space)
* Adding conditions (per condition type)
* MVPA analyses (per analysis)

```mermaid
graph LR;
  preprocess --> dataset;
  dataset --> conditions;
  conditions --> analyses;
```


# Per subject (preprocessing.py)
* **create_directories()**
  * => bunch of directories
* **read_raw()**
  * => raw object
* **downsample()** (optional)
  * => downsampled raw object
* **filter()** (optional)
  * => filtered raw object
* **remove_artifacts()** (optional)
  * => raw object without artifacts
* **epoch()**
  * => epoch file
  * => sensor space data
* **source_localize()**
  * => source space data

```mermaid
graph LR;
    create["create_directories(...)"] --> read_raw;
    read_raw["read_raw(...)"] --> downsample;
    downsample(["downsample(...)"]) --> filter;
    filter(["filter(...)"]) --> remove_artifacts;
    remove_artifacts(["remove_artifacts(...)"]) --> epoch;
    epoch["epoch(...)"] --> source_localize;
    source_localize["source_localize(...)"];
    
    classDef parallel fill: #990000, stroke: #333, stroke-width: 3px, font-size:10px;
    
```
    
# Per cortical area (dataset.py)
* **generate_area_data()**/**generate_area_data_mmap()**


NB: round nodes are optional

