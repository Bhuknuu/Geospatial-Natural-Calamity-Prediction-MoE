<div align="center" style="padding-top: 20px; padding-bottom: 20px;">

## <span style="font-family:'Playfair Display', Georgia, serif; font-weight:500; font-size:1.6em;"> Unified Source Knowledge Base</span>
#### <span style="font-family:'Playfair Display', Georgia, serif; font-weight:200; font-size:1.0em;"> _Indian Terrain-Specific Natural Disaster Prediction Using Geospatial Analysis_</span>

<p style="font-family:'Playfair Display', Georgia, serif; font-weight:250; font-size:1.1em; max-width: 800px; margin: 0 auto; text-align: center;">
Merged technical literature, statistical baselines, preprocessing pipelines, and regional case studies (2020–2025).
</p>

</div>

---

## 1. Natural Calamities Specific to Indian Terrain & Statistical Baseline

India is among the top ten most disaster-prone countries globally due to its unique geo-climatic conditions. Out of 35 states and union territories, 27 are disaster-prone.

### **Vulnerability Distribution by Region**
- **Himalayan Region**: Earthquakes, landslides, flash floods, avalanches, Glacial Lake Outburst Floods (GLOFs).
- **Alluvial Plains (Indus, Ganga, Brahmaputra)**: Extreme monsoon floods, severe river erosion, seismic activity.
- **Western Arid & Semi-Arid Parts (Rajasthan, Gujarat, Maharashtra)**: Prolonged droughts and desertification.
- **Coastal Zones (7,516 km coastline)**: Severe tropical cyclones, storm surges, tsunamis, coastal erosion.
- **Hilly Peninsular Regions (Western Ghats, Eastern Ghats)**: Monsoon-triggered landslides, debris flows.

### **Key National Statistics**
- **Earthquakes**: **58.6%** of the total landmass is prone to moderate to very high-intensity earthquakes.
- **Floods**: Over **40 million hectares** (**12%** of land) is prone to floods and river erosion.
- **Cyclones**: Approximately **5,700 km** of the 7,516 km coastline is vulnerable to cyclones and storm surges.
- **Drought**: **68%** of cultivable land area is vulnerable to drought.
- **Tropical Cyclones**: Average of **5-6 tropical cyclones** annually originating from both the Arabian Sea and the Bay of Bengal.

---

## 2. Deep Learning Models & Architectures in Disaster Prediction

Modern predictive modeling has shifted from classical hydrodynamic and statistical methods to robust deep learning and hybrid architectures tailored to spatial-temporal geospatial data.

### **(a) Flood Prediction (Monsoon-Related & Riverine)**
- **CNN-Swin Transformer Hybrid (SwinFlood)**: Combines coarse-grid hydrodynamic features with fine-grid satellite observations for rapid spatiotemporal flood forecasting.
- **CNN + LSTM Fusion Model**: Merges spatial feature extraction via CNN with temporal sequence modeling via LSTM to forecast monsoon inundation progression.
- **Conv1D & Naive Forecast DL Models**: Deployed for fast time-series river discharge and level pattern prediction in Indian river basins.
- **Vision Transformer (ViT) with CBAM (Convolutional Block Attention Module)**: Achieved **90.75% accuracy** in satellite flood image classification by attending to critical inundation boundaries.
- **Lightweight U-Net**: Applied for rapid flood susceptibility and surface water extent mapping in Kerala, achieving **96.55% overall accuracy** despite extreme class imbalance (only **0.13%** flood pixels).

### **(b) Cyclone Track, Intensity & Storm Surge Prediction**
- **ConvLSTM Framework**: Processes multi-temporal INSAT-3D/3DR and geostationary satellite image sequences for automated cyclone detection, cloud pattern classification, and intensity estimation.
- **Transformer Models**: Used for short-term rolling predictions of tropical cyclone tracks and intensity anomalies.
- **Multi-Task Learning Fusion**: Combines simultaneous track and intensity forecasting models to reduce cumulative error bounds.
- **Dvorak Technique + CNN Automation**: Replaces subjective manual meteorological interpretation with automated CNN feature regression on IR/WV satellite feeds.

### **(c) Glacial Lake Outburst Flood (GLOF) Prediction (Himalayas)**
- **IceWatch Framework**: Integrates multi-sensor optical and SAR data to monitor moraine-dammed glacial lakes in high-altitude Himalayas, tracking area expansion and ice-mass thinning.
- **Deep Convolutional Autoencoders**: Detects subtle structural shifts, ice cliff collapses, and supraglacial lake formations from high-resolution optical imagery.

### **(d) Landslide Susceptibility & Dynamic Triggering (Western Ghats, Nainital, Kerala)**
- **Temporal Convolutional Networks (TCN)**: Applied in Nainital and Himalayan corridors, achieving **R² ~ 0.95** and **F1-score ~ 0.97** for displacement and failure timing forecasting.
- **3D CNN & ResNet Backbones**: Extract volumetric geomorphic features from DEMs, slope, aspect, lithology, and rainfall accumulation grids.
- **Targeted Semantic Segmentation**: Isolates active landslide scars from dense canopy cover using multi-spectral indices (NDVI, NDWI, NDBI).

### **(e) Advanced Geospatial Paradigms**
- **Geospatial Foundation Models (GFMs)**: Pre-trained on petabytes of multi-spectral satellite imagery (e.g., Prithvi, SatlasPretrain) to enable zero-shot or few-shot disaster mapping across Indian agricultural and urban zones.
- **Multi-Temporal InSAR (MT-InSAR) + DL**: Combines phase unwrapping SAR time-series with deep neural networks for millimeter-level land subsidence and landslide creep detection.
- **Geo-LLM Integration**: Leverages Large Language Models fine-tuned on disaster management frameworks (e.g., NDMA guidelines, historical IMD bulletins) to parse multimodal sensor feeds and generate real-time actionable advisory reports.

---

## 3. Satellite and Geospatial Data Sources

### **Indian National Constellations & Portals**
- **ISRO Bhuvan Portal**: National geoportal providing visualization and analysis tools for 47 operational Indian remote sensing satellites.
- **Bhoonidhi & NDEM (National Database for Emergency Management)**: Centralized repositories for near-real-time satellite data and disaster management support systems.
- **IMD DSP (Data Supply Portal) & RSMC New Delhi**: Real-time meteorological observations, weather radar networks, and regional specialized meteorological center advisories for tropical cyclones.
- **MOSDAC (Meteorological and Oceanographic Satellite Data Archiving Centre)**: Provides ocean-atmospheric data payloads from INSAT-3D, INSAT-3DR, Oceansat, and Megha-Tropiques.
- **Key Sensors**:
  - **INSAT-3D / 3DR**: High-frequency (15-minute interval) atmospheric soundings, cloud top temperature, and cyclone monitoring.
  - **Cartosat Series**: High-resolution panchromatic and stereo imagery for detailed post-disaster damage assessment (up to 25cm-1m resolution).
  - **RISAT-1/2 (SAR Series)**: C-band and X-band synthetic aperture radar providing all-weather, day/night imaging essential for monsoon flood and cloud-obscured landslide mapping.

### **International Open Constellations**
- **Landsat Series (USGS/NASA)**: Long-term historical baseline analysis (30m resolution).
- **Sentinel-1 (SAR, ESA)**: C-band SAR imagery (10m resolution) vital for flood extent mapping through clouds.
- **Sentinel-2 (Optical, ESA)**: High-resolution multispectral imagery (10m-20m resolution, 5-day revisit) for optical burn scar and flood analysis.
- **PlanetScope**: Commercial 3m daily optical imagery for rapid tactical damage assessment.
- **NASA Harmonized Landsat Sentinel-2 (HLS)**: Surface reflectance data merged to achieve 1.7-day effective revisit frequency.

---

## 4. Preprocessing and Data Cleaning Pipelines

Rigorous preprocessing is critical to handle noise, cloud cover, atmospheric scattering, and extreme class imbalance in satellite imagery.

### **SAR Data Preprocessing (Sentinel-1 / RISAT)**
- **Thermal Noise Removal**: Eliminates background instrument noise across sub-swaths.
- **Orbit File Calibration**: Applies precise orbit ephemeris data for accurate geolocation.
- **Border Noise Removal**: Strips invalid edge pixels (specifically handling incidence angles between **27° and 45°**).
- **Speckle Filtering**: Uses refined Lee or Gamma MAP filters to suppress granular radar noise while preserving edge sharpness.
- **Radiometric Calibration**: Converts raw digital numbers (DN) to backscatter coefficients ($\sigma^0$ in decibels, dB).
- **Geometric Terrain Correction (RTC)**: Uses SRTM or DEMs to correct foreshortening, layover, and shadow distortions in mountainous terrain (Himalayas, Western Ghats).

### **Optical Data Preprocessing (Sentinel-2 / PlanetScope / Landsat)**
- **Atmospheric Correction**: Converts Top-of-Atmosphere (ToA) radiance to Bottom-of-Atmosphere (BoA) surface reflectance using Sen2Cor / ACOLITE.
- **Cloud & Shadow Masking**: Utilizes S2 L2A Scene Classification Layer (SCL), Planet UDM2 (Usable Data Mask), and Google Cloud Score+ to mask cloud, cirrus, and cloud shadow pixels.
- **Topographic Correction**: Implements Minnaert or C-correction models to adjust for illumination variations on steep mountain slopes.
- **Co-registration**: Sub-pixel alignment of multi-temporal image stacks to prevent false change-detection alarms.

### **Handling Extreme Class Imbalance**
- **Focal Loss & Tversky Loss**: Deployed to address severe background-to-target imbalance (e.g., Kerala flood mapping where flood pixels constitute only **0.13%** of total pixels).
- **Patch-based Spatial Augmentation**: Rotations, scaling, and elastic deformations applied to rare disaster chips.

---

## 5. Datasets and Benchmarks

- **Sen1Floods11**: Global benchmark dataset for flood detection using Sentinel-1 SAR and Sentinel-2 optical imagery, including Indian river basins.
- **MONITRS**: Large-scale dataset (~10,000 FEMA/global damage records) for post-disaster structural damage assessment.
- **ObsHazard-Bench**: Comprehensive benchmark covering 8 categories, 28 sub-categories, across 60+ countries.
- **PANGEA & Geo-Bench**: Standardized geospatial evaluation suites for foundation models.
- **DynamicEarthNet**: High-frequency multi-spectral dataset for land cover and environmental change monitoring.
- **FoMo-Bench & FloodCastBench**: Specialized benchmarks for foundation model performance and flood forecasting robustness.
- **India-Specific Repositories**:
  - **India Flood Atlas (IIT Gandhinagar GitHub)**: Historical flood inundation maps across major Indian river basins.
  - **HydroSense Lab Datasets (IIT Delhi)**: High-resolution hydrological, soil moisture, and flood monitoring datasets for Indian catchments.
  - **Multimodal Monsoon Indian Dataset**: Multi-sensor radar and rain-gauge fusion datasets for Indian summer monsoons.
  - **RiceBaCI (Odisha Cyclone Saline Dataset)**: Specialized dataset tracking cyclone-induced saline water intrusion and agricultural damage in coastal Odisha.

---

## 6. Early Indicators & Multi-Modal Monitoring Signals

| Disaster Type | Primary Early Indicators | Key Sensor / Observational Source |
|---|---|---|
| **Monsoon Floods** | Rapid soil saturation, upstream reservoir level spikes, heavy rainfall accumulation (>150mm/24h), river gauge threshold breaches. | IMD AWS, INSAT-3D rainfall estimates, Sentinel-1 SAR backscatter changes. |
| **Glacial Lake Outburst (GLOF)** | Accelerated lake surface expansion, moraine degradation, supraglacial meltwater pooling, seismic micro-tremors near snouts. | PlanetScope daily optical, Cartosat stereoscopic DEM diffs, IceWatch SAR coherence. |
| **Tropical Cyclones** | Warm sea surface temperatures (>28°C), low-pressure cyclogenesis signatures, spiral cloud band organization, rapid pressure drop. | MOSDAC ocean surface winds (Oceansat), INSAT-3D water vapor channels, RSMC bulletins. |
| **Landslides** | Antecedent cumulative precipitation thresholds (>300mm in 72h), soil moisture saturation indices (API), pre-failure surface displacement. | Sentinel-1 InSAR phase coherence loss, IMD gridded rainfall, local tiltmeters. |
| **Droughts** | Negative NDVI/EVI anomalies, declining soil moisture index (SMI), prolonged meteorological dry spells, land surface temperature (LST) spikes. | Landsat/Sentinel-2 spectral indices, MODIS LST, SMAP soil moisture. |
| **Forest Fires** | Thermal anomalies (Active Fires/Hotspots), high vapor pressure deficit (VPD), low fuel moisture content, wind speed vectors. | MODIS/VIIRS active fire products, Sentinel-2 shortwave infrared (SWIR) bands. |

---

## 7. Existing Solutions & Operational Platforms

- **ISRO / NRSC Operational Systems**: National institutional frameworks delivering real-time flood inundation mapping, cyclone tracking, and drought assessment to central and state disaster management authorities (NDMA/SDMA).
- **Google Flood Hub**: AI-powered river flood forecasting platform operational across major Indian river basins (Ganga, Brahmaputra, Godavari) providing up to 7-day advance warnings.
- **MIT / IBM Global Fire / Flood Trackers**: Advanced machine learning pipelines for automated burn scar and flood hazard delineation.
- **CEMS (Copernicus Emergency Management Service)**: Rapid mapping activation service providing satellite-derived vector boundaries during major international and regional calamities.
- **FireSat / NASA FIRMS**: Near-real-time satellite active fire monitoring and radiative power estimation.

---

## 8. Regional Case Studies & Indian Project Benchmarks

| Project / Region | Hazard Type | Primary Model / Technique | Performance / Key Metric |
|---|---|---|---|
| **Nainital / Himalayan Corridors** | Landslides & Slope Instability | Temporal Convolutional Networks (TCN) | R² ~ 0.95, F1-Score ~ 0.97 |
| **Kerala Flood Mapping** | Monsoon Floods | Lightweight U-Net with Focal Loss | 96.55% overall accuracy (handling 0.13% class imbalance) |
| **Ghatal, West Bengal** | Riverine Inundation | U-Net trained on Sentinel-1 SAR | High precision in cloud-obscured deltaic flooding |
| **Kishtwar, Jammu & Kashmir** | Multi-Hazard Risk Assessment | Geo-LLM + Multimodal Geospatial Fusion | Automated real-time advisory generation from sensor feeds |
| **Bay of Bengal & Arabian Sea** | Tropical Cyclones | ConvLSTM & Dvorak CNN Automation | Sub-hourly intensity regression & track forecasting |

---

## 9. Comprehensive Literature Reviews & Foundational References

1. **"From Spectral Indices to Foundation Models in Remote Sensing"** (~200+ studies reviewed): Traces the evolution of disaster monitoring from simple NDVI/NDWI thresholds to self-supervised geospatial foundation models.
2. **"Deep Learning for Potential Landslide Identification: A Comprehensive Review"** (~400+ studies, 2020–2025): Catalogs spatial modeling architectures, DEM integration, and trigger feature engineering.
3. **"From RNNs to Transformers in Hydrological Forecasting"** (HESS): Evaluates sequence-to-sequence neural network performance against traditional hydrological routing models across Asian river basins.
4. **"Interpretable Deep Learning for Nation-Wide Flood Inundation Mapping in India"**: Synthesizes explainable AI (XAI) techniques applied to CNN-Swin transformer hybrids for transparent disaster decision support.
5. **"Multi-Temporal InSAR and Machine Learning for Geohazard Monitoring"**: Reviews phase unwrapping interferometry combined with gradient boosting and deep neural nets for slow-moving landslide and subsidence tracking.

---

## 10. Summary Recommendations for SIH Project Implementation

| Pipeline Stage | Recommended Tool / Framework | Best Practice / Threshold |
|---|---|---|
| **Data Acquisition** | Google Earth Engine (GEE), ASF Vertex HyP3, OpenSARLab (GPU) | Automate ingestion of Sentinel-1 RTC and Sentinel-2 L2A via STAC APIs. |
| **SAR Preprocessing** | SNAP Graph Processing / HyP3 API | Apply precise orbit files, border noise removal (incidence 27–45°), Lee speckle filter (5x5), and terrain correction. |
| **Optical Preprocessing** | Sen2Cor / Cloud Score+ | Mask clouds and cirrus using SCL / UDM2; apply C-correction for rugged mountain topography. |
| **Model Architecture** | SwinFlood / ConvLSTM / U-Net (Focal Loss) | Use U-Net with Focal Loss for binary flood/landslide masks; ConvLSTM for sequence tracking (cyclones). |
| **Validation & Metrics** | F1-Score, Intersection over Union (IoU), R² | Emphasize F1-Score and IoU over raw accuracy due to extreme spatial class imbalance in disaster datasets. |
| **Deployment** | FastAPI backend + Streamlit / Web GUI | Deploy inference pipelines with interactive geospatial map widgets (Folium / Leaflet) for real-time advisory generation. |

---

## 11. Comprehensive Dataset Collection & Direct URLs

### **11.1 Flood**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Rainfall** | IMD Gridded Rainfall Data (0.25°×0.25°) | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | India Meteorological Department, daily/gridded precipitation |
| **Rainfall** | CHIRPS (Climate Hazards Center InfraRed Precipitation with Station data) | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) | Near-global, 30+ year daily precipitation record; also on [Google Earth Engine](https://developers.google.cn/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY) |
| **Rainfall** | CHIRPS v3 | [https://www.chc.ucsb.edu/data/chirps3](https://www.chc.ucsb.edu/data/chirps3) | 4× more gauge sources than v2 |
| **River discharge / Water level** | India-WRIS (Water Resources Information System) | [https://india-wris.gov.in/](https://india-wris.gov.in/) | National hydrological data portal |
| **River discharge / Water level** | CWC (Central Water Commission) Flood Forecasting | [http://www.cwc.gov.in/](http://www.cwc.gov.in/) | Real-time flood forecasting sites across Indian river basins |
| **Terrain / DEM** | SRTM (Shuttle Radar Topography Mission) | [https://www.earthdata.nasa.gov/data/instruments/srtm](https://www.earthdata.nasa.gov/data/instruments/srtm) | 30m elevation resolution, global coverage |
| **Terrain / DEM** | Bhuvan DEM (ISRO) | [https://bhuvan.isro.gov.in/](https://bhuvan.isro.gov.in/) | Indian-specific 30m DEM products |
| **Historical flood extents** | India Flood Atlas (IIT Gandhinagar) | [https://github.com/IITGN/India-Flood-Atlas](https://github.com/IITGN/India-Flood-Atlas) | Curated historical flood inundation maps |
| **Historical flood extents** | NASA MODIS Flood Mapping | [https://modis.gsfc.nasa.gov/](https://modis.gsfc.nasa.gov/) | Daily global flood detection via MODIS Terra/Aqua |
| **Historical flood extents** | Kaggle India Flood Inventory | [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) | Search "India Flood Inventory" on Kaggle |
| **Satellite imagery (flood)** | Sentinel-1 SAR (ESA) | [https://scihub.copernicus.eu/](https://scihub.copernicus.eu/) | C-band SAR for cloud-penetrating flood mapping |
| **Satellite imagery (flood)** | Sentinel-2 Optical (ESA) | [https://scihub.copernicus.eu/](https://scihub.copernicus.eu/) | Multispectral optical for water extent (NDWI) |

### **11.2 Landslide**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Rainfall thresholds** | IMD Gridded Rainfall | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | 0.25°×0.25° gridded precipitation data |
| **Rainfall thresholds** | CHIRPS | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) | Satellite-rain gauge merged precipitation |
| **Slope, soil, land-use** | Bhuvan (ISRO) | [https://bhuvan.isro.gov.in/](https://bhuvan.isro.gov.in/) | Indian topography, soil, and land-use layers |
| **Slope, soil, land-use** | GSI (Geological Survey of India) Landslide Susceptibility Maps | [https://gsi.gov.in/](https://gsi.gov.in/) | National landslide susceptibility zoning maps |
| **Historical landslide inventory** | GSI Bhukosh | [https://bhuvan.isro.gov.in/](https://bhuvan.isro.gov.in/) | Indian landslide inventory database (via Bhuvan) |
| **Historical landslide inventory** | NASA Global Landslide Catalog | [https://global landslide catalog.evs.larc.nasa.gov/](https://global-landslide-catalog.evs.larc.nasa.gov/) | Global landslide event catalog with timestamps |
| **Satellite imagery** | Sentinel-1 SAR (InSAR) | [https://scihub.copernicus.eu/](https://scihub.copernicus.eu/) | Ground deformation monitoring via interferometry |
| **Satellite imagery** | Cartosat (ISRO) | [https://bhuvan.isro.gov.in/](https://bhuvan.isro.gov.in/) | High-resolution stereo imagery for slope angle derivation |

### **11.3 Droughts**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Rainfall / Precipitation** | IMD Gridded Rainfall | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | 0.25°×0.25° daily/monthly precipitation |
| **Rainfall / Precipitation** | CHIRPS | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) | Station + satellite merged precipitation |
| **Vegetation health** | MODIS NDVI / VHI | [https://modis.gsfc.nasa.gov/](https://modis.gsfc.nasa.gov/) | MODIS Terra/Aqua vegetation indices |
| **Vegetation health** | Sentinel-2 NDVI / NDWI | [https://scihub.copernicus.eu/](https://scihub.copernicus.eu/) | High-resolution vegetation water content |
| **Soil moisture** | NRSC Soil Moisture Products | [https://nresdc.nrsc.gov.in/](https://nresdc.nrsc.gov.in/) | ISRO National Remote Sensing Centre soil moisture |
| **Soil moisture** | GLDAS (Global Land Data Assimilation System) | [https://gldas.gsfc.nasa.gov/](https://gldas.gsfc.nasa.gov/) | NASA land surface model soil moisture |
| **Soil moisture** | SMAP (Soil Moisture Active Passive) | [https://smap.jpl.nasa.gov/](https://smap.jpl.nasa.gov/) | NASA satellite-derived global soil moisture |
| **Agromet indices** | CRIDA (Central Research Institute for Dryland Agriculture) | [https://crida.res.in/](https://crida.res.in/) | Dryland agriculture research, SPI data |
| **Agromet indices** | IMD Standardized Precipitation Index (SPI) | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | Drought monitoring indices |
| **Historical drought records** | Manual for Drought Management (Ministry of Agriculture) | [Department of Agriculture & Farmers Welfare](https://dafw.gov.in/) | District-level drought declarations |
| **Historical drought records** | DEWS (Drought Early Warning System) | [https://dews.gov.in/](https://dews.gov.in/) | Integrated drought early warning portal |

### **11.4 Earthquakes**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Seismicity catalogs** | National Center for Seismology (NCS) India Earthquake Catalog | [https://ncs.gov.in/](https://ncs.gov.in/) | Indian earthquake event catalog |
| **Seismicity catalogs** | USGS Earthquake Hazards Program | [https://earthquake.usgs.gov/](https://earthquake.usgs.gov/) | Global seismicity data (M2.5+) |
| **Seismicity catalogs** | ISC (International Seismological Centre) Bulletin | [https://www.isc.ac.uk/](https://www.isc.ac.uk/) | Comprehensive global earthquake catalog |
| **Seismicity catalogs** | USGS ANSS ComCat | [https://earthquake.usgs.gov/fdsnws/event/1/](https://earthquake.usgs.gov/fdsnws/event/1/) | Advanced National Seismic System Comprehensive Catalog (API) |
| **Fault lines / Seismic zoning** | Bureau of Indian Standards (IS 1893) | [https://bis.gov.in/](https://bis.gov.in/) | Indian seismic zone maps (IS 1893:2015) |
| **Fault lines / Seismic zoning** | GSI Active Fault Mapping | [https://gsi.gov.in/](https://gsi.gov.in/) | Active fault line mapping across India |
| **Ground motion / Soil** | Bhuvan Geological Maps | [https://bhuvan.isro.gov.in/](https://bhuvan.isro.gov.in/) | Indian geological and soil type maps |
| **Ground motion / Soil** | NCS Strong Motion Network | [https://ncs.gov.in/](https://ncs.gov.in/) | Strong motion accelerograph data |
| **Historical events** | USGS ANSS ComCat | [https://earthquake.usgs.gov/](https://earthquake.usgs.gov/) | Historical earthquake archive |
| **Historical events** | NCS Historical Earthquake Archive | [https://ncs.gov.in/](https://ncs.gov.in/) | Indian historical earthquake records |

### **11.5 Cyclone**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Track / Intensity** | IMD RSMC New Delhi Cyclone Bulletins | [https://mausam.imdpune.gov.in/cyclone.htm](https://mausam.imdpune.gov.in/cyclone.htm) | Official North Indian Ocean cyclone warnings |
| **Track / Intensity** | IBTrACS (International Best Track Archive for Climate Stewardship) | [https://www.ncei.noaa.gov/products/sea-surface-temperature/ibtracs/](https://www.ncei.noaa.gov/products/sea-surface-temperature/ibtracs/) | Global unified tropical cyclone best-track dataset |
| **Satellite imagery** | INSAT-3D / 3DR (ISRO) | [https://www.isro.gov.in/](https://www.isro.gov.in/) | Geostationary satellite imagery for cyclone monitoring |
| **Satellite imagery** | NOAA / NASA Satellite Cyclone Imagery | [https://hurricane.science.noaa.gov/](https://hurricane.science.noaa.gov/) | Hurricane/tropical cyclone satellite archives |
| **Ocean parameters** | INCOIS (Indian National Centre for Ocean Information Services) | [https://incois.gov.in/](https://incois.gov.in/) | Sea surface temperature, wave height, ocean currents |
| **Historical cyclones** | IMD Cyclone eAtlas | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | Historical North Indian Ocean cyclone database |
| **Historical cyclones** | JTWC (Joint Typhoon Warning Center) | [https://www.metoc.navy.mil/jtwc/jtwc.html](https://www.metoc.navy.mil/jtwc/jtwc.html) | Global tropical cyclone warnings and archives |
| **Historical cyclones** | CNMOC (Joint Typhoon Warning Center) | [https://www.cnmoc.usff.navy.mil/](https://www.cnmoc.usff.navy.mil/) | U.S. Navy tropical cyclone center archives |

### **11.6 Forest Fires**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Active fire detection** | NASA MODIS/VIIRS FIRMS | [https://firms.modaps.eosdis.nasa.gov/](https://firms.modaps.eosdis.nasa.gov/) | Fire Information for Resource Management System (near-real-time) |
| **Active fire detection** | FSI (Forest Survey of India) Fire Alert System | [https://fsi.nic.in/](https://fsi.nic.in/) | Indian national forest fire alerts and monitoring |
| **Fuel / Vegetation** | Bhuvan Forest Cover Maps | [https://bhuvan.isro.gov.in/](https://bhuvan.isro.gov.in/) | Indian forest cover and vegetation density maps |
| **Fuel / Vegetation** | NDVI-based Fuel Load Estimation | [https://modis.gsfc.nasa.gov/](https://modis.gsfc.nasa.gov/) | MODIS-derived vegetation density for fuel estimation |
| **Weather drivers** | IMD Temperature/Humidity/Wind | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | IMD meteorological data for Fire Weather Index |
| **Weather drivers** | Fire Weather Index (FWI) | [https://www.efdi.org/](https://www.efdi.org/) | European Fire Database / global FWI inputs |
| **Historical fire records** | FSI Annual Forest Fire Reports | [https://fsi.nic.in/](https://fsi.nic.in/) | Yearly Indian forest fire incidence reports |
| **Historical fire records** | GFED (Global Fire Emissions Database) | [https://www.globalfiredata.org/](https://www.globalfiredata.org/) | Global historical fire emissions and burn area |

### **11.7 Heatwaves**

| Data Type | Dataset / Source | URL | Notes |
|---|---|---|---|
| **Temperature** | IMD Surface Weather Data | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | IMD gridded and station-level temperature data |
| **Temperature** | MODIS LST (Land Surface Temperature) | [https://modis.gsfc.nasa.gov/](https://modis.gsfc.nasa.gov/) | Daily global land surface temperature |
| **Temperature** | ERA5 Reanalysis (ECMWF) | [https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) | Hourly global reanalysis temperature/humidity/wind |
| **Vegetation / Urban** | Sentinel-2 Land Surface Temperature | [https://scihub.copernicus.eu/](https://scihub.copernicus.eu/) | High-resolution LST for urban heat island analysis |
| **Historical records** | IMD Heatwave Database | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | Indian heatwave event catalog |
| **Historical records** | WHO Global Heat-Health Information Network | [https://www.who.int/teams/environment-climate-change-and-health/health-protection/heat-health](https://www.who.int/teams/environment-climate-change-and-health/health-protection/heat-health) | Global heatwave health impact data |

---

## 12. Geospatial Difference Detection Analysis

**Question: Is it possible to find differences between disaster types using geospatial means?**

**Answer: Yes, definitively.** Geospatial analysis provides powerful tools to differentiate and compare disaster types through multi-spectral, multi-temporal satellite data. Key approaches include:

### **12.1 Spectral Signature Differentiation**
- **Floods** vs **Landslides**: Floods show distinct NDWI (Normalized Difference Water Index > 0.3), while landslides show spectral signatures of exposed soil/scree (low NDVI, high shortwave infrared reflectance).
- **Droughts** vs **Normal Vegetation**: Drought regions show sustained negative NDVI anomalies and elevated Land Surface Temperature (LST) compared to surrounding areas via MODIS/NOAA data.
- **Forest Fires** vs **Floods**: Fire scars exhibit high SWIR reflectance and low NDVI post-event (recoverable via Sentinel-2 SWIR bands), while flood extents show high NIR/SWIR ratio in visible imagery.
- **Earthquakes** vs **Landslides**: Earthquake impacts (ground fissures, liquefaction) detectable via InSAR deformation (>cm-scale); landslides detectable via morphological DEM analysis (slope angle >30°) and SAR coherence loss.

### **12.2 Temporal Differentiation**
- **Cyclones** vs **Monsoon Floods**: Cyclone signatures on INSAT-3D/3DR show organized spiral cloud bands with warm-core SST anomalies (>28°C); monsoon floods show broad regional precipitation patterns (>150mm/24h) without organized cyclonic structure.
- **GLOFs** vs **Regular Floods**: GLOFs show sudden lake area expansion (via PlanetScope daily 3m imagery) preceded by moraine instability (InSAR deformation); regular floods show gradual riverine spread.

### **12.3 Multi-Sensor Fusion for Discrimination**
- **SAR (Sentinel-1)** differentiates flood extent (high backscatter from water) from urban areas (double-bounce) via polarization analysis.
- **Optical (Sentinel-2 + PlanetScope 3m)** differentiates landslide debris from flood sediment via texture analysis and NDVI thresholds.
- **Multi-temporal InSAR (MT-InSAR)** differentiates earthquake coseismic displacement (sudden cm-scale step) from landslide creep (mm/year deformation rate).

### **12.4 Practical Geospatial Difference Workflow**
1. **Data Acquisition**: Download Sentinel-1 (SAR) + Sentinel-2 (Optical) via Copernicus Open Access Hub
2. **Preprocessing**: Apply RTC (SAR), L2A masks (optical), cloud masking (Cloud Score+)
3. **Index Calculation**: Compute NDWI (flood), NDBI (built-up), NDVI (vegetation), VHI (drought), NDMI (fire/drought)
4. **Classification**: Train Random Forest / U-Net on labeled disaster pixels to classify disaster type
5. **Change Detection**: Use SAR interferometry (InSAR) to detect deformation, optical differencing to detect extent changes
6. **Validation**: Cross-reference with IMD/NCS/GSI ground truth catalogs

**Conclusion: Geospatial means can reliably discriminate between disaster types using spectral, temporal, and deformation signatures from multi-source satellite data (SAR + optical + thermal).**

---

## 13. Data Collection Forms & Templates

**Forms for structured disaster data collection:**

### **13.1 Flood Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Event Date | YYYY-MM-DD | 2024-07-15 |
| Region / District | Indian administrative unit | Uttarakhand |
| Disaster Type | Flood / Landslide / GLOF | Monsoon Flood |
| Rainfall (mm) | IMD / CHIRPS cumulative | 250mm / 72h |
| River Gauge Level | CWC station reading | 12.5m above normal |
| Flood Extent (km²) | Sentinel-1 SAR derived | 1,250 km² |
| Population Affected | NDMA / district report | 450,000 |
| Economic Damage (₹) | State Disaster Response Fund | ₹2,500 crore |
| Satellite Imagery Source | Sentinel-1 / Sentinel-2 / Cartosat | Sentinel-1 IW GRD |
| Preprocessing Applied | Border noise removal, speckle filter, RTC | 27-45° incidence, 50m focal mean |

### **13.2 Landslide Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Event Date | YYYY-MM-DD | 2024-08-03 |
| Region / District | Uttarakhand / Kerala / Western Ghats | Nainital |
| Slope Angle (°) | DEM-derived | 35° |
| Rainfall (mm) | IMD / CHIRPS cumulative | 180mm / 48h |
| Deformation Rate (mm/yr) | TCN / InSAR measurement | 12mm/yr (pre-failure) |
| F1-Score / R² | Model performance | R² ~0.95, F1 ~0.97 |
| Failure Trigger | Rainfall threshold / earthquake | 150mm/72h threshold exceeded |
| Satellite Imagery Source | Sentinel-1 / Cartosat / PlanetScope | Cartosat-2 (0.6m) |

### **13.3 Cyclone Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Cyclone Name | IMD designated name | Cyclone "Remal" |
| Basin | Bay of Bengal / Arabian Sea | Bay of Bengal |
| Category | IMD classification | Very Severe Cyclonic Storm |
| Peak Wind Speed (km/h) | IMD RSMC estimate | 140 km/h |
| Minimum Pressure (hPa) | Central pressure | 978 hPa |
| Landfall Location | Lat/Lon coordinates | 21.5°N, 87.2°E |
| Track Data | IBTrACS / IMD bulletin | 2024-05-24 to 2024-05-28 |
| SST at Landfall (°C) | INCOIS data | 29.5°C |
| Satellite Source | INSAT-3D / MODIS / ASCAT | INSAT-3D imager |
| ConvLSTM Model Output | Track / intensity prediction | 12-hr lead, 85% accuracy |

### **13.4 Drought Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Event Year | YYYY | 2024 |
| Region / District | Maharashtra / Rajasthan | Marathwada |
| SPI Value | IMD Standardized Precipitation Index | SPI-6 = -2.1 (severe) |
| NDVI Anomaly | MODIS NDVI deviation | -35% vs 10-yr mean |
| Soil Moisture (%) | NRSC / GLDAS / SMAP | 12% (vs 30% normal) |
| Crop Affected | Agricultural impact | Rice, cotton |
| DEWS Alert Level | Drought Early Warning System | Alert Level 3 (Severe) |
| CRIDA Assessment | CRIDA district report | Drought declaration issued |

### **13.5 Earthquake Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Event Date | YYYY-MM-DD | 2024-06-18 |
| Magnitude (Mw) | USGS / NCS catalog | 6.2 Mw |
| Depth (km) | Hypocenter depth | 15 km |
| Epicenter Lat/Lon | WGS84 coordinates | 34.1°N, 77.5°E |
| Seismic Zone | IS 1893 zone factor | Zone IV |
| Peak Ground Acceleration (g) | NCS strong motion | 0.35g |
| Liquefaction / Fissures | Observed surface effects | Yes, 2 km fissure |
| Landslide Trigger | Yes / No | Yes (Kashmir region) |
| InSAR Deformation (cm) | Sentinel-1 LOS displacement | 15 cm vertical |

### **13.6 Forest Fire Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Event Date | YYYY-MM-DD | 2024-03-15 |
| Region / Forest Division | Uttarakhand / Himachal Pradesh | Corbett National Park |
| Fire Radiative Power (MW) | MODIS/VIIRS FIRMS | 45 MW |
| NDVI Pre-Fire | Sentinel-2 | 0.65 |
| NDVI Post-Fire | Sentinel-2 | 0.12 |
| Burn Area (km²) | GFED / FSI mapping | 85 km² |
| FWI (Fire Weather Index) | IMD / EFDRI | 28 (High) |
| Satellite Source | MODIS (1km) / VIIRS (375m) | VIIRS Active Fire |

### **13.7 Heatwave Event Data Collection Form**
| Field | Description | Example |
|---|---|---|
| Event Date | YYYY-MM-DD | 2024-05-28 |
| Region / District | Delhi / Andhra Pradesh | Delhi NCR |
| Max Temperature (°C) | IMD station | 48.5°C |
| Min Temperature (°C) | IMD station | 36.2°C |
| LST (MODIS) | Land Surface Temperature | 52.3°C |
| Duration (days) | Continuous heatwave days | 7 days |
| NDVI (Sentinel-2) | Urban vegetation cover | 0.08 (low) |
| UHI Index | Urban Heat Island magnitude | +3.5°C |
| Health Impact | IMD/WHO mortality report | 156 reported deaths |

### **13.8 Multi-Disaster Cross-Comparison Form**
| Field | Flood | Landslide | Cyclone | Drought | Earthquake | Fire | Heatwave |
|---|---|---|---|---|---|---|---|
| **Primary Satellite** | Sentinel-1 SAR / Sentinel-2 | Cartosat / Sentinel-1 | INSAT-3D / MODIS | MODIS / Sentinel-2 | Sentinel-1 InSAR | MODIS FIRMS / Sentinel-2 | MODIS LST / Sentinel-2 |
| **Key Index** | NDWI > 0.3 | NDBI, slope > 30° | SST > 28°C, spiral bands | NDVI anomaly < -30% | InSAR deformation > cm | NBR (Burn Ratio) > 0.66 | LST > 45°C, NDVI < 0.15 |
| **Primary Dataset** | CHIRPS, India-WRIS | GSI Bhukosh, Bhuvan DEM | IBTrACS, INCOIS | GLDAS, SMAP, NRSC | USGS ANSS, NCS | FSI, GFED | IMD, MODIS LST |
| **Model** | U-Net (96.55%) | TCN (R²~0.95) | ConvLSTM | LSTM | Geo-LLM | CNN | Random Forest |
| **Temporal Scale** | Hours-days | Days-weeks | Hours-days | Months-years | Seconds-minutes | Hours-days | Days-weeks |
| **Spatial Scale** | River basin / District | Slope unit / Block | Cyclone track | District / State | Epicentral radius | Forest division | City / District |

---

## 14. Summary & Next Steps for SIH Project

1. **Acquire** all listed datasets using the provided URLs (start with CHIRPS rainfall, Sentinel-1/2 imagery, and SRTM DEM)
2. **Preprocess** using the documented pipelines (SAR: border noise removal 27-45°, 50m focal mean; Optical: L2A masks, Cloud Score+, Minnaert correction)
3. **Address class imbalance** using Focal Loss (target: 0.13% flood pixels, rare landslide events)
4. **Train** and validate models using the benchmarks (F1-score, IoU, R² — not raw accuracy)
5. **Apply** geospatial difference detection (Section 12) to discriminate disaster types
6. **Collect** ground truth data using the standardized forms (Section 13) for model validation
7. **Deploy** with FastAPI + Streamlit using the recommendations from Section 10

**Primary Data Sources Priority:**
1. **Immediate**: CHIRPS (rainfall), Sentinel-1 SAR (floods/landslides), SRTM (terrain), INSAT-3D (cyclones)
2. **Secondary**: MODIS (vegetation/fire/LST), SMAP (soil moisture), GSI/NRSC (Indian-specific)
3. **Tertiary**: PlanetScope (3m daily), Cartosat (0.6m high-res), GLDAS/SMAP (soil moisture)

