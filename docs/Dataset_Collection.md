<div align="center" style="padding-top: 20px; padding-bottom: 20px;">

## <span style="font-family:'Playfair Display', Georgia, serif; font-weight:500; font-size:1.6em;"> Comprehensive Dataset Collection</span>
#### <span style="font-family:'Playfair Display', Georgia, serif; font-weight:200; font-size:1.0em;"> _for Indian Natural Disaster & Geospatial Image Analysis (2020–2025)_</span>

<p style="font-family:'Playfair Display', Georgia, serif; font-weight:250; font-size:1.1em; max-width: 800px; margin: 0 auto; text-align: center;">
Verified URLs, satellite endpoints, and geospatial features for Indian subcontinent hazards.
</p>

</div>

---

## 11.1 Flood

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Rainfall** | IMD Gridded Rainfall Data (0.25°×0.25°) | [https://www.imdpune.gov.in/cmpg/griddata/rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/griddata/rainfall_25_Bin.html) | Daily 0.25° × 0.25° gridded precipitation |
| **Rainfall** | CHIRPS v2.0 | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) | 30+ year daily, 0.05°, also on [GEE](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY) |
| **River discharge / Water level** | India-WRIS | [https://indiawris.gov.in/](https://indiawris.gov.in/) | Real-time hydrological data portal (gauge-level) |
| **River discharge / Water level** | CWC Flood Forecasting | [https://cwc.gov.in/](https://cwc.gov.in/) | River level/discharge across Indian basins |
| **Terrain / DEM** | SRTM 30m | [https://www.earthdata.nasa.gov/data/instruments/srtm](https://www.earthdata.nasa.gov/data/instruments/srtm) | 30m elevation, global coverage |
| **Terrain / DEM** | Bhuvan DEM (ISRO) | [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) | Indian-specific 30m DEM products |
| **Historical flood extents** | India Flood Atlas (IITGN) | [https://github.com/IITGN/India-Flood-Atlas](https://github.com/IITGN/India-Flood-Atlas) | Curated historical flood inundation maps |
| **Historical flood extents** | NASA MODIS Flood Mapping | [https://floodmap.modaps.eosdis.nasa.gov/](https://floodmap.modaps.eosdis.nasa.gov/) | Global MODIS-based flood mapping |
| **Geospatial means** | Sentinel-1 SAR Water Index | [COPERNICUS_S1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1) | SAR backscatter for water extent |
| **Geospatial means** | NDWI / MNDWI from Sentinel-2 | [COPERNICUS_S2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) | Water indices for flood delineation |

---

## 11.2 Landslide

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Rainfall thresholds** | IMD Gridded Rainfall | [https://www.imdpune.gov.in/cmpg/griddata/rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/griddata/rainfall_25_Bin.html) | Critical rainfall thresholds for triggering |
| **Rainfall thresholds** | CHIRPS | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) | Satellite-derived rainfall for early warning |
| **Slope / Soil / Land-use** | Bhuvan (ISRO) | [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) | High-res slope, soil, land-use |
| **Slope / Soil / Land-use** | GSI Landslide Susceptibility Maps | [https://gsi.gov.in/](https://gsi.gov.in/) | Geological Survey of India maps |
| **Historical inventory** | GSI Bhukosh | [https://bhukosh.gsi.gov.in/](https://bhukosh.gsi.gov.in/) | National landslide inventory |
| **Historical inventory** | NASA Global Landslide Catalog | [https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Expanded/GH2D-9p94](https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Expanded/GH2D-9p94) | Global landslide events with geospatial metadata |
| **Geospatial means** | InSAR Ground Motion | [GSSC](https://gssc.esa.int/) | Sentinel-1 InSAR for cm-level ground displacement |
| **Geospatial means** | NDWI / NDVI Change Detection | [GEE](https://developers.google.com/earth-engine/) | Pre/post-event vegetation/water index change |

---

## 11.3 Droughts

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Rainfall / Precipitation** | IMD Gridded Rainfall (0.25°) | [https://www.imdpune.gov.in/cmpg/griddata/rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/griddata/rainfall_25_Bin.html) | Standardized Precipitation Index (SPI) |
| **Rainfall / Precipitation** | CHIRPS | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) | Satellite rainfall for drought monitoring |
| **Vegetation health** | MODIS NDVI / VHI | [https://modis.gsfc.nasa.gov/data/dataprod/ndvi.php](https://modis.gsfc.nasa.gov/data/dataprod/ndvi.php) | Vegetation Health Index |
| **Vegetation health** | Sentinel-2 NDVI / NDWI | [COPERNICUS_S2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) | 10m high-res vegetation indices |
| **Soil moisture** | NRSC Soil Moisture | [https://www.nrsc.gov.in/](https://www.nrsc.gov.in/) | National Remote Sensing Centre products |
| **Soil moisture** | GLDAS | [https://ldas.gsfc.nasa.gov/gldas](https://ldas.gsfc.nasa.gov/gldas) | Global Land Data Assimilation System |
| **Soil moisture** | SMAP (NASA) | [https://smap.jpl.nasa.gov/](https://smap.jpl.nasa.gov/) | Soil Moisture Active Passive satellite |
| **Agromet indices** | CRIDA | [https://www.crida.in/](https://www.crida.in/) | Central Research Institute for Dryland Agriculture |
| **Agromet indices** | IMD SPI Data | [https://www.imdpune.gov.in/](https://www.imdpune.gov.in/) | Drought monitoring and classification |
| **Historical records** | MoAFW Manual for Drought Management | [https://agriculture.gov.in/](https://agriculture.gov.in/) | District-level drought declarations |
| **Historical records** | Drought Early Warning System (DEWS) | [https://www.dews.in/](https://www.dews.in/) | Integrated drought monitoring portal |
| **Geospatial means** | NDVI Trend Analysis | [GEE](https://developers.google.com/earth-engine/) | Vegetation health decline over time |
| **Geospatial means** | SMAP Soil Moisture Anomaly | [NASA](https://nssdc.gsfc.nasa.gov/) | Soil moisture deficit mapping |

---

## 11.4 Earthquakes

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Seismicity catalogs** | NCS India Earthquake Catalog | [https://ncs.gov.in/](https://ncs.gov.in/) | Indian earthquake event catalog |
| **Seismicity catalogs** | USGS Earthquake Hazards Program | [https://earthquake.usgs.gov/](https://earthquake.usgs.gov/) | Global seismicity data (M2.5+) |
| **Seismicity catalogs** | ISC Bulletin | [http://www.isc.ac.uk/](http://www.isc.ac.uk/) | Comprehensive global earthquake catalog |
| **Seismicity catalogs** | USGS ANSS ComCat API | [https://earthquake.usgs.gov/fdsnws/event/1/](https://earthquake.usgs.gov/fdsnws/event/1/) | Advanced National Seismic System Catalog |
| **Fault lines / Seismic zoning** | BIS IS 1893 | [https://bis.gov.in/](https://bis.gov.in/) | Indian seismic zone maps (IS 1893:2015) |
| **Fault lines / Seismic zoning** | GSI Active Fault Mapping | [https://gsi.gov.in/](https://gsi.gov.in/) | Active fault line mapping across India |
| **Ground motion / Soil** | Bhuvan Geological Maps | [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) | Indian geological and soil type maps |
| **Ground motion / Soil** | NCS Strong Motion Network | [https://ncs.gov.in/](https://ncs.gov.in/) | Strong motion accelerograph data |
| **Historical events** | USGS ANSS ComCat | [https://earthquake.usgs.gov/](https://earthquake.usgs.gov/) | Historical earthquake archive |
| **Historical events** | NCS Historical Archive | [https://ncs.gov.in/](https://ncs.gov.in/) | Indian historical earthquake records |
| **Geospatial means** | GPS Network (GARUD) | [NGDC](https://ngdc.noaa.gov/) | Crustal motion monitoring |
| **Geospatial means** | InSAR Ground Displacement | [GSSC](https://gssc.esa.int/) | Sentinel-1 InSAR for pre-seismic strain |

---

## 11.5 Cyclone

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Track / Intensity** | IMD RSMC New Delhi | [http://www.rsmcnewdelhi.imd.gov.in/](http://www.rsmcnewdelhi.imd.gov.in/) | Official North Indian Ocean cyclone bulletins |
| **Track / Intensity** | IBTrACS | [https://www.ncei.noaa.gov/products/sea-surface-temperature/ibtracs/](https://www.ncei.noaa.gov/products/sea-surface-temperature/ibtracs/) | Global tropical cyclone best-track dataset |
| **Satellite imagery** | INSAT-3D / 3DR (MOSDAC) | [https://mosdac.gov.in/](https://mosdac.gov.in/) | Geostationary satellite imagery for cyclone monitoring |
| **Satellite imagery** | NOAA/NASA Cyclone Imagery | [https://hurricane.science.noaa.gov/](https://hurricane.science.noaa.gov/) | Hurricane/tropical cyclone satellite archives |
| **Ocean parameters** | INCOIS | [https://incois.gov.in/](https://incois.gov.in/) | Sea surface temperature, wave height, ocean currents |
| **Historical cyclones** | IMD Cyclone eAtlas | [https://mausam.imdpune.gov.in/cyclone_atlas/](https://mausam.imdpune.gov.in/cyclone_atlas/) | Historical North Indian Ocean cyclone database |
| **Historical cyclones** | JTWC | [https://www.metoc.navy.mil/jtwc/jtwc.html](https://www.metoc.navy.mil/jtwc/jtwc.html) | Global tropical cyclone warnings |
| **Historical cyclones** | RSMC Chennai (NCICS) | [https://ncics.org.in/](https://ncics.org.in/) | Cyclone tracking and advisory archives |
| **Geospatial means** | Dvorak Technique (automated) | [NASA](https://gpm.nasa.gov/missions/dvorak) | Satellite-based intensity estimation |
| **Geospatial means** | SAR Ocean Surface Wind | [ASF](https://asf.alaska.edu/) | Sentinel-1 SAR wind speed mapping |

---

## 11.6 Forest Fires

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Active fire detection** | NASA MODIS/VIIRS FIRMS | [https://firms.modaps.eosdis.nasa.gov/](https://firms.modaps.eosdis.nasa.gov/) | Fire Information for Resource Management System |
| **Active fire detection** | FSI Fire Alert System | [https://fsiforestfire.gov.in/](https://fsiforestfire.gov.in/) | Forest Survey of India real-time fire alerts |
| **Fuel / Vegetation** | Bhuvan Forest Cover Maps | [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) | ISRO forest cover and fuel load mapping |
| **Fuel / Vegetation** | MODIS NDVI Fuel Load | [https://modis.gsfc.nasa.gov/data/dataprod/ndvi.php](https://modis.gsfc.nasa.gov/data/dataprod/ndvi.php) | Vegetation health for fuel estimation |
| **Weather drivers** | IMD Temperature / Humidity / Wind | [https://www.imdpune.gov.in/](https://www.imdpune.gov.in/) | Fire Weather Index inputs |
| **Historical fire records** | FSI Annual Reports | [https://fsi.nic.in/](https://fsi.nic.in/) | Indian forest fire incidence reports |
| **Historical fire records** | GFED | [https://www.globalfiredata.org/](https://www.globalfiredata.org/) | Global Fire Emissions Database |
| **Geospatial means** | NBR (Normalized Burn Ratio) | [GEE](https://developers.google.com/earth-engine/) | Pre/post-event burn severity mapping |
| **Geospatial means** | Thermal Anomaly (VIIRS IBI) | [NASA](https://viirs.gsfc.nasa.gov/) | Active fire thermal anomaly detection |

---

## 11.7 Heatwaves

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Temperature** | IMD Gridded Temperature (1°×1°) | [https://www.imdpune.gov.in/cmpg/griddata/tem_1_Bin.html](https://www.imdpune.gov.in/cmpg/griddata/tem_1_Bin.html) | Daily max/min gridded temperature |
| **Temperature** | NASA POWER | [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/) | Reanalysis temperature/humidity data |
| **Heat index / Humidity** | IMD Heatwave Bulletins | [https://mausam.imdpune.gov.in/heatwave.php](https://mausam.imdpune.gov.in/heatwave.php) | Official heatwave criteria and bulletins |
| **Heat index / Humidity** | ERA5 Reanalysis | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) | ECMWF hourly temperature/humidity/wind |
| **Urban heat** | Bhuvan LST (MODIS/Landsat) | [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) | Land Surface Temperature for urban heat island |
| **Urban heat** | MODIS LST (MOD11A1/MYD11A1) | [https://lpdaac.usgs.gov/products/mod11a1v061/](https://lpdaac.usgs.gov/products/mod11a1v061/) | Daily 1km land surface temperature |
| **Urban heat** | Landsat TIRS Collection 2 | [https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) | High-resolution surface temperature (100m) |
| **Historical records** | NDMA Action Plans | [https://ndma.gov.in/](https://ndma.gov.in/) | National Disaster Management Authority heatwave action plans |
| **Historical records** | IMD Extreme Temperature Archive | [https://www.imdpune.gov.in/](https://www.imdpune.gov.in/) | Historical temperature anomaly records |
| **Geospatial means** | Sentinel-2 LST (UHI) | [COPERNICUS_S2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) | 10m resolution urban heat mapping |
| **Geospatial means** | ERA5 Soil Temperature | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) | Subsurface soil temperature for heat propagation |
| **Geospatial means** | MODIS NDVI Stress Index | [MODIS](https://modis.gsfc.nasa.gov/) | Vegetation heat stress detection |

---

## 11.8 Hailstorms

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Radar** | IMD Doppler Weather Radar (DWR) | [https://mausam.imd.gov.in/dwr/](https://mausam.imd.gov.in/dwr/) | Reflectivity data for convective cell detection |
| **Atmospheric instability** | ERA5 / NCEP Reanalysis | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) | CAPE, wind shear, freezing-level height |
| **Satellite** | INSAT-3D IR Brightness Temp | [https://mosdac.gov.in/](https://mosdac.gov.in/) | Cold cloud tops for strong convection |
| **Satellite** | GPM IMERG | [https://gpm.nasa.gov/data/imerg](https://gpm.nasa.gov/data/imerg) | Global precipitation measurement (hail proxy) |
| **Historical records** | IMD Severe Weather Archives | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | Thunderstorm/hail event records |
| **Historical records** | State Agriculture Dept. | [https://agriculture.gov.in/](https://agriculture.gov.in/) | Crop damage reports (hail tracking) |
| **Geospatial means** | ESWD Proxy Methodology | [https://eswd.eu/](https://eswd.eu/) | Convective intensity metrics for hail |
| **Geospatial means** | MODIS Cloud-Top Temp | [https://modis.gsfc.nasa.gov/](https://modis.gsfc.nasa.gov/) | High-res convective cell tracking |
| **Geospatial means** | VIIRS Nighttime Lights Anomaly | [NASA](https://viirs.gsfc.nasa.gov/) | Post-event damage assessment |

---

## 11.9 Dust Storms

| Data Type | Dataset / Source | URL | Resolution/Notes |
|---|---|---|---|
| **Aerosol data** | NASA MODIS/VIIRS AOD | [https://ladsweb.modaps.eosdis.nasa.gov/](https://ladsweb.modaps.eosdis.nasa.gov/) | Aerosol Optical Depth for dust events |
| **Aerosol data** | INSAT-3D Dust RGB | [https://mosdac.gov.in/](https://mosdac.gov.in/) | False-color imagery for dust detection |
| **Wind data** | IMD Surface Wind | [https://www.imdpune.gov.in/](https://www.imdpune.gov.in/) | Surface wind speed/direction |
| **Wind data** | ERA5 Wind Fields | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) | 3D wind reanalysis |
| **Land surface** | Bhuvan Land Use Maps | [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) | Arid/semi-arid land cover mapping |
| **Land surface** | NRSC Soil Moisture | [https://www.nrsc.gov.in/](https://www.nrsc.gov.in/) | Dry soil moisture precondition detection |
| **Historical records** | IMD Dust Storm Advisories | [https://mausam.imdpune.gov.in/](https://mausam.imdpune.gov.in/) | Pre-monsoon advisories (Rajasthan/Punjab/Haryana/Delhi) |
| **Historical records** | CPCB Air Quality Data | [https://cpcb.nic.in/](https://cpcb.nic.in/) | PM10 spikes during dust events |
| **Geospatial means** | MERRA-2 Dust Mass Reanalysis | [https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) | Global aerosol reanalysis (Dust Mass) |
| **Geospatial means** | NOAA HYSPLIT Trajectory | [https://www.ready.noaa.gov/HYSPLIT.php](https://www.ready.noaa.gov/HYSPLIT.php) | Dust transport path modeling |
| **Geospatial means** | Deep Blue AOD Algorithm | [NASA](https://eogdata.org/) | High-quality aerosol optical depth over land |

---

## Geospatial Means: Key Techniques for Disaster Differentiation

### Spectral Signatures
| Disaster Type | Primary Index | Satellite Sources |
|---|---|---|
| **Flood** | NDWI, MNDWI, AWEI | Sentinel-2, Landsat, MODIS |
| **Landslide** | NDVI Change, Slope Change | Sentinel-1 InSAR, Sentinel-2 |
| **Drought** | NDVI, EVI, VHI | MODIS, Sentinel-2 |
| **Cyclone** | Cloud-Top Temperature, OLR | INSAT-3D, MODIS, VIIRS |
| **Forest Fire** | NBR, BAI, TCBRT | MODIS, VIIRS, Sentinel-2 |
| **Heatwave** | LST, UHI | MODIS, Landsat TIRS, Sentinel-2 |
| **Hailstorm** | Cloud-Top Temp, Reflectivity | INSAT-3D, DWR, GPM |
| **Dust Storm** | AOD, Dust RGB | MODIS, VIIRS, INSAT-3D |

### Temporal Differentiation
- **Flood**: Sudden NDWI increase, followed by gradual decline
- **Landslide**: Rapid slope change (InSAR), vegetation disturbance
- **Drought**: Progressive NDVI decline over weeks/months
- **Cyclone**: Rotating cloud patterns, central dense overcast
- **Forest Fire**: Sudden thermal anomaly, NBR decrease
- **Heatwave**: Progressive LST increase, UHI intensification
- **Hailstorm**: Extremely cold cloud-top temperatures (< −52°C)
- **Dust Storm**: AOD spike, visibility reduction, PM10 increase

### Multi-Sensor Fusion
- **SAR + Optical**: Cloud-penetrating SAR for all-weather flood/landslide monitoring
- **Thermal + Optical**: LST + NDVI for heatwave/drought interaction
- **Radar + InSAR**: DWR reflectivity + Sentinel-1 for convective storm analysis
- **Aerosol + Wind**: MODIS AOD + ERA5 for dust storm trajectory modeling

---

## Next Steps

1. **Real-time Monitoring Setup**: Integrate IMD, ISRO, NASA APIs into GEE or local server
2. **Dataset Validation**: Cross-validate satellite-derived indices with ground truth (IMD stations, NCS seismic network)
3. **Model Development**: Train deep learning models (U-Net, ConvLSTM, ViT) on multi-sensor datasets
4. **Geospatial Difference Detection**: Implement spectral index time-series analysis for disaster classification
5. **Early Warning System**: Develop rule-based or ML-driven alert system using the datasets above