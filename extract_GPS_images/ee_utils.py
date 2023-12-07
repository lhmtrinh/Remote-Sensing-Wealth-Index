from typing import Any, Mapping, Optional, Tuple, Union

import ee
import pandas as pd
import time
from tqdm.auto import tqdm


Numeric = Union[int, float]


def df_to_fc(df: pd.DataFrame, 
             lat_colname: str = 'centroid_lat',
             lon_colname: str = 'centroid_lon') -> ee.FeatureCollection:
    '''
    Args
    - csv_path: str, path to CSV file that includes at least two columns for
        latitude and longitude coordinates
    - lat_colname: str, name of latitude column
    - lon_colname: str, name of longitude column

    Returns: ee.FeatureCollection, contains one feature per row in the CSV file
    '''
    # convert values to Python native types
    # see https://stackoverflow.com/a/47424340
    df = df.astype('object')

    ee_features = []
    for i in range(len(df)):
        props = df.iloc[i].to_dict()

        # oddly EE wants (lon, lat) instead of (lat, lon)
        _geometry = ee.Geometry.Point([
            props[lon_colname],
            props[lat_colname],
        ])
        ee_feat = ee.Feature(_geometry, props)
        ee_features.append(ee_feat)

    return ee.FeatureCollection(ee_features)


def date_range_for_year_quarter(quarter: int ,survey_year: int) -> Tuple[str, str]:
    '''Returns the start and end dates for filtering satellite images for a
    survey in the specified year and quarter.

    Args
    - survey_year: int, year that survey was started
    - quarter: int, quarter that image is in

    Returns
    - start_date: str, represents start date for filtering satellite images
    - end_date: str, represents end date for filtering satellite images
    '''
    if quarter == 1:
        start_date = f'{survey_year}-1-1'
        end_date = f'{survey_year}-3-31'
    if quarter == 2:
        start_date = f'{survey_year}-4-1'
        end_date = f'{survey_year}-6-30'
    if quarter == 3:
        start_date = f'{survey_year}-7-1'
        end_date = f'{survey_year}-9-30'
    else:
        start_date = f'{survey_year}-10-1'
        end_date = f'{survey_year}-12-31'
    return start_date, end_date

def tfexporter(collection: ee.FeatureCollection, prefix: str,
               fname: str, selectors: Optional[ee.List] = None,
               dropselectors: Optional[ee.List] = None) -> ee.batch.Task:
    '''Creates and starts a task to export a ee.FeatureCollection to a TFRecord
    file in Google Drive or Google Cloud Storage (GCS).

    GCS:   gs://bucket/prefix/fname.tfrecord
    Drive: prefix/fname.tfrecord

    Args
    - collection: ee.FeatureCollection
    - prefix: str, folder name in Drive or GCS to export to, no trailing '/'
    - fname: str, filename
    - selectors: None or ee.List of str, names of properties to include in
        output, set to None to include all properties
    - dropselectors: None or ee.List of str, names of properties to exclude
    - bucket: None or str, name of GCS bucket, only used if export=='gcs'

    Returns
    - task: ee.batch.Task
    '''
    if dropselectors is not None:
        if selectors is None:
            selectors = collection.first().propertyNames()

        selectors = selectors.removeAll(dropselectors)


    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=fname,
        folder=prefix,
        fileNamePrefix=fname,
        fileFormat='TFRecord',
        selectors=selectors)

    task.start()
    return task


def sample_patch(point: ee.Feature, 
                 patches_array: ee.Image,
                 scale: Numeric) -> ee.Feature:
    '''Extracts an image patch at a specific point.

    Args
    - point: ee.Feature
    - patches_array: ee.Image, Array Image
    - scale: int or float, scale in meters of the projection to sample in

    Returns: ee.Feature, 1 property per band from the input image
    '''
    arrays_samples = patches_array.sample(
        region=point.geometry(),
        scale=scale,
        projection='EPSG:3857',
        factor=None,
        numPixels=None,
        dropNulls=False,
        tileScale=12)
    sample_feature = arrays_samples.first()

    # Copy properties from the original image and the point
    sample_feature = sample_feature.copyProperties(source=patches_array)
    sample_feature = sample_feature.copyProperties(source=point)

    return sample_feature

def get_array_patches(
        img: ee.Image, 
        scale: Numeric, 
        ksize: Numeric,
        points: ee.FeatureCollection, 
        prefix: str, 
        fname: str,
        selectors: Optional[ee.List] = None,
        dropselectors: Optional[ee.List] = None, 
        ) -> ee.batch.Task:
    '''Creates and starts a task to export square image patches in TFRecord
    format to Google Drive or Google Cloud Storage (GCS). The image patches are
    sampled from the given ee.Image at specific coordinates.

    Args
    - img: ee.Image, image covering the entire region of interest
    - scale: int or float, scale in meters of the projection to sample in
    - ksize: int or float, radius of square image patch
    - points: ee.FeatureCollection, coordinates from which to sample patches
    - export: str, 'drive' for Google Drive, 'gcs' for GCS
    - prefix: str, folder name in Drive or GCS to export to, no trailing '/'
    - fname: str, filename for export
    - selectors: None or ee.List, names of properties to include in output,
        set to None to include all properties
    - dropselectors: None or ee.List, names of properties to exclude
    - bucket: None or str, name of GCS bucket, only used if export=='gcs'

    Returns: ee.batch.Task
    '''
    kern = ee.Kernel.square(radius=ksize, units='pixels')
    patches_array = img.neighborhoodToArray(kern)

    # ee.Image.sampleRegions() does not cut it for larger collections,
    # using mapped sample instead
    samples = points.map(lambda pt: sample_patch(pt, patches_array, scale))

    # export to a TFRecord file which can be loaded directly in TensorFlow
    return tfexporter(collection=samples, prefix=prefix,
                      fname=fname, selectors=selectors,
                      dropselectors=dropselectors)


def wait_on_tasks(tasks: Mapping[Any, ee.batch.Task],
                  show_probar: bool = True,
                  poll_interval: int = 20,
                  ) -> None:
    '''Displays a progress bar of task progress.

    Args
    - tasks: dict, maps task ID to a ee.batch.Task
    - show_progbar: bool, whether to display progress bar
    - poll_interval: int, # of seconds between each refresh
    '''
    remaining_tasks = list(tasks.keys())
    done_states = {ee.batch.Task.State.COMPLETED,
                   ee.batch.Task.State.FAILED,
                   ee.batch.Task.State.CANCEL_REQUESTED,
                   ee.batch.Task.State.CANCELLED}

    progbar = tqdm(total=len(remaining_tasks))
    while len(remaining_tasks) > 0:
        new_remaining_tasks = []
        for taskID in remaining_tasks:
            status = tasks[taskID].status()
            state = status['state']

            if state in done_states:
                progbar.update(1)

                if state == ee.batch.Task.State.FAILED:
                    state = (state, status['error_message'])
                elapsed_ms = status['update_timestamp_ms'] - status['creation_timestamp_ms']
                elapsed_min = int((elapsed_ms / 1000) / 60)
                progbar.write(f'Task {taskID} finished in {elapsed_min} min with state: {state}')
            else:
                new_remaining_tasks.append(taskID)
        remaining_tasks = new_remaining_tasks
        time.sleep(poll_interval)
    progbar.close()

    
class Sentinel:
    def __init__(self, filterpoly: ee.Geometry, 
                 start_date: str,
                 end_date: str) -> None:
        '''
        Args
        - filterpoly: ee.Geometry
        - start_date: str, string representation of start date
        - end_date: str, string representation of end date
        '''
        self.filterpoly = filterpoly
        self.start_date = start_date
        self.end_date = end_date
        self.buffer_area = self.filterpoly.buffer(224 * 20 / 2)

        self.sentinel_id = 'COPERNICUS/S2_HARMONIZED'
        self.sentinel = (
            self.init_coll(self.sentinel_id)
            .map(self.rescale)
            .map(lambda img: self.calculate_cloud_coverage(img ,self.buffer_area ))
            .sort('cloud_coverage')
            .sort('system:time_start')
            )

    def init_coll(self, name: str) -> ee.ImageCollection:
        '''
        Creates a ee.ImageCollection containing images of desired points
        between the desired start and end dates.

        Args
        - name: str, name of collection

        Returns: ee.ImageCollection
        '''
        return (ee.ImageCollection(name)
                .filterBounds(self.filterpoly)
                .filterDate(self.start_date, self.end_date)
                )
    
    @staticmethod
    def rescale(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, Sentinel-2

        Returns
        - img: ee.Image, with bands rescaled
        Check for the implementation
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED
        '''
        # bands blue, green, red, NIR, SWIR1 and SWIR2
        opt = img.select(['B2','B3','B4','B8','B11','B12']).divide(10000)
        
        # Cloud mask
        masks = img.select(['QA60'])

        scaled = ee.Image.cat([opt, masks]).copyProperties(img)

        # system properties are not copied
        scaled = scaled.set('system:time_start', img.get('system:time_start'))
        return scaled
    
    @staticmethod
    def calculate_cloud_coverage(image, buffer_area):
        """Calculates cloud coverage within a given buffer area.

        Args:
            image (ee.Image): A Sentinel-2 image.
            buffer_area (ee.Geometry): The area over which to calculate cloud coverage.

        Returns:
            ee.Image: A Sentinel-2 image with a cloud coverage property.
        """
        qa = image.select('QA60')  # Ensure QA60 band is integer

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Flags indicating clouds and cirrus.
        cloud_mask = qa.bitwiseAnd(cloud_bit_mask).neq(0)
        cirrus_mask = qa.bitwiseAnd(cirrus_bit_mask).neq(0)

        # Combine cloud and cirrus masks
        combined_mask = cloud_mask.Or(cirrus_mask)

        # Calculate the sum of cloud and cirrus pixels within the buffer area.
        cloud_pixel_count = combined_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer_area,
            scale=60,  # Match the scale to QA60 band resolution
            maxPixels=1e9
        ).getNumber('QA60')

        # Calculate the total number of pixels in the buffer area.
        total_pixel_count = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer_area,
            scale=60,  # Match the scale to QA60 band resolution
            maxPixels=1e9
        ).getNumber('area')

        # Calculate cloud coverage percentage.
        cloud_coverage = cloud_pixel_count.divide(total_pixel_count).multiply(100)

        return image.set('cloud_coverage', cloud_coverage)