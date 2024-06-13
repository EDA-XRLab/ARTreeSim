# ARTreeSim


## Data Link
https://drive.google.com/drive/folders/1Nw1pUPu5PKESKSTGUeCnHd8o4e4Ckgu1?usp=drive_link


## Step 1
Download data from link above

## Step 2
cd to the the ARTREESIM directory

## Step 3
install requirements with pip

'pip install -r requirements.txt'

## Step 4
launch the application with cli argument --data

python3 launch.py --data PATH_TO_DATA


To view response examples go to

http://0.0.0.0:8100/docs


## Endpoints

/video 
Streams the realtime RGB images

/position
Streams the realtime position

/tree
Streams realtime tree detections

## Schemas

### /tree
This endpoint sends realtime tree detections as a geojson 'FeatureCollection'. All property units are in meters.

'''
{
    "bbox": [
        -114.04663625949121,
        46.857762026945935,
        -114.0464842183559,
        46.85789010246256
    ],
    "type": "FeatureCollection",
    "features": [
        {
            "bbox": [
                -114.04648742296072,
                46.85788931085731,
                -114.04648742296072,
                46.85788931085731
            ],
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    -114.04648742296072,
                    46.85788931085731,
                    975.0020748685424
                ]
            },
            "properties": {
                "x": 28.034755271308256,
                "y": -32.79832845335936,
                "z": -0.39807105655704644,
                "pixel_center": [
                    592,
                    493
                ],
                "diameter": 0.38640638910083297,
                "time": "2024-05-10T17:36:58.226831"
            },
            "id": "1963"
        },
        {
            "bbox": [
                -114.04657608680303,
                46.85784802951782,
                -114.04657608680303,
                46.85784802951782
            ],
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    -114.04657608680303,
                    46.85784802951782,
                    978.6482081685598
                ]
            },
            "properties": {
                "x": 21.272475801178818,
                "y": -37.3882469577442,
                "z": 3.2480630472325442,
                "pixel_center": [
                    1107,
                    329
                ],
                "diameter": 0.334226751973949,
                "time": "2024-05-10T17:36:58.226831"
            },
            "id": "1964"
        },
        {
            "bbox": [
                -114.04654301100568,
                46.8578365381372,
                -114.04654301100568,
                46.8578365381372
            ],
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    -114.04654301100568,
                    46.8578365381372,
                    978.1045350993861
                ]
            },
            "properties": {
                "x": 23.79514219896643,
                "y": -38.665905031155916,
                "z": 2.704373456103781,
                "pixel_center": [
                    916,
                    362
                ],
                "diameter": 0.44628364988984737,
                "time": "2024-05-10T17:36:58.226831"
            }
        }
        ]
    }
'''

### /position
This enpoint sends realtime positions as a geojson 'Feature'.
All property values are in Meters

'''
{
    "bbox": [
        -114.04685928258284,
        46.8581811749801,
        -114.04685928258284,
        46.8581811749801
    ],
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -114.04685928258284,
            46.8581811749801,
            975.198172217235
        ]
    },
    "properties": {
        "x": 0.3348601758480072,
        "y": -0.1429950296878814,
        "z": -0.2415633797645568,
        "azimuth": 64.41861496827346,
        "pitch": 0.0018508708201353,
        "roll": 0.0013460795586325,
        "time": "2024-05-10T17:34:41.351587"
    },
    "id": "0"
}
'''