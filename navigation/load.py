import wget
import os

files_name = ["2011_09_26_calib.zip",
"2011_09_26_drive_0001",
"2011_09_26_drive_0002",
"2011_09_26_drive_0005",
"2011_09_26_drive_0009",
"2011_09_26_drive_0011",
"2011_09_26_drive_0013",
"2011_09_26_drive_0014",
"2011_09_26_drive_0015",
"2011_09_26_drive_0017",
"2011_09_26_drive_0018",
"2011_09_26_drive_0019",
"2011_09_26_drive_0020",
"2011_09_26_drive_0022",
"2011_09_26_drive_0023",
"2011_09_26_drive_0027",
"2011_09_26_drive_0028",
"2011_09_26_drive_0029",
"2011_09_26_drive_0032",
"2011_09_26_drive_0035",
"2011_09_26_drive_0036",
"2011_09_26_drive_0039",
"2011_09_26_drive_0046",
"2011_09_26_drive_0048",
"2011_09_26_drive_0051",
"2011_09_26_drive_0052",
"2011_09_26_drive_0056",
"2011_09_26_drive_0057",
"2011_09_26_drive_0059",
"2011_09_26_drive_0060",
"2011_09_26_drive_0061",
"2011_09_26_drive_0064",
"2011_09_26_drive_0070",
"2011_09_26_drive_0079",
"2011_09_26_drive_0084",
"2011_09_26_drive_0086",
"2011_09_26_drive_0087",
"2011_09_26_drive_0091",
"2011_09_26_drive_0093",
"2011_09_26_drive_0095",
"2011_09_26_drive_0096",
"2011_09_26_drive_0101",
"2011_09_26_drive_0104",
"2011_09_26_drive_0106",
"2011_09_26_drive_0113",
"2011_09_26_drive_0117",
"2011_09_26_drive_0119",
]

for file_name in files_name:
    if file_name.endswith('.zip'):
        short_name = file_name
        full_name = file_name
    else:
        short_name = file_name + '_sync.zip'
        full_name = file_name + '/' + file_name + '_sync.zip'

    url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/' + full_name
    print("Downloading: " + short_name)
    print('url: ' + url)
    wget.download(url, out=short_name)
