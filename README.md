#Installation
Use the 'requirements.txt' file to install all the required packages and their dependencies. 
```python
pip install -r requirements.txt 
```
####or
```python
pip3 install -r requirements.txt
```


#How the filesystem works for data
```bash
.
└── Root/
    ├── Testing/
    │   ├── Gesture Tracking/
    │   ├── Models/
    │   ├── Cleaning Videos/
    │   ├── Audio Cleaning/
    │   ├── Speech to Text/
    │   └── Test Users/
    │       ├── Test_1/
    │       ├── Test_2/
    │       ├── Test_3/
    │       ├── Test_4/
    │       └── Test_.../
    ├── Ground_Truth/
    │   ├── Program_1/
    │   │   ├── Videos/
    │   │   ├── gesture_data/
    │   │   └── text_data/
    │   ├── Program_2/
    │   ├── Program_3/
    │   └── Program_.../
    └── Users/
        ├── User_1/
        │   └── baseline/
        ├── User_2/
        ├── User_3/
        └── User_.../
```