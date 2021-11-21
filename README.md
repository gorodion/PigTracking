## План запуска
1. Скачать [модель детектора](https://drive.google.com/file/d/16xJD6_TLCeYWvIi-gqFrwhFF_lzVbe5l/view?usp=sharing) и [модель сегметации](https://drive.google.com/file/d/1pnYm_Ofyf_OS4CQp6arRv3aqSlK-bkYr/view?usp=sharing)
2. Указать в config.py путь до детектора (DETECTOR_PATH) и путь до модели сегментации (SEGMENTATOR_PATH)
3. Скрипт запускатся командой
```
python script.py VID_PATH
```
где VID_PATH - путь до папки с видео

## DEMO
[![Watch the video](https://github.com/gorodion/PigTracking/blob/main/demo.jpg)](https://youtu.be/Yr1gsxaYExQ)
