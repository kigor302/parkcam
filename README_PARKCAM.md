# ParkCam

Raspberry Pi CM4 parking camera service.

## Build

```sh
cd ~/projects/parkcam
make
```

## Run

```sh
cd ~/projects/parkcam
./parkcam
```

Open `http://172.16.0.27:8080/` for events, `http://172.16.0.27:8080/snapshot` for the installation/focus page, and `http://172.16.0.27:8080/config` for settings.

## Configuration

Settings are stored in `~/projects/parkcam/parkcam.conf` and can also be edited from the web UI.

Important defaults:

- `detection_interval_ms=1000`
- `retention_days=30`
- `track_timeout_ms=60000`
- `motion_enabled=1`
- `motion_min_area=5000`
- `capture_width=1280`
- `capture_height=720`
- Resolution presets in the web UI: `1280x720`, `1920x1080`, `2560x1440`, `3840x2160`, `4000x3000`
- `http_port=8080`

## Data

- SQLite DB: `~/projects/parkcam/data/parkcam.sqlite3`
- Event images: `~/projects/parkcam/data/events/`
- Snapshot images are kept in memory and JPEG-encoded only when the web UI requests `/snapshot-image`.

## Notes

Vehicle triggering uses a lightweight OpenCV motion/background-subtraction gate plus blob tracking. Each active track creates one event, and tracks remain alive for `track_timeout_ms` so a vehicle pause under one minute does not create duplicate events. Separate foreground blobs can create separate events for tailgating. If two cars are completely merged into one blob from the camera angle, tune camera placement, `motion_min_area`, and the field of view for better separation.

FastALPR is used only when an event is created. The installed FastALPR package is Python-only, so ParkCam keeps a persistent worker process alive and reuses the loaded model instead of launching Python for each event. If a plate cannot be read, the event is stored with `Unknown`.


## Current Detection Pipeline

Motion is only a CPU-saving gate now. A motion track must also match the OpenCV DNN MobileNet-SSD vehicle detector before ParkCam creates an event. ALPR runs asynchronously after the event image is stored, so a slow FastALPR result does not pause capture/detection. Default `event_confirm_frames` is `1` so a fast car can trigger from a single confirmed frame.

Vehicle confirmation uses MobileNet-SSD model files in `~/projects/parkcam/models/`. Tune `vehicle_confidence`, `vehicle_min_area_ratio`, and `vehicle_overlap_threshold` from Settings if background vehicles are still too sensitive or distant valid vehicles are missed.
