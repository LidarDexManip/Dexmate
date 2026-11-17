# Human demonstration capture (gloves + PICO trackers + headset stereo)

Record a person manipulating an object with the same Wuji gloves and PICO wrist
trackers as the teleop. The result is hand and object reference motion in one
world frame, with the object grounded to the hands. Retargeting then turns it
into Wuji hand motion, using the `~/Desktop/retargeting` checkout. The robot is
not involved.

```text
laptop: gloves (skeleton, fingertip sensors, palm IMU) + trackers/headset (PC Service)
headset: stereo passthrough video + head pose + trackers   (XRoboToolkit recording)
   -> clocks aligned by the trackers seen in both
   -> world hands: forearm tracker + palm IMU + glove skeleton
   -> stereo depth -> object mesh tracked in depth, carried by the hand when hidden
   -> grounding: retargeting's held-segment refinement against the hands
   -> viser world view, retargeting input bundle
```

## Requirements

- **PICO 4 Ultra Enterprise with VST camera access.** It must be approved
  through PICO's enterprise service and enabled in the XRoboToolkit app. Without
  it, the app records poses but no video.
- **Both gloves on the glove network** (see [Connect the gloves](#connect-the-gloves)),
  and the XRoboToolkit PC Service.
- **A mesh of the object** in metres (or millimetres with `--mesh-scale 0.001`).
- **The retargeting environment:** `~/Desktop/retargeting/.venv-vision`.
  - Install it with `cd ~/Desktop/retargeting && ./setup_retarget.sh`.
  - The system Python lacks `python3.12-venv`, so create the environment
    first with `uv venv --python /usr/bin/python3.12 .venv-vision`.
- **`adb`**, to pull the headset files.
- **Nothing else holding the gloves.** No teleop or rehearsal may be running;
  the recorder refuses to start if one is.

## Connect the gloves

Do this every time the gloves are plugged in, after a replug, and after a
reboot. Turn the gloves on and wait about 30 s for them to boot first.

```bash
cd ~/Desktop/Dexmate
ip -br link                                          # find the glove port (see below)
nmcli device disconnect <glove-port>                 # free the port
bash scripts/configure_wuji_network.sh <glove-port>  # set it up for the gloves
.venv/bin/python -m wuji_teleop discover --seconds 3 # both gloves should be listed
```

- **The glove port** is the Ethernet port the gloves' cable is plugged into.
  On this laptop it is the USB Ethernet adapter `enx8cae4cebd67f`; the name
  stays the same in any USB socket, and a different adapter gets a different
  `enx...` name.
- **Success:** the script ends with "Glove network ready: 192.168.1.100
  answers." (and the same for .101). If the gloves are already set up, it says
  "Glove route(s) already use ..." and changes nothing, so running it again is
  harmless.
- **Always name the port.** Without it, the script takes the first wired port
  with a cable, which can be the robot's (`enp128s31f6`, profile
  `robot-lan`). It then stops with "already has an IPv4 address from another
  network profile". Do not disconnect the robot's port.

**Why these steps are needed.** The laptop's Wi-Fi (`LIDAR_WiFi_5G`) uses the
same addresses as the gloves (192.168.1.x), so without this setup the laptop
sends glove traffic over Wi-Fi, where there are no gloves. The glove port is
also set up for the lab network (profile `Default`, 10.10.x) and takes it back
by itself whenever the cable is replugged. The script gives the glove port the
laptop's glove address (192.168.1.50) and sends only the two glove addresses
over it. Wi-Fi, the internet and the robot link are not changed.

**What it changes:**

- The lab network is off on the glove port until `nmcli connection up Default`.
- The glove setup is temporary: a replug or a reboot removes it (run the steps
  again).
- **Before running the teleop with the robot hands,** remove it with
  `nmcli connection delete wuji-glove-sim-<glove-port>`. Otherwise
  192.168.1.50 is on two ports (this one and `robot-lan`), the hands' SDK
  can connect from the wrong one, and a hand locks up until a 12 V power cycle.

## Record

1. `scripts/record_human_demo.sh` (or `--out ~/vega-demos/<name>`). It connects
   both gloves and waits for the trackers and headset.
2. On the headset, in XRoboToolkit:
   - Connect to the laptop and turn on Head tracking.
   - In the **Pico Motion Tracker** tab, choose object tracking, then **Send**.
   - In **Data Collection**, turn on Tracking + Vision, then **Record**. A
     resolution of 2160x810 is a good start.
3. The recorder then guides a wrist calibration:
   - **still** (3 s): forearms level and forward, palms down, wrists straight.
   - **rotate** (10 s): keep the wrists straight and turn the forearms around.
     Turn them palms up and down, and point forward, sideways and up. At least
     30 degrees of turning is needed.
4. Demonstrate. Keep the object in view of the headset as much as possible.
5. Stop the headset **Record** first, then press Ctrl-C in the recorder.

The trackers stay on the forearms (watch-style). The hand's orientation comes
from the glove's palm IMU, and the calibration relates the two. The wrist joint
centre is assumed to be 25 mm below each tracker and 20 mm ahead of it along the
hand. Measure your mounting and pass `--mount-height` / `--mount-back` to
`hands` if it differs.

## Live view of the headset cameras (optional)

The stock XRoboToolkit app also streams its passthrough cameras to the laptop,
with no change to the APK. Use it to check framing, focus and lighting before
recording:

```bash
scripts/launch_pico_stereo_view.sh --headset <headset-wifi-ip>   # or omit --headset with adb attached
```

- **What it does:** it sends an `OPEN_CAMERA` request to the app (TCP 13579).
  The headset then connects back to the laptop (TCP 12345) and streams
  side-by-side H.264, shown left | right in an always-on-top window. Keys:
  `s` saves a PNG to `~/Pictures/pico-stereo`, `t` toggles on top, `q` quits.
  Quitting sends `CLOSE_CAMERA`.
- **On the headset:** keep the XRoboToolkit app open with camera access on.
  Nothing needs pressing; the app answers the request by itself.
- **Not while recording.** Recording and streaming share the headset's camera
  and encoder, so stop the stream before pressing **Record**.
- **Frame times.** The app fixes the bitrate at about 5 Mbit/s and sends no
  capture time or head pose, so frames carry only their arrival time here.
  `--record PREFIX` saves the raw H.264 and each packet's arrival time. To use
  the stream as the vision input, record it with `--stereo` (next section).
- **Tested** on the real headset (2026-11-12): 640x480 per eye at 20-30 fps
  (the app picks the rate), arrival jitter p95 5 ms.

## Record the live stereo stream with depth (instead of the headset recording)

`--stereo HEADSET_IP` records the headset's stereo stream on the laptop, with
Fast-FoundationStereo disparity for every frame, next to the gloves, trackers
and head pose. No adb pull, no headset Record.

```bash
scripts/record_human_demo.sh --stereo <headset-wifi-ip>      # add --out ~/vega-demos/<name>
scripts/process_human_demo.sh sync <session>
```

- **Setup.** Fast-FoundationStereo lives in `~/Desktop/Fast-FoundationStereo`
  with its own `.venv` (Python 3.12, torch 2.8 cu128 for the RTX 5080; the
  README's cu124 build does not run on it) and the research weights from
  NVIDIA's Google Drive in `weights/`. `--stereo-python` points elsewhere.
- **On the headset:** the XRoboToolkit app in front, the headset worn and
  awake (the app ignores camera requests otherwise), Head tracking on, trackers
  sent. Do not press Data Collection Record, and close any other stereo viewer:
  the app serves one camera client at a time.
- **Steps.** After all links are live, the recorder asks for a `headshake`
  (5 s; look at a textured scene, turn the head left and right, nod), then the
  wrist calibration (`still`, `rotate`), then the demonstration. Ctrl-C stops.
  The headshake times the camera stream; without head motion `sync` cannot
  find the stream latency.
- **What it writes** (`<session>/stereo/`, about 8 MB/s at 30 fps):
  - `stream.h264` and `stream.times.csv`: the raw stream and each packet's
    arrival time.
  - `frames.csv`: each frame's packet arrival and decode time (host UTC ns).
  - `left/NNNNNN.jpg`: the left eye.
  - `disparity/NNNNNN.png`: disparity in 1/64 px (uint16, 0 = no match).
  - `meta.json`: the nominal intrinsics (76.35 x 61.05 deg field of view).
- **Live view** (always on top; `--no-stereo-window` hides it):
  - It opens at once and refreshes at 20 Hz.
  - The ego camera (left eye, always the newest frame) sits next to depth (the
    newest computed). Red is near, grey is outside 0.2-1.5 m, black is no
    match.
  - A header shows the recorder's current step with its instructions and
    countdown, and the link status (gloves, trackers, headset), red while
    something is missing.
  - Hovering reads the depth at that pixel; `t` toggles on-top.
  - Without `--stereo-baseline-m` it shows disparity (0-128 px) instead of
    metres.
- **Depth speed.** The default checkpoint `20-30-48` takes about 30 ms per
  frame and keeps up with the 30 fps stream, so the depth view stays live.
  `--stereo-checkpoint 23-36-37` is more accurate (about 41 ms). At 30 fps it
  falls behind: a queue builds (about 1 GB per 3 min) and is finished after
  Ctrl-C, in about a third of the recording time.
- **`sync`** (`wuji_teleop/demo/stream_sync.py`) puts everything on host UTC:
  - Gloves and palm IMUs carry SDK stamps in host UTC already.
  - Trackers and the head pose carry headset stamps. These are mapped to the
    host by the lower envelope of (arrival - stamp): the headset clock ran
    0.21-0.25 s ahead.
  - Frames: the capture time is the packet arrival minus the stream latency.
    The latency is the shift that best matches the image's rotation between
    frames (optical flow) to the head pose's rotation. The same pairs give the
    camera's rotation in the head frame.
  - Writes `processed/synced.npz`: per frame, the capture time, head and
    camera poses, tracker poses, the glove keypoints in the glove wrist frame,
    palm IMU quaternions, and the world-frame hands when the wrist calibration
    succeeds.
  - Also writes `sync_report.json`: rates, jitter, coverage, latency and
    correlation.
  - Also writes `sync_check.png`: head versus image rotation after the shift,
    and the correlation curve.
  - It stops when the correlation is below 0.6 (no head motion); `--latency
    SECONDS` overrides.
- **Depth.** `stream_sync.load_frame(session, i)` returns the left image,
  disparity and depth. Depth needs `--baseline-m` (sync) or
  `--stereo-baseline-m` (record), which is not yet measured for this headset. To measure it, hold a textured object at a known distance
  Z and read its disparity d: the baseline is Z * d / 407 (the nominal fx at
  640x480).
- **First real result** (2026-11-12, worn, one head turn): stream latency 62 ms
  at correlation 0.96. Head, trackers, gloves and palm IMUs covered 100% of
  frames. The camera rotation in the head frame was 1.2 deg from nominal.
  - The image turned 7% less than the head (28 pairs), so the true focal
    length may be about 7% below the nominal 407 px. That affects sideways
    positions, not depth from a known-distance baseline.
  - Turn the head left and right several times in the headshake: one turn
    left the correlation peak 78 ms wide.
- **Headset on a stand.** A headset that does not move cannot time the
  stream; `sync` then stops. Pass the latency measured in a worn take
  (`--latency 0.062`). The hands do not depend on it, only which image each
  hand pose is paired with. Also pass `--nominal-camera`: the camera rotation
  estimated from a still headset came out 5 deg off on 2026-11-16 and moved the
  hands further from the gloves.
- **Camera position and baseline (2026-11-16).** By default `sync` puts the
  left camera at the head pose's origin. The tracker axes then land beside the
  real pucks in the image (the right one about 5 cm, 47 px). Use:

  ```bash
  scripts/process_human_demo.sh sync <session> --latency 0.062 --nominal-camera \
    --camera-offset -0.023 -0.011 -0.074 --baseline-m 0.064
  ```

  - **Baseline 64.0 mm: the headset's factory calibration.** Every Data
    Collection recording starts its `trackingData_*.txt` with the two colour
    cameras' poses (`cameraExtrinsics`, from PICO's enterprise SDK); read it
    with `adb shell head -n 1 /sdcard/Download/trackingData_<stamp>.txt`. The
    two cameras are 63.97 mm apart. The live stream itself carries no camera
    data. Fitting the right tracker to its puck (detected in the image, with
    its depth) gave 65.5 mm independently. The 83 mm estimated on 2026-11-12
    assumed the camera at the head origin and is wrong.
  - **Camera offset: fitted.** The header's camera positions are relative to a
    device frame whose origin sits about 5 cm to one side of the cameras'
    midpoint, not the tracked head origin, so they don't give the offset
    directly. With the baseline fixed at 64.0 mm, the puck fit places the left
    camera 2.3 cm left, 1.1 cm down and 7.4 cm forward of the head origin (3D
    puck error 5.9 -> 0.9 cm median). `--camera-offset` is in the OpenXR head
    frame (x right, y up, z back).
  - **Intrinsics:** the header's (fx 407 at 640 px wide) are computed from the
    nominal 76.35 x 61.05 deg field of view, not measured: the same nominal
    values `sync` uses.
  - A wrong baseline scales every depth by the same factor but does not change
    the tracker or hand poses, or where they project in the image.
- **Which tracker is which wrist.** With the gloves on, `sync` pairs each
  glove with the tracker whose rotation speed correlates with its palm IMU
  (about 0.8 against 0.2). It uses the headset's heading only without gloves.
  The heading rule had swapped the wrists in a take.
- **Wrist calibration.** The `rotate` step needs at least 30 deg of forearm
  turning per arm; turn palms fully up and down. With 35-41 deg (2026-11-12)
  the projected right hand sat 7-10 cm and about 30-45 deg off the glove in
  the image. `--calibration-from <other session>/processed/sync_report.json`
  supplies a calibration for a hand whose own fails, valid while the trackers
  stay on their straps and the gloves are not restarted.

## MANO hands and the 3D view (live stereo sessions)

```bash
scripts/fit_mano.sh <session> [<session> ...]      # after sync
scripts/process_human_demo.sh view <session>          # http://localhost:8120
```

- **Model files.** MANO's model is licensed: register at
  https://mano.is.tue.mpg.de and download `mano_v1_2.zip` ("Models & Code")
  to `~/Downloads` (or pass `--mano`). The pickles are read without the
  unmaintained `chumpy`.
- **Fit** (`wuji_teleop/demo/mano_fit.py`, Fast-FoundationStereo's
  environment, GPU, a few seconds per hand):
  - Per hand and take: one shape (10 betas) and, per camera frame, the global
    orientation, translation and the 15 finger joint rotations (relative to
    MANO's mean hand).
  - Targets are the 21 world keypoints from `synced.npz`, weighted by glove
    confidence.
  - Priors: no twist of a finger bone about its own axis, which keypoints
    cannot show; hinge-like finger middle and end joints; poses near the mean
    hand; smooth over time.
  - Writes `processed/mano_<side>.npz` (vertices, joints, parameters,
    per-keypoint error), `mano_report.json`, and `mano_overlay.mp4` (the left
    camera with the MANO hands and glove keypoints drawn in, H.264).
  - First results (2026-11-12, four takes): keypoint error median 4.4-5.2 mm,
    fingertips 1.4-2.7 mm, 6-8 s per hand. The largest misfits (1-2 cm) are at
    the thumb base and the knuckles, where the glove model and MANO place
    joints differently.
- **Lost glove finger.** A finger whose glove confidence stays below 0.6 over
  the take (median) is not fitted to: its joints bend as the mean of its
  neighbours' (`--no-stand-in` turns this off). The right glove's ring read
  about 0.3 and lay straight in every frame of the 2026-11-12 takes (the same
  sensor that failed on 2026-11-06). Hiding the healthy left ring this way put
  its tip 14-16 mm (median) from the glove's, against 46-52 mm with no stand-in.
- **Depth alignment** (`wuji_teleop/demo/depth_align.py`, `--depth-align
  track|initial|off`, default `track`). The trackers and the wrist calibration
  place the hands only to within 5-15 cm and 35 degrees, and the error changes
  through a take, so after the fit each hand is moved onto the stereo depth:
  - First, one rigid correction (turn about the wrist, then shift) over the
    first 3 s with the hand in view: a GPU grid search that projects the
    camera-facing MANO vertices into the left image and compares them with the
    depth, errors truncated at 1.5 cm.
  - `track` then carries it in 0.5 s windows every 0.25 s (1 s windows scored 2% worse on frames they were not fitted on, take2 2026-11-16), each searched near the
    previous one with a prior towards it, interpolated per frame. `initial`
    applies the first correction to the whole take: on 2026-11-12 it held for
    only 3-6 s on the right hand (no single correction, in the world, hand or
    tracker frame, held across a take).
  - A correction over 15 cm or 45 degrees, or still missing most of the depth,
    is refused (the hand stays as fitted): with the hand out of view, a flat
    hand fits the empty table.
  - The per-frame correction is kept in `mano_<side>.npz` (`depth_R`, `depth_t`);
    the overlay video and `view` move the glove keypoints with it. `error_mm`
    stays against the glove.
  - Depth needs f*B: `--baseline-m`, sync's, else a wrist tracker seen in the
    depth (tracked distance times disparity at its projection; a tracker whose
    projection misses its puck reads scattered values and is not used).
  - **Baseline:** the 2026-11-12 estimate (f*B 34.4 and 33.2 px*m from the
    left tracker in takes 3 and 4, about 83 mm) assumed the camera at the head
    origin and is superseded by the factory 64.0 mm with `--camera-offset` (see "Camera
    position and baseline" above).
  - **On hold (2026-11-16).** Depth alignment is paused until the camera
    position, baseline and wrist offset are settled. On `sy_test` it did not
    put the hands on the gloves more often than the corrected camera alone, and
    moved the left hand off its glove. `--inflate-m`, `--rotation-frame hand`
    and `--shift-frame tracker` (a 4 mm thicker hand, a hand-frame turn and a
    tracker-frame shift) were tried there and gave the same result. Use
    `--depth-align off` for now.
  - Results (2026-11-12, `--baseline-m 0.083`): depth misfit (share of
    camera-facing vertices off the depth by over 1.5 cm, truncated) fell from
    0.80-0.98 to 0.52-0.58 in takes 3-4; the right hand moved 8-12 cm and
    turned about 30 degrees. Takes 1-2: the left hand never enters the view
    (skipped); right 0.94 -> 0.80, partly windows with the hand out of view.
  - The right tracker read about 5 cm left of its puck in the image; the left
    one projected onto its puck.
- **Accuracy.** Without the depth alignment the fit follows the tracker and
  glove keypoints, so the wrist calibration error above carries into MANO. The
  MANO hand is a little smaller than the gloved hand in the image (glove bulk).
- **`view`** on a live-stereo session (`hands_view.py`) shows each frame's MANO
  meshes (or the glove keypoints before a fit), the trackers and the camera
  frustum with the left image. With a baseline (or `fit_mano`'s f*B) it
  also shows the depth cloud, which carries held objects (the pipette in takes
  20261112_193517 and _193547). A hand that never enters the camera image is
  hidden at first ("Hands out of view"): in those two takes the left hand is
  not used and hangs out of view.
- **Camera-view layers:** `scripts/view_demo_layers.sh <session>`
  (`layer_view.py`) shows the left image with the depth beside it, never on top,
  both scaled to the window. Each layer has a check box or number key:
  1. trackers (axes);
  2. wrist (yellow, joined to its tracker; a grey ring where it was before the
     depth fit);
  3. hand orientation (thick axes);
  4. depth panel;
  5. finger skeleton;
  6. hand model.

  The **Model** box picks any `processed/<name>_<side>.npz` that holds a surface
  (`mano`, or an experiment's), and the wrist, orientation and skeleton follow
  that model's depth correction; "none" shows them as synced. Keys: Space to
  play, Left/Right one frame (Shift: one second), L/R to hide a hand.
- **`view --video`** renders the same scene to `processed/hands_view.mp4`: a
  headless Chrome is the viser client, and each frame's 3D render (fixed
  viewpoint: `--video-azimuth 30 --video-elevation 55 --video-distance 0.65`,
  turned from the head camera's side of the hands; the frustum drops its image,
  which hid the hands) sits beside the left
  camera image with the MANO hands drawn in. About 0.2 s per frame (software
  WebGL); `--video-step N` renders every Nth frame.

## Process

Every stage is `scripts/process_human_demo.sh STAGE SESSION [options]`:

| Stage | Does | Writes (`SESSION/processed/`) |
|---|---|---|
| `pull` | Copies the newest `CameraRecord_*.mp4` and `trackingData_*.txt` off the headset (adb). | `SESSION/headset/` |
| `inspect` | Reports rates, trackers, camera intrinsics/extrinsics and the video timing. | (prints) |
| `hands` | Aligns the clocks, calibrates the wrists, and computes world hands and head-camera poses at every video frame. | `timeline.npz`, `calibration.json` |
| `depth` | Rectifies each stereo pair and computes depth (SGBM + WLS, half resolution). | `frames/NNNNNN.jpg` / `.png` (mm) |
| `view` | Opens the viser world viewer at http://localhost:8120. | |
| `track` | Tracks the object mesh in the depth from the pose saved in the viewer. | `object_track.npz`, `object.json` |
| `ground` | Detects holds and grounds the object to the hand in each. | `grounded.npz`, `ground_report.json` |
| `export` | Writes the retargeting input bundle and prints the `retarget.sh` command. | `retarget_input/` |

A first session:

```bash
S=~/vega-demos/<name>
scripts/process_human_demo.sh pull $S
scripts/process_human_demo.sh inspect $S
scripts/process_human_demo.sh hands $S
scripts/process_human_demo.sh depth $S
scripts/process_human_demo.sh view $S --mesh /path/to/object.obj
#   In "Place object": pick a frame where the object is clear, press "Object to
#   view centre", drag the gizmo onto the point cloud, then "Save initial pose".
scripts/process_human_demo.sh track $S
scripts/process_human_demo.sh ground $S
scripts/process_human_demo.sh export $S
scripts/process_human_demo.sh view $S        # hands, tracked and grounded object
```

### Check the camera conventions on the first real recording

PICO does not document these, so verify them once:

- **Extrinsic direction:** head-from-camera or camera-from-head.
- **Camera axes:** OpenXR or OpenCV.
- **Intrinsics scope:** one eye, or the whole side-by-side frame.
- **Video timestamps:** `first-image` or `predict`.

`inspect` prints the raw values. In the `view` stage, the hand skeletons must
sit on the hands in the depth point cloud. If they are mirrored, behind the
camera or offset, rerun `hands` with `--extrinsic camera_from_head`,
`--camera-axes opencv`, `--intrinsics per-eye|full` or
`--frame-time predict|first-image`, then `depth` again.

## How each part works

- **Clock.** The headset's trackers (its own UTC) are matched to the same
  trackers relayed to the laptop (host UTC). Speed is cross-correlated, then
  position is fitted. The relay delay (a few ms) is folded into the offset.
  Glove SDK timestamps are already host-UTC.
- **Hands.** The wrist calibration solves `R_tracker X = Y R_imu_wrist`, in which
  `X` is the tracker mounting and `Y` is the IMU world in the PICO world.
  Afterwards:
  - **Hand orientation:** `Y R_imu_wrist(t)`.
  - **Wrist position:** the tracker position plus the fixed mount offset.
  - **Fingers:** the glove skeleton, placed by that wrist pose.
  - **Heading-drift check:** `hands` prints the hand's twist about the forearm
    axis. It should stay near zero; a steadily growing twist means the palm IMU
    heading is drifting.
- **Object tracking.** Each frame fits the mesh to the depth with robust
  point-to-plane ICP.
  - **Excluded points:** the table and the depth points on the hands.
  - **Carrying:** while two or more hand regions touch the object, its
    prediction moves with that hand's wrist. The resting pose stays a second
    hypothesis, so a release is detected.
  - **Occlusion:** a frame that does not fit is marked not fitted and keeps the
    hand-carried prediction.
- **Grounding.** This step uses retargeting's
  `baselines.held_refinement.Problem`.
  - **What it solves:** small rigid corrections of the object (`object-only`,
    the default), or of the object and the hand root (`object-hand-root`),
    over each hold plus padding.
  - **Residuals:** bidirectional penetration, contact patches on the regions
    that touch (median gap at most 4 mm), correction size and smoothness, a
    palm-relative prior when the grasp is steady, and reprojection drift in the
    moving head camera.
  - **Bounds:** at most 30 mm / 20 deg on the object and 10 mm / 10 deg on the
    hand root. Corrections taper to zero, so frames outside holds are
    unchanged.
  - **Changes in this repo:** a capsule hand surface (`hands.HandSurface`)
    replaces MANO, there is one camera pose per frame, and holds are detected
    automatically. `--holds FILE` overrides the detection, and
    `--penetration-weight` (default 5) strengthens the penetration term.
- **Export.** The bundle's frame is the gravity-aligned PICO world, shifted up
  so that z > 0. Then retarget with, for example:

  ```bash
  cd ~/Desktop/retargeting && MUJOCO_GL=egl ./retarget.sh --method toporetarget \
    --hand right --input SESSION/processed/retarget_input --continuity-experiment limited \
    --output outputs/<name>_right
  ```

## Limits

- **Testing so far.** The pipeline has passed a synthetic end-to-end session
  (`tests/test_demo_pipeline.py`) but no real recording yet. The headset camera
  conventions above are assumptions until checked.
- **Object depth.** Stereo depth on thin or shiny objects is weak, and an object
  hidden by the hand is carried by the hand, not measured. A hold with a single
  contact region constrains the object only along the contact normal.
- **Grounding scope.**
  - Each hold is grounded against one hand. When both hands hold the object at
    once, each hand's hold is grounded in turn, and the later one wins where
    they overlap.
  - The capsule hand is coarser than a fitted MANO skin.
  - Nothing here checks physics or grasp stability.
