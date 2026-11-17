"""Human demonstration capture with the Wuji gloves and PICO wrist trackers.

`record` runs in this repository's `.venv` (Wuji SDK, xrobotoolkit_sdk) and
logs the laptop-side streams. The XRoboToolkit app on the headset records the
stereo passthrough video, head pose and trackers at the same time. Everything
else (`headset`, `align`, `hands`, `stereo`, `object_track`, `ground`,
`viewer`, `export`, `process`) runs offline in the retargeting repository's
`.venv-vision`, which provides viser, libigl and its held-segment refinement.

Frames: world is the PICO OpenXR world re-expressed in robot axes (x forward,
y left, z up, metres; `xr_tracker_source.xr_pose_to_robot`). Cameras use
OpenCV axes (x right, y down, z forward). `T_A_B` maps B coordinates into A.
"""
