# Vega browser motion dashboard — no Pico

Launch `scripts/launch_vega_ee_preview.sh`, then open **http://127.0.0.1:8088**
on the laptop. This defaults to simulation only. The robot preview, controls,
component dashboard and activity log all appear in the browser; no desktop GUI
is required. Right arm is selected initially; use the arm dropdown to switch
between arms in simulation. The other arm holds its starting position.

Install dependencies on a new machine with:

```bash
.venv/bin/python -m pip install -r requirements-wuji-web.txt
```

## Preview and execute

1. Use **Current pose** to populate the target and centre the drag pads.
2. Set X/Y/Z and roll/pitch/yaw, use the keyboard, or drag the target pads.
3. Review the automatically checked path playback (or click **Preview path**).
4. Click **Execute reviewed path** only after reviewing the motion. In simulation,
   this executes on fake devices; with `--hardware`, it commands the real robot.
5. **Enable glove** controls the selected fingers while both arms hold. Pause
   before planning an arm motion. Fingers hold during arm execution so the
   checked collision geometry stays fixed.

Keyboard and drag actions edit/preview only; they never execute a robot path.
A changed target, table alignment, arm selection or finger/robot pose invalidates
execution approval. The API rejects old preview IDs and execution before playback
has finished. Position targets refer to Vega's end-effector frame, not fingertips.
Coordinates use the model scene: X forward, Y left, Z up; metres and degrees.
Orientation is extrinsic XYZ Euler angles. Joint-space path speed is 0.15 rad/s,
acceleration 0.3 rad/s². The end effector need not travel in a Cartesian straight line.

## Inverse kinematics

The web dashboard uses the repository's seven-joint damped least-squares Jacobian
solver, configured for Cartesian pose targets. It is not the GMR/mink tracker
solver. Each solve starts at the current commanded arm joints, with 160 iterations,
0.02 damping, and a 0.1-radian iteration-step norm limit. Position/orientation
weights are 1.0 / 0.5; teleop elbow-flexion, elbow-down, roll-home and posture
preferences are disabled so they cannot pull an already-reached GUI target away.

The final pose must remain within 10 mm and 5 degrees of the request. Failure
reports residual position/orientation error as non-convergence, not proof that
the target is unreachable. Joint limits and separate path/table/hand collision
checks still apply. No acceptance tolerance was loosened for this change.

## Keyboard, mouse and camera

Click a target pad to focus keyboard control:

- Arrow Up/Down: X forward/back. Arrow Left/Right: Y left/right.
- Page Up/Page Down: Z up/down.
- I/K: roll. J/L: pitch. U/O: yaw.
- Shift: fine steps (1 mm or 0.4 degrees); default 5 mm or 2 degrees.

Drag the teal dot in **Top view** for X/Y or **Side view** for X/Z. The requested
position is marked by a teal sphere in the robot preview. Drag release and key
idle trigger a checked preview. Rejected targets report an inline reason.

Drag the robot image to orbit the camera, scroll to zoom, or click **Full-body
view**. Dragging the image changes the camera; dragging the pads changes the target.
Text fields preserve their normal typing and cursor keys. Space pauses outside
text/select fields; Esc requests software E-stop. The buttons always remain available.

## End-effector gizmo in the 3D preview

The displayed end effector has a small XYZ triad. The larger draggable target
axes use red X, green Y and blue Z. Select **Translate** to drag an arrow along
one axis, or **Rotate** to drag a ring around an axis. Select **End effector** for
local axes or **World** for scene axes. Rotation preserves target position.

Dragging a handle updates only the target; release starts the checked preview.
**Execute reviewed path** remains separate. Hold Shift for fine adjustment.
Translation is bounded to 30 cm per drag. If an axis is nearly aligned with the
camera or a ring is edge-on, orbit the view before dragging it. Empty-space drags
still orbit the camera. Pointer cancellation restores the pre-drag target.

The overlay uses projection metadata from each rendered image, rather than an
approximate screen mapping. After upgrading a running server, safely pause/stop
and relaunch it to load the new renderer and metadata; refreshing HTML alone is
not enough. Never restart an active hardware controller merely to update the UI.

## Robot model and tabletop

The complete supplied Vega-1U URDF is used: base (including caster geometry in its
mesh), lift, torso, head and both arms, with Wuji hands attached. There are no
separate wheel links. Base/lift/torso/head joints are fixed, not streamed from the
robot; the physical lift/torso configuration must match the model.

For this installation, the web model rotates the right-hand mount +90 degrees
about the end effector's local +Z axis. At the forearm-forward neutral pose, the
right palm faces robot-left (+Y); the fingers still point forward. This changes
the hand root transform, not any arm joint command. Rendering and collision
checks use the same corrected model. The left mount is unchanged. Re-preview
paths after this correction: the sideways hand can have less table clearance.

The GLB converter flattens scene objects before OBJ export so MuJoCo retains all
body faces. This is a conversion fix, not a change to the URDF submodule.

Your table is 30 inches (0.762 m) above the floor. Enter the robot base origin's
height above the floor and apply alignment. The initial 0.15 m is an unverified
model assumption. The model's fixed base offset is included in the conversion.
The checkbox confirming a measured height is required for hardware preview and
glove control. Visual alignment alone is an estimate.

The brown rectangle visualizes part of an infinite horizontal collision plane.
The check is conservative: it covers all X/Y even outside the real table's
footprint, including both arms/hands. Paths are sampled at no more than 0.01 rad
joint spacing with 3 cm preview clearance. People and other objects are not modelled.

## Components and real inputs

Network rows ping the installation's configured addresses for Jetson, robot SoC,
both gloves and both hands. **Reachable** does not imply SDK connection or live
streaming. Separate rows show arm heartbeat, hand feedback and selected glove
stream status; simulation devices are explicitly marked simulated. Network checks
never open device control connections. Pico is not used.

To use a physical glove in simulation, stop other glove consumers, then run:

```bash
scripts/launch_vega_ee_preview.sh --glove --side right
```

The selected arm is locked for a connected session. Restart with `--side left`
to change devices. Stop the existing server with Ctrl+C before relaunching.

For the physical robot, stop other teleop sessions and safely stop robot motion:

```bash
scripts/launch_vega_ee_preview.sh --hardware --glove --side right
```

The application reads both arms, but connects and enables **only the selected
Wuji hand** and reads only its matching glove. The opposite arm holds its initial
measured position. The other hand and glove do not need to be powered or connected.
Their dashboard rows are marked unused, even if network pings receive no reply.

Because the inactive hand's finger pose is unknown, it is hidden and replaced by
an amber conservative sphere (about 19 cm radius, computed from its kinematic
chain and mesh bounds). Every moving-arm/hand path and glove target is checked
against this envelope. Bounding spheres first prune distant pairs; nearby moving
links use point-to-convex-mesh distance against the inactive-hand sphere. This
avoids treating a long moving link as a large sphere. The inactive bound still
covers all finger hinge angles and can reject otherwise safe moves near that hand.
Rejections name the moving link and distinguish start-pose clearance from clearance
farther along the path. The 3 cm margin is unchanged. No guessed inactive finger angles
are used for clearance. The inactive hand's existing contact with its own arm or
the table is not validated because neither is commanded to move.

If feedback for the **selected** hand is missing, the dashboard opens READ-ONLY:
it displays measured arm joints, does not enable hands or stream arm commands,
and blocks arm execution and glove control. Restore that hand and restart; it
never automatically transitions from read-only to active control.

With all links available, startup holds the current arm pose;
there is no automatic reset. Hardware Execute also asks for
confirmation in the browser. Never run competing controllers/discovery connection
tests alongside it. Disconnected gloves hold their last target; restart to rebuild
a lost SDK connection.

Arm lag above 0.15 rad faults and holds. Lost heartbeat or hand feedback pauses.
In hardware mode, missing dashboard polling for 1.5 seconds pauses and invalidates
the preview. Keep the dashboard foregrounded. Use the physical E-stop for imminent
contact: software E-stop depends on the server/network responding. Closing the
server is not an emergency stop.

The server binds only to `127.0.0.1`; request host/origin and per-session keys are
checked. Do not expose it on a public or shared network. `--port` changes the local
port. No assets or telemetry are sent to a hosted website.

## Validation

New unit/API tests cover input mappings, preview-only behavior, path collision
rejection, inactive-arm hold, changed-pose rejection, pause/E-stop gates, arm
selection, offscreen image rendering, stale preview IDs and cross-origin rejection.
A mesh-conversion regression checks multiple transformed GLB objects retain all faces.
Physical execution has not been tested by the implementation agent.
