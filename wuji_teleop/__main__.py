"""Run from the workspace root with .venv/bin/python -m wuji_teleop."""

import argparse
import sys
import time


def positive_seconds(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def parser():
    result = argparse.ArgumentParser(description="Wuji glove → standalone Wuji Hand 2 in MuJoCo")
    commands = result.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="List gloves and measure their tracking streams")
    discover.add_argument("--glove-sn", help="Inspect only this glove serial number")
    discover.add_argument("--seconds", type=positive_seconds, default=2.0)
    play = commands.add_parser("run", help="Control the simulated hand")
    play.add_argument("--input", choices=("glove", "synthetic"), default="glove")
    play.add_argument("--side", choices=("auto", "left", "right"), default="auto",
                      help="Synthetic side (auto means right); with a glove, validate the detected side")
    play.add_argument("--glove-sn", help="Required when multiple gloves are discovered")
    play.add_argument("--user-id", default="", help="Named calibrated SDK user ID; default: built-in model")
    play.add_argument("--headless", action="store_true")
    play.add_argument("--duration", type=positive_seconds, help="Stop after this many wall-clock seconds")
    play.add_argument("--report", help="Write final run metrics as JSON")
    return result


def discover(args, sdk):
    from .sources import GloveSource, scan_gloves, select_glove
    manager = sdk.SdkManager.instance()
    devices = scan_gloves(manager, sdk)
    if args.glove_sn or not devices:
        devices = [select_glove(devices, args.glove_sn)]
    healthy = True
    for device in devices:
        source = GloveSource(device.sn, sdk=sdk)
        try:
            print(f"SN={source.serial} side={source.side} firmware={source.firmware} address={source.address}", flush=True)
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:
                time.sleep(min(0.05, args.seconds))
            snapshot = source.read()
            elapsed = time.monotonic() - start
            age = None if snapshot.frame is None else time.monotonic() - snapshot.frame.received_at
            age_text = "n/a" if age is None else f"{age * 1000:.1f} ms"
            print(f"  skeleton={snapshot.received / elapsed:.1f} Hz frames={snapshot.received} age={age_text} error={snapshot.error}")
            healthy &= snapshot.frame is not None and age <= 0.250 and snapshot.error is None
        finally:
            source.close()
    return 0 if healthy else 1


def main(argv=None):
    argparser = parser()
    args = argparser.parse_args(argv)
    if args.command == "run":
        if args.headless and args.duration is None:
            argparser.error("--headless requires --duration")
        if args.input == "synthetic" and (args.glove_sn or args.user_id):
            argparser.error("--glove-sn and --user-id apply only to glove input")
    try:
        import wuji_sdk as sdk
        sdk.set_log_level("warn")
        if args.command == "discover":
            return discover(args, sdk)
        from .sources import GloveSource, SyntheticSource
        from .simulation import run
        source = (GloveSource(args.glove_sn, args.user_id, sdk=sdk) if args.input == "glove"
                  else SyntheticSource("right" if args.side == "auto" else args.side))
        try:
            if args.side != "auto" and source.side != args.side:
                raise ValueError(f"Selected glove is {source.side}, but --side is {args.side}")
            if args.input == "glove":
                print(f"Glove {source.serial}: side={source.side}, firmware={source.firmware}, address={source.address}")
            run(source, headless=args.headless, duration=args.duration, report_path=args.report)
        finally:
            source.close()
        return 0
    except KeyboardInterrupt:
        print("Stopped; connections closed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
