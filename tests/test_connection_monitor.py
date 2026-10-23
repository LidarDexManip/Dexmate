"""Connection monitor parsing and per-link verdicts, without touching the network."""

import unittest
from unittest import mock

from wuji_teleop import connection_monitor as monitor
from wuji_teleop.connection_monitor import Check


IP_ADDR = """\
2: enp128s31f6    inet 192.168.50.10/24 brd 192.168.50.255 scope global enp128s31f6
2: enp128s31f6    inet 192.168.1.50/32 scope global enp128s31f6
3: wlx7419f817622a    inet 192.168.1.194/24 brd 192.168.1.255 scope global wlx7419f817622a
7: enx8cae4cebd67f    inet 192.168.1.50/32 scope global enx8cae4cebd67f
"""
HANDS = [
    {"sn": "WH2JA01260722011", "type": "DeviceType.WujiHand2",
     "address": "192.168.77.110:7447", "ip": "192.168.77.110"},
    {"sn": "WH2KA01260827009", "type": "DeviceType.WujiHand2",
     "address": "192.168.1.111:50001", "ip": "192.168.1.111"},
]
GLOVES = [
    {"sn": "WG1JA06260812032", "type": "DeviceType.WujiGlove",
     "address": "192.168.1.100:50001", "ip": "192.168.1.100"},
    {"sn": "WG1KA06260815543", "type": "DeviceType.WujiGlove",
     "address": "192.168.1.101:50001", "ip": "192.168.1.101"},
]


def _by_name(checks):
    return {check.name: check for check in checks}


class ParsingTests(unittest.TestCase):
    def test_route_device(self):
        self.assertEqual(monitor.route_device(
            "x", "192.168.1.100 dev enx8cae4cebd67f src 192.168.1.50 uid 1001\n    cache"),
            "enx8cae4cebd67f")
        self.assertIsNone(monitor.route_device("x", ""))

    def test_interface_addresses(self):
        self.assertEqual(monitor.interface_addresses(IP_ADDR), [
            ("enp128s31f6", "192.168.50.10"), ("enp128s31f6", "192.168.1.50"),
            ("wlx7419f817622a", "192.168.1.194"), ("enx8cae4cebd67f", "192.168.1.50")])


class NetworkCheckTests(unittest.TestCase):
    def _checks(self, routes):
        addresses = monitor.interface_addresses(IP_ADDR)
        with mock.patch.object(monitor, "is_wireless", lambda name: name.startswith("wl")), \
                mock.patch.object(monitor, "route_device", lambda ip: routes[ip]), \
                mock.patch.object(monitor, "wifi_ssid", return_value="LIDAR_WiFi_5G"), \
                mock.patch.object(monitor, "unconfigured_wired_interfaces",
                                  return_value=["enx8cae4cebd67f"]):
            return _by_name(monitor.network_checks(addresses))

    def test_wired_glove_routes_pass(self):
        checks = self._checks({"192.168.1.100": "enx8cae4cebd67f",
                               "192.168.1.101": "enx8cae4cebd67f"})
        self.assertEqual(checks["Wi-Fi (Pico)"].status, "ok")
        self.assertIn("LIDAR_WiFi_5G", checks["Wi-Fi (Pico)"].detail)
        self.assertEqual(checks["left glove route"].status, "ok")
        self.assertTrue(all(check.status == "ok" for check in checks.values()))

    def test_glove_route_over_wifi_fails_with_the_configure_command(self):
        checks = self._checks({"192.168.1.100": "wlx7419f817622a",
                               "192.168.1.101": "wlx7419f817622a"})
        check = checks["right glove route"]
        self.assertEqual(check.status, "fail")
        self.assertIn("configure_wuji_network.sh enx8cae4cebd67f", check.hint)


class AmbiguousSourceTests(unittest.TestCase):
    def test_source_on_two_interfaces_is_unsafe(self):
        reason = monitor.ambiguous_hand_source(
            "192.168.1.111", monitor.interface_addresses(IP_ADDR),
            "192.168.1.111 dev enp128s31f6 src 192.168.1.50 uid 1001")
        self.assertIn("enp128s31f6 and enx8cae4cebd67f", reason)

    def test_unique_source_is_safe(self):
        self.assertIsNone(monitor.ambiguous_hand_source(
            "192.168.50.20", monitor.interface_addresses(IP_ADDR),
            "192.168.50.20 dev enp128s31f6 src 192.168.50.10 uid 1001"))


class DeviceCheckTests(unittest.TestCase):
    def _checks(self, scanned, down=(), unsafe=None):
        with mock.patch.object(monitor, "ping", lambda ip: ip not in down), \
                mock.patch.object(monitor, "ambiguous_hand_source",
                                  lambda ip: unsafe if ip == "192.168.1.111" else None):
            return _by_name(monitor.device_checks(scanned))

    def test_hand_on_an_ambiguous_address_fails_with_the_subnet_fix(self):
        checks = self._checks(HANDS + GLOVES, unsafe="laptop 192.168.1.50 is on two interfaces")
        self.assertEqual(checks["right hand"].status, "fail")
        self.assertEqual(checks["right hand"].hint, monitor.HAND_SUBNET_HINT)
        self.assertEqual(checks["left hand"].status, "ok")

    def test_all_devices_present(self):
        checks = self._checks(HANDS + GLOVES)
        self.assertTrue(all(check.status == "ok" for check in checks.values()), checks)
        self.assertIn("WH2KA01260827009", checks["right hand"].detail)

    def test_missing_hand_and_silent_glove_are_reported(self):
        checks = self._checks(HANDS[:1] + GLOVES, down={"192.168.1.100", "192.168.50.21"})
        self.assertEqual(checks["right hand"].status, "fail")
        self.assertIn("12 V", checks["right hand"].hint)
        self.assertEqual(checks["left glove"].status, "fail")
        self.assertEqual(checks["robot SoC"].status, "fail")
        self.assertEqual(checks["left hand"].status, "ok")

    def test_glove_answering_ping_but_not_discovery_warns(self):
        checks = self._checks(HANDS + GLOVES[1:])
        self.assertEqual(checks["left glove"].status, "warn")

    def test_two_hands_on_one_side_warn(self):
        extra = dict(HANDS[1], sn="WH2KA99999999999", ip="192.168.1.112",
                     address="192.168.1.112:50001")
        checks = self._checks(HANDS + [extra] + GLOVES)
        self.assertEqual(checks["right hand"].status, "warn")
        self.assertIn("--right-hand-sn", checks["right hand"].hint)


class RenderTests(unittest.TestCase):
    def test_hints_only_for_problems_and_summary_counts(self):
        text = monitor.render([
            Check("Hands", "left hand", "ok", "fine", "unused hint"),
            Check("Hands", "right hand", "fail", "missing", "power it"),
            Check("Pico", "left hand tracking", "warn", "no frames", "enter AR"),
        ], title="T")
        self.assertNotIn("unused hint", text)
        self.assertIn("→ power it", text)
        self.assertIn("1 missing, 1 warning(s).", text)
        self.assertIn("All connections OK.", monitor.render(
            [Check("Hands", "left hand", "ok", "fine")], title="T"))


if __name__ == "__main__":
    unittest.main()
