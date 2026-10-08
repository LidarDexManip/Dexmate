#!/usr/bin/env bash
# Temporary, dedicated glove connection; leave Wi-Fi/default routing alone.
set -euo pipefail

wuji_interface="${1:-enp3s0}"
wuji_profile="wuji-glove-sim-${wuji_interface}"

if [[ ! -e "/sys/class/net/${wuji_interface}/carrier" ]]; then
  echo "Ethernet interface not found: ${wuji_interface}" >&2
  exit 1
fi
if [[ "$(cat "/sys/class/net/${wuji_interface}/carrier")" != 1 ]]; then
  echo "No Ethernet carrier on ${wuji_interface}. Check the powered glove adapter and cable." >&2
  exit 1
fi
if ip -4 -o address show dev "$wuji_interface" | rg -q 'inet '; then
  echo "${wuji_interface} already has an IPv4 address. Inspect it before changing networks:" >&2
  ip -4 address show dev "$wuji_interface"
  exit 1
fi
if ip -4 route show 192.168.1.0/24 | rg -q .; then
  echo "A route to 192.168.1.0/24 already exists. Resolve the subnet overlap first." >&2
  exit 1
fi
if nmcli -g connection.id connection show "$wuji_profile" >/dev/null 2>&1; then
  echo "Profile ${wuji_profile} already exists; inspect or remove that temporary profile first." >&2
  exit 1
fi

wuji_previous_default="$(ip -4 route show default)"
# save no keeps this profile in memory only. DAD rejects a duplicate .50 address.
nmcli connection add save no type ethernet ifname "$wuji_interface" \
  con-name "$wuji_profile" connection.autoconnect no \
  ipv4.method manual ipv4.addresses 192.168.1.50/24 \
  ipv4.never-default yes ipv4.dad-timeout 3000 ipv6.method disabled
if ! nmcli --wait 15 connection up "$wuji_profile"; then
  nmcli connection delete "$wuji_profile"
  exit 1
fi
if [[ "$(ip -4 route show default)" != "$wuji_previous_default" ]]; then
  echo "Default route changed during configuration; removing the glove connection." >&2
  nmcli connection down "$wuji_profile"
  nmcli connection delete "$wuji_profile"
  exit 1
fi
ip -4 address show dev "$wuji_interface"
echo "Glove network ready. To remove: nmcli connection delete '${wuji_profile}'"
