#!/usr/bin/env bash
# Temporary, dedicated glove connection; leave Wi-Fi/default routing alone.
# The gloves are at 192.168.1.100/.101 and need the laptop at 192.168.1.50.
# When Wi-Fi (for the Pico) is also on 192.168.1.0/24, only those two glove
# addresses are routed over Ethernet so the Pico keeps using Wi-Fi.
set -euo pipefail

wuji_interface="${1:-}"
default_glove_ips=("192.168.1.100" "192.168.1.101")
glove_ips=("${@:2}")
explicit_glove_ips=false
if (( ${#glove_ips[@]} )); then
  explicit_glove_ips=true
else
  glove_ips=("${default_glove_ips[@]}")
fi
laptop_ip="192.168.1.50"

if [[ "$wuji_interface" == "--help" || "$wuji_interface" == "-h" ]]; then
  echo "Usage: $0 [ETHERNET_INTERFACE] [GLOVE_IP ...]"
  echo "Defaults: first wired interface with a link; routes both 192.168.1.100 and .101."
  echo "For two Ethernet ports, run once per port with that port's glove IP."
  echo "Profiles are in memory only (gone after reboot)."
  exit 0
fi
for glove_ip in "${glove_ips[@]}"; do
  if [[ "$glove_ip" != "192.168.1.100" && "$glove_ip" != "192.168.1.101" ]]; then
    echo "Unsupported glove IP: ${glove_ip} (expected 192.168.1.100 or 192.168.1.101)" >&2
    exit 2
  fi
done
if [[ -z "$wuji_interface" ]]; then
  for iface in /sys/class/net/*; do
    name="$(basename "$iface")"
    [[ -d "$iface/wireless" || ! -e "$iface/device" ]] && continue
    if [[ "$(cat "$iface/carrier" 2>/dev/null)" == 1 ]]; then
      wuji_interface="$name"
      break
    fi
  done
  if [[ -z "$wuji_interface" ]]; then
    echo "No wired interface with a link. Check the powered glove adapter and cable." >&2
    exit 1
  fi
fi
wuji_profile="wuji-glove-sim-${wuji_interface}"

if [[ ! -e "/sys/class/net/${wuji_interface}/carrier" ]]; then
  echo "Ethernet interface not found: ${wuji_interface}" >&2
  exit 1
fi
if [[ "$(cat "/sys/class/net/${wuji_interface}/carrier")" != 1 ]]; then
  echo "No Ethernet carrier on ${wuji_interface}. Check the powered glove adapter and cable." >&2
  exit 1
fi
routes_ready=true
for glove_ip in "${glove_ips[@]}"; do
  if ! ip -4 route get "$glove_ip" | grep -q "dev ${wuji_interface}" ||
      ! ping -c 1 -W 1 "$glove_ip" >/dev/null 2>&1; then
    routes_ready=false
  fi
done
if [[ "$routes_ready" == true ]]; then
  echo "Glove route(s) already use ${wuji_interface}; all requested gloves are reachable."
  exit 0
fi
if nmcli -g connection.id connection show "$wuji_profile" >/dev/null 2>&1; then
  nmcli connection delete "$wuji_profile" >/dev/null
fi
if ip -4 -o address show dev "$wuji_interface" | grep -q 'inet '; then
  echo "${wuji_interface} already has an IPv4 address from another network profile:" >&2
  ip -4 address show dev "$wuji_interface" >&2
  echo "Disconnect it first (nmcli device disconnect ${wuji_interface}), then rerun this script." >&2
  exit 1
fi

if ip -4 route show 192.168.1.0/24 | grep -q .; then
  route_csv=""
  for glove_ip in "${glove_ips[@]}"; do
    route_csv+="${route_csv:+,}${glove_ip}/32"
  done
  echo "192.168.1.0/24 is already on another interface (likely the Pico Wi-Fi); routing ${glove_ips[*]} over ${wuji_interface}."
  addresses="${laptop_ip}/32"
  routes="$route_csv"
else
  addresses="${laptop_ip}/24"
  routes=""
fi

wuji_previous_default="$(ip -4 route show default)"
# save no keeps this profile in memory only. DAD rejects a duplicate .50 address.
nmcli connection add save no type ethernet ifname "$wuji_interface" \
  con-name "$wuji_profile" connection.autoconnect no \
  ipv4.method manual ipv4.addresses "$addresses" ipv4.routes "$routes" \
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
unreachable=()
for glove_ip in "${glove_ips[@]}"; do
  if ping -c 2 -W 1 "$glove_ip" >/dev/null 2>&1; then
    echo "Glove network ready: ${glove_ip} answers."
  else
    echo "Network configured, but ${glove_ip} does not answer yet. Check glove power, then: ping ${glove_ip}" >&2
    unreachable+=("$glove_ip")
  fi
done
echo "To remove: nmcli connection delete '${wuji_profile}'"
if [[ "$explicit_glove_ips" == true && ${#unreachable[@]} -gt 0 ]]; then
  exit 1
fi
if [[ "$explicit_glove_ips" == false && " ${unreachable[*]} " == *" 192.168.1.100 "* ]]; then
  exit 1
fi
