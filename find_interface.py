"""
find_interface.py — Find the right network interface name for Gloo on Windows
=============================================================================

Run this ONCE to find the correct interface name for your machine:
    python find_interface.py

Then use whatever it prints in run_windows.bat
"""

import socket
import subprocess
import sys

print("Finding the right network interface for Gloo on Windows...\n")

# Method 1: get the local hostname's IP
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
    print(f"Your hostname : {hostname}")
    print(f"Your local IP : {local_ip}")
except Exception as e:
    local_ip = "127.0.0.1"
    print(f"Could not resolve hostname ({e}), using 127.0.0.1")

# Method 2: try connecting outward to find the real interface IP
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    outbound_ip = s.getsockname()[0]
    s.close()
    print(f"Outbound IP   : {outbound_ip}")
except Exception:
    outbound_ip = local_ip

print()

# Method 3: run ipconfig and parse interface names
try:
    result = subprocess.run(["ipconfig"], capture_output=True, text=True)
    lines  = result.stdout.splitlines()
    current_iface = None
    found = []
    for line in lines:
        # Interface section header e.g. "Ethernet adapter Ethernet:"
        if "adapter" in line.lower() and line.strip().endswith(":"):
            current_iface = line.strip().rstrip(":").split("adapter")[-1].strip()
        # IPv4 line
        if "IPv4" in line and current_iface:
            ip = line.split(":")[-1].strip()
            found.append((current_iface, ip))

    if found:
        print("Network interfaces found on your machine:")
        for name, ip in found:
            print(f"  [{name}]  →  {ip}")
        print()

        # Pick the best one: prefer the outbound IP match, then local_ip match
        best = None
        for name, ip in found:
            if ip == outbound_ip:
                best = (name, ip)
                break
        if not best:
            for name, ip in found:
                if ip == local_ip:
                    best = (name, ip)
                    break
        if not best:
            best = found[0]  # just take the first one

        print(f"RECOMMENDED interface: [{best[0]}]  (IP: {best[1]})")
        print()
        print("=" * 60)
        print("Add this line to run_windows.bat (replace the existing")
        print("GLOO_SOCKET_IFNAME line):")
        print()
        print(f'  set GLOO_SOCKET_IFNAME={best[0]}')
        print()
        print("And set MASTER_ADDR to your IP:")
        print(f'  set MASTER_ADDR={best[1]}')
        print("=" * 60)
    else:
        print("Could not parse ipconfig output.")
        print("Try setting:  set GLOO_SOCKET_IFNAME=Ethernet")
        print("or:           set GLOO_SOCKET_IFNAME=Wi-Fi")

except Exception as e:
    print(f"ipconfig failed: {e}")
    print("Manually run 'ipconfig' and look for your active adapter name.")
