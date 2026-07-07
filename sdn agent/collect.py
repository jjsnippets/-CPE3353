"""
collect.py: Traffic Data Collection Script

Purpose:
    Sniffs ICMP Echo Request packets arriving on the mirror interface. For every
    sliding 2-second window (stepped every 1 second), it computes per-source-IP
    feature vectors and appends labeled rows to a CSV file.

Usage:
    Run TWICE, once for each label:
        python collect.py --mirror-ip <control_ip> --label 0  # normal traffic
        python collect.py --mirror-ip <control_ip> --label 1  # attack traffic

    --mirror-ip : IP address assigned to the mirror NIC.
    --label     : 0 = normal, 1 = attack
    --duration  : Collection time in seconds (default: 300)
    --output    : Output CSV file path (default: traffic_data.csv)

Requirements:
    pip install scapy numpy pandas
    Npcap must be installed (https://npcap.com) — run with Administrator privileges.

Feature vector (per source IP, per window):
    icmp_count      : total ICMP Echo Requests from this source in the 2s window
    icmp_rate       : packets per second (count / WINDOW_DURATION)
    avg_pkt_size    : mean byte length of captured ICMP frames
    iat_mean        : mean inter-arrival time between consecutive ICMP packets (seconds)
    iat_std         : standard deviation of inter-arrival time
    label           : 0 (normal) or 1 (attack), supplied by the operator via CLI
"""

import argparse
import csv
import os
import sys
import threading
import time
import collections

import numpy as np
import pandas as pd

try:
    from scapy.all import sniff, ICMP, IP
    from scapy.arch.windows import get_windows_if_list
except ImportError:
    print("[ERROR] Scapy is not installed. Run: pip install scapy")
    sys.exit(1)

# Constants
# Must match with those in other files exactly
WINDOW_DURATION = 2.0   # seconds: lookback for each sliding window computation
STEP_SIZE       = 1.0   # seconds: how often the main loop runs the classifier
CSV_COLUMNS = ["src_ip", "icmp_count", "icmp_rate", "avg_pkt_size",
               "iat_mean", "iat_std", "label"]  # Features list and order

# Shared packet buffer
packet_buffer: list[tuple[float, str, int]] = []
buffer_lock   = threading.Lock()

def packet_callback(pkt):
    '''
    Packet callback for every captured ICMP frame.
    '''
    global MIRROR_IP

    # Discard non-IP frames (raw Ethernet, ARP, etc.)
    if pkt.haslayer(IP) and pkt.haslayer(ICMP):
        # Echo Request only; ignore Echo Replies (type 0)
        # also ignore the mirror host's own traffic
        if pkt[ICMP].type == 8 and pkt[IP].src != MIRROR_IP:
            ts  = time.time()       # current timestamp in seconds (float)
            src = pkt[IP].src       # source IP address (string)
            sz  = len(pkt)          # total frame length in bytes
            with buffer_lock:
                packet_buffer.append((ts, src, sz))

def compute_features(pkts: list[tuple[float, int]]) -> dict:
    '''
    Feature computation.
    '''
    if not pkts:
        return None

    # icmp_count: Number of packets in this window
    count     = len(pkts)

    # icmp_rate: Packets per second
    rate      = count / WINDOW_DURATION

    # avg_pkt_size: Mean packet size in bytes
    sizes     = [sz for (_, sz) in pkts]
    avg_size  = float(np.mean(sizes))           

    # iat_mean, iat_std: Mean and stddev of inter-arrival times
    timestamps = sorted(ts for (ts, _) in pkts)
    if len(timestamps) > 1:     # For 2 or more packets
        # Differences between consecutive timestamps
        iats     = np.diff(timestamps)
        iat_mean = float(np.mean(iats))
        iat_std  = float(np.std(iats))
    else:                       # Otherwise, return 0
        iat_mean = 0.0
        iat_std  = 0.0

    return {
        "icmp_count"  : count,
        "icmp_rate"   : rate,
        "avg_pkt_size": avg_size,
        "iat_mean"    : iat_mean,
        "iat_std"     : iat_std,
    }

def open_csv_writer(output_path: str):
    '''
    Appends results to a CSV file.
    '''
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    fh = open(output_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    
    if write_header:    # If file is new or empty, write the header row
        writer.writeheader()
        print(f"Created new CSV file: {output_path}")
    else:               # Otherwise, append to existing file
        print(f"Appending to existing CSV file: {output_path}")
    return fh, writer


def run_collection(iface: str, label: int, duration: float, output_path: str):
    """
    Main collection loop.
     - Takes a snapshot of packets within the last WINDOW_DURATION seconds
     - Groups them by source IP
     - Computes features for each source IP
     - Writes a labeled row to the CSV
    """
    print(f"\nStarting collection: label={label}, duration={duration}s")
    print(f"Mirror host IP: {MIRROR_IP} ('{iface}')")
    print(f"Writing to: {output_path}\n")

    fh, writer = open_csv_writer(output_path)

    # Sniffer thread in the background
    sniff_thread = threading.Thread(
        target=lambda: sniff(
            iface   = iface,            # Scapy interface name
            filter  = "icmp",           # Non-ICMP frames are dropped
            prn     = packet_callback,  # Function called for each captured packet
            store   = False,            # do not accumulate packets in memory
            timeout = duration + 5,     # auto-stop slightly after main loop ends
        ),
        daemon=True,                    # thread dies when main thread exits
        name="SnifferThread",
    )
    sniff_thread.start()
    print("Background sniffer thread started.")

    # Main collection loop
    start_time   = time.time()
    rows_written = 0

    try:
        while True:
            # Stop condition
            elapsed = time.time() - start_time
            if elapsed >= duration:
                print(f"\nDuration reached ({duration}s)!\n")
                break

            # Sleep until next window step
            time.sleep(STEP_SIZE)
            now = time.time()

            # Take a thread-safe snapshot of the packet buffer
            with buffer_lock:
                # Extract packets inside the current sliding window
                window_pkts = [
                    (ts, src, sz) for (ts, src, sz) in packet_buffer
                    if now - ts <= WINDOW_DURATION
                ]
                # Prune anything older than 2× the window
                packet_buffer[:] = [
                    (ts, src, sz) for (ts, src, sz) in packet_buffer
                    if now - ts <= WINDOW_DURATION * 2
                ]

            # Group packets in the window by source IP
            per_src: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
            for (ts, src, sz) in window_pkts:
                per_src[src].append((ts, sz))

            # Compute features and write one CSV row per source IP
            for src_ip, pkts in per_src.items():
                # Compute features for this source IP
                feats = compute_features(pkts)
                if feats is None:
                    continue

                # Write a labeled row to the CSV
                row = {"src_ip": src_ip, "label": label, **feats}
                writer.writerow(row)
                rows_written += 1

                # Live status display
                print(f"  [t={elapsed:5.1f}s] {src_ip:15s} | "
                      f"count={feats['icmp_count']:4d} | "
                      f"rate={feats['icmp_rate']:7.2f} pps | "
                      f"avg_size={feats['avg_pkt_size']:7.2f}B | "
                      f"iat_mean={feats['iat_mean']:.4f}s | "
                      f"iat_std={feats['iat_std']:.4f}s | "
                      f"label={label}")

            fh.flush()

    except KeyboardInterrupt:
        print("\nCollection stopped by user!\n")

    finally:
        fh.close()
        print("\nData collection finished.")
        print(f"{rows_written} rows written to '{output_path}'.")
        sniff_thread.join(timeout=6)
        print("Sniffer thread terminated.")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDN SOAR Agent: Data Collection Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mirror-ip", required=True,
        help="IP address assigned to the mirror NIC"
    )
    parser.add_argument(
        "--label", type=int, required=True, choices=[0, 1],
        help="Traffic label: 0 = normal, 1 = attack (ping flood)"
    )
    parser.add_argument(
        "--duration", type=float, default=300.0,
        help="How many seconds to collect data (default: 300)"
    )
    parser.add_argument(
        "--output", type=str, default="traffic_data.csv",
        help="Output CSV file (default: traffic_data.csv)"
    )
    args = parser.parse_args()

    # --mirror-ip should exist
    MIRROR_IP = args.mirror_ip
    try:
        resolved_iface = None
        for iface in get_windows_if_list():
            name = iface.get("name", "")
            ips = iface.get("ips", [])

            normalized_ips = {ip.split("/")[0] for ip in ips}
            if MIRROR_IP in normalized_ips:
                resolved_iface = name
                break

        if not resolved_iface:
            raise ValueError(f"Could not find a Scapy interface that has IP {MIRROR_IP}")
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # Start the main collection loop
    run_collection(
        iface       = resolved_iface,
        label       = args.label,
        duration    = args.duration,
        output_path = args.output,
    )
