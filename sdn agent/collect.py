"""
collect.py — Traffic Data Collection Script
============================================
Phase: Data Collection (run BEFORE training)

Purpose:
    Sniffs ICMP Echo Request packets arriving on the mirror interface (the port on
    PC3 that receives a copy of all traffic forwarded by OVS on the RPi). For every
    sliding 2-second window (stepped every 1 second), it computes per-source-IP
    feature vectors and appends labeled rows to a CSV file.

Usage:
    Run TWICE — once for each label:
        python collect.py --iface "Ethernet 2" --label 0 --duration 300  # normal traffic
        python collect.py --iface "Ethernet 2" --label 1 --duration 300  # attack traffic

    --iface     : Name of the mirror NIC as reported by Scapy on Windows.
                  Find yours by running: python -c "from scapy.arch.windows import
                  get_windows_if_list; [print(i['name'], i['description'])
                  for i in get_windows_if_list()]"
    --label     : 0 = normal, 1 = attack
    --duration  : Collection time in seconds (default: 300)
    --output    : Output CSV file path (default: traffic_data.csv)
    --pc3-ip    : IP address of PC3 itself, to filter out self-generated traffic
                  (default: auto-detect from the chosen interface)

Requirements:
    pip install scapy numpy pandas
    Npcap must be installed (https://npcap.com) — run with Administrator privileges.

Feature vector (per source IP, per window):
    icmp_count  — total ICMP Echo Requests from this source in the 2s window
    icmp_rate   — packets per second (count / WINDOW_DURATION)
    avg_pkt_size — mean byte length of captured ICMP frames
    iat_mean    — mean inter-arrival time between consecutive ICMP packets (seconds)
    iat_std     — standard deviation of inter-arrival time
    label       — 0 (normal) or 1 (attack), supplied by the operator via CLI

Note on icmp_ratio:
    Because a BPF filter of "icmp" is applied at the kernel level (Npcap/libpcap),
    all captured packets are already ICMP — so an ICMP-to-total ratio would always
    be 1.0 and adds no information. It is intentionally omitted here. This keeps
    the feature vector at 5 dimensions, which is clean and sufficient for KNN.

Note on self-traffic filtering:
    OVS with select-all=true also mirrors PC3's own frames back to it. We discard
    packets whose IP source is PC3's own IP so the agent's REST API calls (which
    generate response packets) do not pollute the training data.
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

# ---------------------------------------------------------------------------
# Scapy import — must happen AFTER Npcap is installed and as Administrator
# ---------------------------------------------------------------------------
try:
    from scapy.all import sniff, ICMP, IP, get_if_addr
    from scapy.arch.windows import get_windows_if_list
except ImportError:
    print("[ERROR] Scapy is not installed. Run: pip install scapy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants — match these in agent.py exactly
# ---------------------------------------------------------------------------
WINDOW_DURATION = 2.0   # seconds: lookback for each sliding window computation
STEP_SIZE       = 1.0   # seconds: how often the main loop runs the classifier

# CSV column order — must match FEATURES list in train.py and agent.py
CSV_COLUMNS = ["src_ip", "icmp_count", "icmp_rate", "avg_pkt_size",
               "iat_mean", "iat_std", "label"]

# ---------------------------------------------------------------------------
# Shared packet buffer — written by the sniffer thread, read by the main loop
# Each entry: (timestamp: float, src_ip: str, pkt_size: int)
# ---------------------------------------------------------------------------
packet_buffer: list[tuple[float, str, int]] = []
buffer_lock   = threading.Lock()


# ---------------------------------------------------------------------------
# Packet callback — called by Scapy's sniff() for every captured ICMP frame
# ---------------------------------------------------------------------------
def packet_callback(pkt):
    """
    Runs in the sniffer thread. Appends qualifying packets to the shared buffer.

    Filtering logic:
        - pkt.haslayer(IP)   : discard non-IP frames (raw Ethernet, ARP, etc.)
        - pkt.haslayer(ICMP) : redundant given the BPF filter, but defensive
        - ICMP type == 8     : Echo Request only; ignore Echo Replies (type 0)
                               which appear as mirrored return traffic
        - src != pc3_ip      : ignore PC3's own traffic (see module docstring)
    """
    global PC3_IP
    if pkt.haslayer(IP) and pkt.haslayer(ICMP):
        if pkt[ICMP].type == 8 and pkt[IP].src != PC3_IP:
            ts  = time.time()
            src = pkt[IP].src
            sz  = len(pkt)          # total frame length in bytes
            with buffer_lock:
                packet_buffer.append((ts, src, sz))


# ---------------------------------------------------------------------------
# Feature computation — identical logic must be reproduced in agent.py
# ---------------------------------------------------------------------------
def compute_features(pkts: list[tuple[float, int]]) -> dict:
    """
    Given a list of (timestamp, size) tuples for a single source IP within the
    current window, return a dict of the 5 scalar features.

    Args:
        pkts: list of (timestamp_float, packet_size_int) for one source IP

    Returns:
        dict with keys: icmp_count, icmp_rate, avg_pkt_size, iat_mean, iat_std
        Returns None if the list is empty (should not happen in practice).
    """
    if not pkts:
        return None

    count     = len(pkts)
    rate      = count / WINDOW_DURATION                  # packets / second

    sizes     = [sz for (_, sz) in pkts]
    avg_size  = float(np.mean(sizes))

    # Sort timestamps to compute inter-arrival times
    timestamps = sorted(ts for (ts, _) in pkts)
    if len(timestamps) > 1:
        iats     = np.diff(timestamps)                   # differences between consecutive timestamps
        iat_mean = float(np.mean(iats))
        iat_std  = float(np.std(iats))
    else:
        # Only one packet in this window — IAT is undefined; use 0 as sentinel
        # KNN will still classify correctly because count/rate are very low for normal
        iat_mean = 0.0
        iat_std  = 0.0

    return {
        "icmp_count"  : count,
        "icmp_rate"   : rate,
        "avg_pkt_size": avg_size,
        "iat_mean"    : iat_mean,
        "iat_std"     : iat_std,
    }


# ---------------------------------------------------------------------------
# CSV writer — opens in append mode so multiple runs merge into one file
# ---------------------------------------------------------------------------
def open_csv_writer(output_path: str):
    """
    Opens the CSV file for appending. Writes the header row only if the file
    does not yet exist or is empty. Returns (file_handle, csv.DictWriter).
    """
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    fh = open(output_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        print(f"[INFO] Created new CSV file: {output_path}")
    else:
        print(f"[INFO] Appending to existing CSV file: {output_path}")
    return fh, writer


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
def run_collection(iface: str, label: int, duration: float, output_path: str):
    """
    Runs the main collection loop for `duration` seconds.

    The sniffer runs in a daemon background thread so it never blocks the main
    loop. Every STEP_SIZE seconds, the main loop:
        1. Takes a snapshot of packets within the last WINDOW_DURATION seconds
        2. Groups them by source IP
        3. Computes features for each source IP
        4. Writes a labeled row to the CSV
        5. Prunes the packet buffer to prevent unbounded memory growth

    The loop exits cleanly after `duration` seconds or on KeyboardInterrupt.
    """
    print(f"\n[COLLECT] Starting collection — label={label}, duration={duration}s")
    print(f"[COLLECT] Sniffing on interface: '{iface}'")
    print(f"[COLLECT] Writing to: {output_path}")
    print(f"[COLLECT] PC3 self-IP (excluded): {PC3_IP}\n")
    print("Press Ctrl+C to stop early.\n")

    fh, writer = open_csv_writer(output_path)

    # -----------------------------------------------------------------------
    # Start the Scapy sniffer in a background daemon thread.
    # store=False: prevents Scapy from accumulating a growing list of packets
    #              in memory — critical for long collection sessions.
    # filter="icmp": BPF filter applied at the kernel (Npcap) level — only
    #                ICMP frames are passed to packet_callback; ARP, SSDP,
    #                TCP, etc. are dropped before reaching Python.
    # -----------------------------------------------------------------------
    sniff_thread = threading.Thread(
        target=lambda: sniff(
            iface   = iface,
            filter  = "icmp",           # kernel-level BPF filter (efficient)
            prn     = packet_callback,  # callback per packet
            store   = False,            # do not accumulate packets in RAM
            timeout = duration + 5,     # auto-stop slightly after main loop ends
        ),
        daemon=True,                    # thread dies when main thread exits
        name="SnifferThread",
    )
    sniff_thread.start()
    print("[SNIFFER] Background sniffer thread started.")

    start_time   = time.time()
    rows_written = 0

    try:
        while True:
            elapsed = time.time() - start_time

            # Stop condition
            if elapsed >= duration:
                print(f"\n[COLLECT] Duration reached ({duration}s). Stopping.")
                break

            # Sleep until next window step
            time.sleep(STEP_SIZE)
            now = time.time()

            # ------------------------------------------------------------------
            # Take a thread-safe snapshot of the packet buffer.
            # We also prune packets older than 2× WINDOW_DURATION to keep the
            # buffer from growing indefinitely during long sessions.
            # ------------------------------------------------------------------
            with buffer_lock:
                # Extract packets inside the current sliding window
                window_pkts = [
                    (ts, src, sz) for (ts, src, sz) in packet_buffer
                    if now - ts <= WINDOW_DURATION
                ]
                # Prune anything older than 2× the window (already processed)
                packet_buffer[:] = [
                    (ts, src, sz) for (ts, src, sz) in packet_buffer
                    if now - ts <= WINDOW_DURATION * 2
                ]

            # ------------------------------------------------------------------
            # Group packets in the window by source IP
            # ------------------------------------------------------------------
            per_src: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
            for (ts, src, sz) in window_pkts:
                per_src[src].append((ts, sz))

            # ------------------------------------------------------------------
            # Compute features and write one CSV row per source IP
            # ------------------------------------------------------------------
            for src_ip, pkts in per_src.items():
                feats = compute_features(pkts)
                if feats is None:
                    continue

                row = {"src_ip": src_ip, "label": label, **feats}
                writer.writerow(row)
                rows_written += 1

                # Live status line so the operator knows data is being collected
                print(f"  [t={elapsed:5.1f}s] {src_ip:15s} | "
                      f"count={feats['icmp_count']:4d} | "
                      f"rate={feats['icmp_rate']:7.2f} pps | "
                      f"iat_mean={feats['iat_mean']:.4f}s | "
                      f"label={label}")

            # Flush to disk periodically so data is not lost on abrupt exit
            fh.flush()

    except KeyboardInterrupt:
        print("\n[COLLECT] Interrupted by user.")

    finally:
        fh.close()
        print(f"\n[COLLECT] Done. {rows_written} rows written to '{output_path}'.")
        print("[COLLECT] Waiting for sniffer thread to finish...")
        sniff_thread.join(timeout=6)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDN SOAR — Data Collection Script (collect.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iface", required=True,
        help="Scapy interface name for the mirror port (e.g., 'Ethernet 2'). "
             "Find with: from scapy.arch.windows import get_windows_if_list; "
             "[print(i['name'], i['description']) for i in get_windows_if_list()]"
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
        help="Output CSV file (default: traffic_data.csv). Rows are appended "
             "so you can run for label=0 and label=1 into the same file."
    )
    parser.add_argument(
        "--pc3-ip", type=str, default=None,
        help="IP of PC3 to exclude from capture. Auto-detected if not set."
    )

    args = parser.parse_args()

    # Resolve PC3's own IP so self-traffic can be filtered in packet_callback
    if args.pc3_ip:
        PC3_IP = args.pc3_ip
    else:
        try:
            PC3_IP = get_if_addr(args.iface)
            if not PC3_IP or PC3_IP == "0.0.0.0":
                # Fallback: try socket
                import socket
                PC3_IP = socket.gethostbyname(socket.gethostname())
        except Exception:
            PC3_IP = ""  # no filtering — operator should specify --pc3-ip
        print(f"[INFO] Auto-detected PC3 IP: {PC3_IP} "
              f"(use --pc3-ip to override)")

    run_collection(
        iface       = args.iface,
        label       = args.label,
        duration    = args.duration,
        output_path = args.output,
    )
