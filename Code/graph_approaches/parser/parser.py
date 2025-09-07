# +
import os
from datetime import datetime, timedelta

from scapy.all import PcapReader


def getLabel(packet, label_rules, protocol, dt):
    for label, conditions in label_rules.items():
        cond_ok = True
        for condition, value in conditions.items():
            match condition:
                case "ip":
                    if packet["IP"].src not in value and packet["IP"].dst not in value:
                        cond_ok = False
                        break
                case "start_time":
                    if dt < value:
                        cond_ok = False
                        break
                case "end_time":
                    if dt >= value:
                        cond_ok = False
                        break
                case "protocol":
                    if protocol not in value:
                        cond_ok = False
                        break
                case "source_port":
                    if packet[protocol].sport not in value:
                        cond_ok = False
                        break
                case "destination_port":
                    if packet[protocol].dport not in value:
                        cond_ok = False
                        break
        if cond_ok:
            return label
    # Probably mistake in the dataset
    return "skip"


def packetHandler(
    packets,
    label_rules,
    time_difference,
    packetsfile,
    count,
    packet_threshold,
    csv_path,
    file_count,
):
    for packet in packets:
        if packet.haslayer("TCP") and packet.haslayer("IP"):
            # Compute timestamp and account for timezone of original author for the later labeling of packets based on time
            dt = datetime.utcfromtimestamp(float(packet.time)) - timedelta(
                hours=time_difference
            )
            label = getLabel(packet, label_rules, "TCP", dt)
            if label != "skip":  # Potential mistake in dataset
                if count >= packet_threshold:
                    packetsfile.close()
                    packetsfile, file_count = openNewCSVFile(csv_path, file_count)
                    count = 1
                packetsfile.write(
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{dt.microsecond:06d},{packet['IP'].src},{packet['IP'].dst},{packet['TCP'].sport},{packet['TCP'].dport},{len(packet)},TCP {packet['TCP'].flags},{label}\n"
                )
                count += 1
        elif packet.haslayer("UDP") and packet.haslayer("IP"):
            dt = datetime.utcfromtimestamp(float(packet.time)) - timedelta(
                hours=time_difference
            )
            label = getLabel(packet, label_rules, "UDP", dt)
            if label != "skip":  # Potential mistake in dataset
                if count >= packet_threshold:
                    packetsfile.close()
                    packetsfile, file_count = openNewCSVFile(csv_path, file_count)
                    count = 1
                packetsfile.write(
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{dt.microsecond:06d},{packet['IP'].src},{packet['IP'].dst},{packet['UDP'].sport},{packet['UDP'].dport},{len(packet)},UDP,{label}\n"
                )
                count += 1
    return packetsfile, file_count, count


def openNewCSVFile(csv_path, file_count):
    packetsfile = open(csv_path + f"/{file_count}_packets.csv", "w")
    packetsfile.write(
        "Timestamp,Source IP,Destination IP,Source Port,Destination Port,Length,Protocol/Flags,Label\n"
    )
    packetsfile.flush()
    file_count += 1
    return packetsfile, file_count


def pcapsToCSVs(
    pcaps_path,
    pcaps_list,
    csv_path,
    packet_threshold,
    label_rules,
    time_difference,
):
    file_count = 0
    packetsfile, file_count = openNewCSVFile(csv_path, file_count)
    i = 1
    count = 1
    for pcap_name in pcaps_list:
        print(f"{i}/{len(pcaps_list)} Exporting data from {pcap_name}")
        pcap_path = os.path.join(pcaps_path, pcap_name)
        pcap = PcapReader(
            pcap_path
        )  # Streaming pcaps with PcapReader due to memory constraints. Otherwise, use rdpcap from scapy (drop in replacement) to load entire pcaps into memory
        packetsfile, file_count, count = packetHandler(
            pcap,
            label_rules,
            time_difference,
            packetsfile,
            count,
            packet_threshold,
            csv_path,
            file_count,
        )
        packetsfile.flush()
        i += 1
    packetsfile.close()
