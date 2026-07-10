import os
from collections import defaultdict
import math
import re

from tabular_data_to_images.paths import MERGED_CSV_PATH

def to_float(value):
    try:
        return float(value)
    except ValueError:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)
        if numbers:
            return float(numbers[0])
        else:
            return 0.0
        
def update_stats(stats, src_ip, var, value):
    if src_ip not in stats:
        stats[src_ip] = {}
    if var not in stats[src_ip]:
        stats[src_ip][var] = {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'sum_sq': 0, 'count': 0}

    stats[src_ip][var]['min'] = min(stats[src_ip][var]['min'], value)
    stats[src_ip][var]['max'] = max(stats[src_ip][var]['max'], value)
    stats[src_ip][var]['sum'] += value
    stats[src_ip][var]['sum_sq'] += value ** 2
    stats[src_ip][var]['count'] += 1

def finalize_stats():
    final_stats = defaultdict(lambda: defaultdict(dict))
    for src_ip, vars in stats.items():
        for var, data in vars.items():
            mean = data['sum'] / data['count']
            variance = abs((data['sum_sq'] / data['count']) - (mean ** 2))  
            sd = math.sqrt(variance)
            final_stats[src_ip][var] = {'min': data['min'], 'max': data['max'], 'mean': mean, 'sd': sd}
    return final_stats
        
numeric_vars = ["duration", "packets_count", "handshake_duration", "fwd_packets_count", 
                "bwd_packets_count", "payload_bytes_max", "payload_bytes_min", "payload_bytes_mean", "payload_bytes_std", "payload_bytes_variance", "payload_bytes_median", "payload_bytes_skewness", "payload_bytes_cov", "payload_bytes_mode", "fwd_payload_bytes_max", "fwd_payload_bytes_min", "fwd_payload_bytes_mean", "fwd_payload_bytes_std", "fwd_payload_bytes_variance", "fwd_payload_bytes_median", "fwd_payload_bytes_skewness", "fwd_payload_bytes_cov", "fwd_payload_bytes_mode", "bwd_payload_bytes_max", "bwd_payload_bytes_min", "bwd_payload_bytes_mean", "bwd_payload_bytes_std", "bwd_payload_bytes_variance", "bwd_payload_bytes_median", "bwd_payload_bytes_skewness", "bwd_payload_bytes_cov", "bwd_payload_bytes_mode", "total_header_bytes", "max_header_bytes", "min_header_bytes", "mean_header_bytes", "std_header_bytes", "median_header_bytes", "skewness_header_bytes", "cov_header_bytes", "mode_header_bytes", "variance_header_bytes", "fwd_total_header_bytes", "fwd_max_header_bytes", "fwd_min_header_bytes", "fwd_mean_header_bytes", "fwd_std_header_bytes", "fwd_median_header_bytes", "fwd_skewness_header_bytes", "fwd_cov_header_bytes", "fwd_mode_header_bytes", "fwd_variance_header_bytes", "bwd_total_header_bytes", "bwd_max_header_bytes", "bwd_min_header_bytes", "bwd_mean_header_bytes", "bwd_std_header_bytes", "bwd_median_header_bytes", "bwd_skewness_header_bytes", "bwd_cov_header_bytes", "bwd_mode_header_bytes", "bwd_variance_header_bytes", "fwd_avg_segment_size", "bwd_avg_segment_size", "avg_segment_size", "fwd_init_win_bytes", "bwd_init_win_bytes", "active_min", "active_max", "active_mean", "active_std", "active_median", "active_skewness", "active_cov", "active_mode", "active_variance", "idle_min", "idle_max", "idle_mean", "idle_std", "idle_median", "idle_skewness", "idle_cov", "idle_mode", "idle_variance", "bytes_rate", "fwd_bytes_rate", "bwd_bytes_rate", "packets_rate", "bwd_packets_rate", "fwd_packets_rate", "down_up_rate", "avg_fwd_bytes_per_bulk", "avg_fwd_packets_per_bulk", "avg_fwd_bulk_rate", "avg_bwd_bytes_per_bulk", "avg_bwd_packets_bulk_rate", "avg_bwd_bulk_rate", "fwd_bulk_state_count", "fwd_bulk_per_packet", "fwd_bulk_duration", "bwd_bulk_state_count", "bwd_bulk_per_packet", "bwd_bulk_duration", "fin_flag_counts", "psh_flag_counts", "urg_flag_counts", "ece_flag_counts", "syn_flag_counts", "ack_flag_counts", "cwr_flag_counts", "rst_flag_counts", "fwd_fin_flag_counts", "fwd_psh_flag_counts", "fwd_urg_flag_counts", "fwd_ece_flag_counts", "fwd_syn_flag_counts", "fwd_ack_flag_counts", "fwd_cwr_flag_counts", "fwd_rst_flag_counts", "bwd_fin_flag_counts", "bwd_psh_flag_counts", "bwd_urg_flag_counts", "bwd_ece_flag_counts", "bwd_syn_flag_counts", "bwd_ack_flag_counts", "bwd_cwr_flag_counts", "bwd_rst_flag_counts", "fin_flag_percentage_in_total", "psh_flag_percentage_in_total", "urg_flag_percentage_in_total", "ece_flag_percentage_in_total", "syn_flag_percentage_in_total", "ack_flag_percentage_in_total", "cwr_flag_percentage_in_total", "rst_flag_percentage_in_total", "fwd_fin_flag_percentage_in_total", "fwd_psh_flag_percentage_in_total", "fwd_urg_flag_percentage_in_total", "fwd_ece_flag_percentage_in_total", "fwd_syn_flag_percentage_in_total", "fwd_ack_flag_percentage_in_total", "fwd_cwr_flag_percentage_in_total", "fwd_rst_flag_percentage_in_total", "bwd_fin_flag_percentage_in_total", "bwd_psh_flag_percentage_in_total", "bwd_urg_flag_percentage_in_total", "bwd_ece_flag_percentage_in_total", "bwd_syn_flag_percentage_in_total", "bwd_ack_flag_percentage_in_total", "bwd_cwr_flag_percentage_in_total", "bwd_rst_flag_percentage_in_total", "fwd_fin_flag_percentage_in_fwd_packets", "fwd_psh_flag_percentage_in_fwd_packets", "fwd_urg_flag_percentage_in_fwd_packets", "fwd_ece_flag_percentage_in_fwd_packets", "fwd_syn_flag_percentage_in_fwd_packets", "fwd_ack_flag_percentage_in_fwd_packets", "fwd_cwr_flag_percentage_in_fwd_packets", "fwd_rst_flag_percentage_in_fwd_packets", "bwd_fin_flag_percentage_in_bwd_packets", "bwd_psh_flag_percentage_in_bwd_packets", "bwd_urg_flag_percentage_in_bwd_packets", "bwd_ece_flag_percentage_in_bwd_packets", "bwd_syn_flag_percentage_in_bwd_packets", "bwd_ack_flag_percentage_in_bwd_packets", "bwd_cwr_flag_percentage_in_bwd_packets", "bwd_rst_flag_percentage_in_bwd_packets", "packets_IAT_mean", "packet_IAT_std", "packet_IAT_max", "packet_IAT_min", "packet_IAT_total", "packets_IAT_median", "packets_IAT_skewness", "packets_IAT_cov", "packets_IAT_mode", "packets_IAT_variance", "fwd_packets_IAT_mean", "fwd_packets_IAT_std", "fwd_packets_IAT_max", "fwd_packets_IAT_min", "fwd_packets_IAT_total", "fwd_packets_IAT_median", "fwd_packets_IAT_skewness", "fwd_packets_IAT_cov", "fwd_packets_IAT_mode", "fwd_packets_IAT_variance", "bwd_packets_IAT_mean", "bwd_packets_IAT_std", "bwd_packets_IAT_max", "bwd_packets_IAT_min", "bwd_packets_IAT_total", "bwd_packets_IAT_median", "bwd_packets_IAT_skewness", "bwd_packets_IAT_cov", "bwd_packets_IAT_mode", "bwd_packets_IAT_variance", "subflow_fwd_packets", "subflow_bwd_packets", "subflow_fwd_bytes", "subflow_bwd_bytes", "handshake_state", "min_bwd_packets_delta_time", "max_bwd_packets_delta_time", "mean_packets_delta_time", "mode_packets_delta_time", "variance_packets_delta_time", "std_packets_delta_time", "median_packets_delta_time", "skewness_packets_delta_time", "cov_packets_delta_time", "mean_bwd_packets_delta_time", "mode_bwd_packets_delta_time", "variance_bwd_packets_delta_time", "std_bwd_packets_delta_time", "median_bwd_packets_delta_time", "skewness_bwd_packets_delta_time", "cov_bwd_packets_delta_time", "min_fwd_packets_delta_time", "max_fwd_packets_delta_time", "mean_fwd_packets_delta_time", "mode_fwd_packets_delta_time", "variance_fwd_packets_delta_time", "std_fwd_packets_delta_time", "median_fwd_packets_delta_time", "skewness_fwd_packets_delta_time", "cov_fwd_packets_delta_time", "min_packets_delta_len", "max_packets_delta_len", "mean_packets_delta_len", "mode_packets_delta_len", "variance_packets_delta_len", "std_packets_delta_len", "median_packets_delta_len", "skewness_packets_delta_len", "min_bwd_packets_delta_len", "max_bwd_packets_delta_len", "mean_bwd_packets_delta_len", "mode_bwd_packets_delta_len", "variance_bwd_packets_delta_len", "std_bwd_packets_delta_len", "median_bwd_packets_delta_len", "skewness_bwd_packets_delta_len", "min_fwd_packets_delta_len", "max_fwd_packets_delta_len", "mean_fwd_packets_delta_len", "mode_fwd_packets_delta_len", "variance_fwd_packets_delta_len", "std_fwd_packets_delta_len", "median_fwd_packets_delta_len", "skewness_fwd_packets_delta_len", "min_header_bytes_delta_len", "max_header_bytes_delta_len", "mean_header_bytes_delta_len", "mode_header_bytes_delta_len", "variance_header_bytes_delta_len", "std_header_bytes_delta_len", "median_header_bytes_delta_len", "skewness_header_bytes_delta_len", "min_bwd_header_bytes_delta_len", "max_bwd_header_bytes_delta_len", "mean_bwd_header_bytes_delta_len", "mode_bwd_header_bytes_delta_len", "variance_bwd_header_bytes_delta_len", "std_bwd_header_bytes_delta_len", "median_bwd_header_bytes_delta_len", "skewness_bwd_header_bytes_delta_len", "min_fwd_header_bytes_delta_len", "max_fwd_header_bytes_delta_len", "mean_fwd_header_bytes_delta_len", "mode_fwd_header_bytes_delta_len", "variance_fwd_header_bytes_delta_len", "std_fwd_header_bytes_delta_len", "median_fwd_header_bytes_delta_len", "skewness_fwd_header_bytes_delta_len", "min_payload_bytes_delta_len", "max_payload_bytes_delta_len", "mean_payload_bytes_delta_len", "mode_payload_bytes_delta_len", "variance_payload_bytes_delta_len", "std_payload_bytes_delta_len", "median_payload_bytes_delta_len", "skewness_payload_bytes_delta_len", "min_bwd_payload_bytes_delta_len", "max_bwd_payload_bytes_delta_len", "mean_bwd_payload_bytes_delta_len", "mode_bwd_payload_bytes_delta_len", "variance_bwd_payload_bytes_delta_len", "std_bwd_payload_bytes_delta_len", "median_bwd_payload_bytes_delta_len", "skewness_bwd_payload_bytes_delta_len", "min_fwd_payload_bytes_delta_len", "max_fwd_payload_bytes_delta_len", "mean_fwd_payload_bytes_delta_len", "mode_fwd_payload_bytes_delta_len", "variance_fwd_payload_bytes_delta_len", "std_fwd_payload_bytes_delta_len", "median_fwd_payload_bytes_delta_len", "skewness_fwd_payload_bytes_delta_len", "cov_packets_delta_len", "cov_bwd_packets_delta_len", "cov_fwd_packets_delta_len", "cov_header_bytes_delta_len", "cov_bwd_header_bytes_delta_len", "cov_fwd_header_bytes_delta_len", "cov_payload_bytes_delta_len", "cov_bwd_payload_bytes_delta_len", "cov_fwd_payload_bytes_delta_len"]  # Add all numeric variables here

def file_exists(path):
    return os.path.exists(path)

for var in numeric_vars:
    print(f"Processing {var}")
    if file_exists(f"continous/{var}.csv"):
        print(f"Skipping {var}, file already exists.")
        continue

    stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'sum_sq': 0, 'count': 0})

    with open(MERGED_CSV_PATH, 'r') as file:
        header = next(file).strip().split(',')
        idx = {var: header.index(var) for var in ['src_ip', var]}  
        for line in file:
            columns = line.strip().split(',')
            src_ip = columns[idx['src_ip']]
            value = to_float(columns[idx[var]])
            update_stats(stats, src_ip, var, value)

    final_stats = finalize_stats()

    with open(f"continous/{var}.csv", 'w') as f:
        f.write(f'src_ip,{var}_min,{var}_max,{var}_mean,{var}_sd\n')
        for src_ip, vars in final_stats.items():
            if var in vars:
                f.write(f"{src_ip},{vars[var]['min']},{vars[var]['max']},{vars[var]['mean']},{vars[var]['sd']}\n")