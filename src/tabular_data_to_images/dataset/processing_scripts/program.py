from collections import defaultdict

from tabular_data_to_images.paths import MERGED_CSV_PATH

ip_counts = defaultdict(int)

with open(MERGED_CSV_PATH, 'r') as file:
    next(file)   
    for line in file:
        columns = line.split(',')
        src_ip = columns[2]  
        ip_counts[src_ip] += 1   

with open('N.csv', 'w') as out_file:
    out_file.write('src_ip,N\n')   
    for src_ip, count in ip_counts.items():
        out_file.write(f'{src_ip},{count}\n')  