from collections import defaultdict

from tabular_data_to_images.paths import MERGED_CSV_PATH

def process_variable(variable_name, data_dict):
    all_values = set(val for counts in data_dict.values() for val in counts)
    all_values = sorted(all_values)
    
    with open(f'categorical/{variable_name}.csv', 'w') as f:
        f.write(f'src_ip,{",".join(all_values)}\n')
        for src_ip, counts in data_dict.items():
            row = [str(counts.get(val, 0)) for val in all_values]
            f.write(f'{src_ip},{",".join(row)}\n')

data_dicts = {
    'src_port': defaultdict(lambda: defaultdict(int)),
    'dst_port': defaultdict(lambda: defaultdict(int)),
    'protocol': defaultdict(lambda: defaultdict(int)),
    'label': defaultdict(lambda: defaultdict(int)),
    'activity': defaultdict(lambda: defaultdict(int)), 
}

well_known_non_technical = {80, 443, 25, 110, 143, 53, 21}
well_known_technical = {22, 23, 20, 69, 161, 162, 123, 389, 514, 67, 68, 135, 445}
registered_ports = range(1024, 49152)
dynamic_ports = range(49152, 65536)

well_known_others = {port for port in range(0, 1024)} - well_known_non_technical - well_known_technical

def classify_dst_port(port):
    port = int(port)
    if port in well_known_non_technical:
        return 'well_known_non_technical'
    elif port in well_known_technical:
        return 'well_known_technical'
    elif port in well_known_others:
        return 'well_known_others'
    elif port in registered_ports:
        return 'registered'
    elif port in dynamic_ports:
        return 'dynamic'
    else:
        return 'unknown'

def classify_src_port(port):
    port = int(port)
    if port in dynamic_ports:
        return 'dynamic'
    elif port in registered_ports:
        return 'registered'
    elif port in well_known_non_technical or port in well_known_technical or port in well_known_others:
        return 'well_known'
    else:
        return 'unknown'

with open(MERGED_CSV_PATH, 'r') as file:
    header = next(file).strip().split(',')
    idx = {var: header.index(var) for var in data_dicts.keys() | {'src_ip'}}
    
    for line in file:
        columns = line.strip().split(',')
        src_ip = columns[idx['src_ip']]
        for var in data_dicts:
            value = columns[idx[var]]
            if var == 'dst_port':
                category = classify_dst_port(value)
                data_dicts[var][src_ip][var + "_" + category] += 1
            elif var == 'src_port':
                category = classify_src_port(value)
                data_dicts[var][src_ip][var + "_" + category] += 1
            else:
                data_dicts[var][src_ip][var + "_" + value] += 1

for var, data_dict in data_dicts.items():
    process_variable(var, data_dict)
