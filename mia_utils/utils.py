import os
import re
import glob


def get_superset_file(config, random=False):
    if random==True:
        output_dir = 'results/target_docs'
        # Extract values for Top and N from args.name using regular expressions
        match = re.search(r'(.*?)-Top(\d+)-M(\d+)-N(\d+)', config.attack_config.name)
        if not match:
            print("Invalid filename format. Expected format: 'prefix-TopX-MY-NZ'")
            return

        prefix = match.group(1)  # Everything before -Top
        desired_top_k = int(match.group(2))
        m_value = match.group(3)
        n_value = match.group(4)
        suffix = f'-M{m_value}-N{n_value}.json'

        print(f"Extracted Prefix: {prefix}, Desired Top-k: {desired_top_k}, Suffix: {suffix}")

        # Find all available files in the directory that match the same prefix and suffix
        print(f'{output_dir}/{prefix}-Top*{suffix}')
        files = glob.glob(f'{output_dir}/{prefix}-Top*{suffix}')
        available_top_ks = []
        print(f"args.name: {config.attack_config.name}")
        print(f"Matching files found: {files}")

        for file in files:
            file_match = re.search(r'-Top(\d+)-M(\d+)-N(\d+).json$', file)
            if file_match:
                top_k_value = int(file_match.group(1))
                # Consider only files with a higher Top-k value
                if top_k_value > desired_top_k:
                    available_top_ks.append((top_k_value, file))

        # Sort the available top-k values and select the smallest valid superset
        available_top_ks.sort()
        superset_filename = None
        for top_k_value, file in available_top_ks:
            if top_k_value > desired_top_k:
                superset_filename = file
                break

        if not superset_filename:
            print("No suitable superset file found. Cannot perform post-filtering.")
            return
    else:
        output_dir = 'results/target_docs'
        # Extract values for Top and N from args.name using regular expressions
        match = re.search(r'-Top(\d+)-M\d+-N(\d+)$', config.attack_config.name)
        if not match:
            print("Invalid filename format. Expected format: 'base-TopX-MY-NZ'")
            return
        desired_top_k = int(match.group(1))
        n_value = max(desired_top_k, int(match.group(2)))
        superset_filename = f'{output_dir}/{config.attack_config.name.replace(f"-Top{desired_top_k}", f"-Top{n_value}")}.json'

        if not os.path.exists(superset_filename):
            print(f"Superset file {superset_filename} does not exist. Cannot perform post-filtering.")
            return
    
    print(superset_filename)
    return superset_filename
