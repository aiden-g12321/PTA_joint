"Script to simplify NANOGrav .par files and save in par folder."

import os
import re

# Define the parameters to retain
params_to_keep = {
    "PSR", "EPHEM", "CLOCK", "UNITS", "INFO", "TIMEEPH", "T2CMETHOD",
    "DILATEFREQ", "CHI2", "ELONG", "ELAT", "PMELONG", "PMELAT", "PX",
    "ECL", "POSEPOCH", "F0", "F1", "PEPOCH"
}

# Define the epoch value to set for all epoch parameters
new_epoch_value = "53000.0000000000000000"

# Process a single .par file
def process_par_file(input_path):
    psr_name = None
    output_lines = []
    
    with open(input_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            
            param_name = line.split()[0]
            if param_name in params_to_keep:
                if param_name == "PSR":
                    psr_name = line.split()[1]
                    
                    # Check if the pulsar is observed by GBT or AO
                    if psr_name.lower().endswith("ao"):
                        print(f"Skipping {psr_name} (observed by Arecibo)")
                        return  # Skip Arecibo files
                    elif psr_name.lower().endswith("gbt"):
                        psr_name = psr_name[:-3]  # Remove 'gbt' suffix
                        line = f"{param_name:20} {psr_name}\n"

                # Modify epoch values to the new value
                if re.match(r".*EPOCH", param_name):
                    line = f"{param_name:20} {new_epoch_value}\n"
                
                output_lines.append(line)
    
    if psr_name is None:
        raise ValueError(f"PSR parameter not found in {input_path}")
    
    # Define the output file name using the psr_name without the 'gbt' suffix
    output_path = f"par/{psr_name}.par"
    
    with open(output_path, "w") as file:
        file.writelines(output_lines)

    print(f"Processed {input_path} -> {output_path}")

# Process all .par files in a directory
def process_all_par_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".par"):
            input_path = os.path.join(directory, filename)
            try:
                process_par_file(input_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# Usage example:
process_all_par_files("NG_par")