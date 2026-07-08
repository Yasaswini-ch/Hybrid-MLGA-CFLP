import os

def preprocess():
    raw_dir = "data/raw"
    
    # Instance specifications from Table 1 of J.E. Beasley (1988)
    configs = {
        "capa": [
            ("capa1", 8000.0),
            ("capa2", 10000.0),
            ("capa3", 12000.0),
            ("capa4", 14000.0)
        ],
        "capb": [
            ("capb1", 5000.0),
            ("capb2", 6000.0),
            ("capb3", 7000.0),
            ("capb4", 8000.0)
        ],
        "capc": [
            ("capc1", 5000.0),
            ("capc2", 5750.0),
            ("capc3", 6500.0),
            ("capc4", 7250.0)
        ]
    }
    
    for base_name, instances in configs.items():
        src_path = os.path.join(raw_dir, f"{base_name}.txt")
        if not os.path.exists(src_path):
            print(f"[Error] Source file not found: {src_path}")
            continue
            
        with open(src_path, "r") as f:
            content = f.read()
            
        for inst_name, cap_val in instances:
            # Replace placeholder 'capacity' with the numeric capacity value
            # Beasley's files use the literal lowercase string 'capacity'
            inst_content = content.replace("capacity", f"{cap_val:.1f}")
            
            dest_path = os.path.join(raw_dir, f"{inst_name}.txt")
            with open(dest_path, "w") as f_out:
                f_out.write(inst_content)
                
            print(f"Generated: {dest_path} with capacity {cap_val}")

if __name__ == "__main__":
    preprocess()
