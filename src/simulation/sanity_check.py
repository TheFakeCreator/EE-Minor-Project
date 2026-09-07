import andes
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print(f"ANDES Version: {andes.__version__}")
    
    # Locate IEEE 39-bus case
    try:
        # get_case typically expects the exact file name if it's inside the cases folder, 
        # or the folder name if it has an init. We will try a few common patterns.
        try:
            case_path = andes.get_case('ieee39/ieee39.xlsx')
        except Exception:
            case_path = andes.get_case('ieee39')
            
        print(f"Found IEEE 39-bus case at: {case_path}")
    except Exception as e:
        print(f"Error finding case: {e}")
        return

    # Run power flow
    print("\n--- Running Power Flow ---")
    ss = andes.run(case_path, routine='pflow')
    
    # Run time-domain simulation
    print("\n--- Running Time-Domain Simulation ---")
    ss = andes.run(case_path, routine='tds')
    
    print("\nSanity check completed successfully.")

if __name__ == "__main__":
    main()
