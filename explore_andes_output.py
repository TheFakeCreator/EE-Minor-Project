import andes
import os
import numpy as np

andes.config_logger(stream_level=50)

def main():
    case_path = andes.get_case('ieee39/ieee39.xlsx')
    ss = andes.load(case_path, setup=False)
    ss.add('Fault', dict(idx='f1', bus=2, tf=1.0, tc=1.1, xf=1e-4, rf=0.0))
    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf = 3.0
    ss.TDS.run()
    
    # Try to extract rotor angles (delta) and speeds (omega) from memory
    # GENROU models are typically used. Let's see if they exist.
    if 'GENROU' in ss.models:
        gen = ss.GENROU
        print("GENROU delta indices:", gen.delta.a)
        
    # Check what is in the npz
    out_name = ss.files.res_name + '.npz'
    if os.path.exists(out_name):
        print(f"NPZ file {out_name} exists.")
        npz = np.load(out_name)
        print("NPZ keys:", npz.files)
        
    # Is there a cleaner way using andes?
    # Yes, typically ss.dae.ts holds time, and ss.dae.xy holds values.
    # But let's verify if ss.GENROU.delta has a history or if we must parse npz/lst.
        
if __name__ == '__main__':
    main()
