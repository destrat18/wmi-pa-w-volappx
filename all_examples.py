import time, argparse, os, logging
from wmipa.integration import FazaIntegrator, LatteIntegrator, VolestiIntegrator
from wmipa import WMI
from importlib.machinery import SourceFileLoader
import pathlib, json, pandas as pd

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        prog='Mega Miners',
        description='I am approximating!'
        )
        
    parser.add_argument("--degree", help="Handelman degree", type=int, default=None)
    parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
    parser.add_argument("--threshold", help="Error threshold", type=float, default=0.1)
    
    args = parser.parse_args()

    examples_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "data/examples")

    results = []

    for f_name in sorted(os.listdir(examples_path)):
        try:
            example = SourceFileLoader(f_name.split(".")[0], os.path.join(examples_path, f_name)).load_module()
            phi = example.phi
            w = example.w
            chi = example.chi

            print("####################### Example:", f_name, "#######################")

            print("Formula:", phi.serialize())
            print("Weight function:", w.serialize())
            print("Support:", chi.serialize())
            
            faza_integrator = FazaIntegrator(max_workers=args.max_workers, threshold=args.threshold)
            
            for integrator in [
                # LatteIntegrator(),
                # VolestiIntegrator(),
                faza_integrator,
            ]:
                for mode in [
                        WMI.MODE_SAE4WMI
                    ]:
                    start_time = time.time()
                    try:
                        faza_integrator.logs = []
                        wmi = WMI(chi, w, integrator=integrator)
                        volume, n_integrations = wmi.computeWMI(phi, mode=mode)
                        total_time = time.time()-start_time
                        print(
                            "WMI with mode {}, \t integrator = {}, \t volume = {}, \t # integrations = {}, \t time = {:.2f}s({:.2f}h)".format(
                                mode, integrator.__class__.__name__, volume, n_integrations, total_time,(total_time)/3600
                            )
                        )
                        
                        results.append(
                            {
                                'time': total_time,
                                'example': f_name,
                                'integrator': integrator.__class__.__name__,
                                'mode': mode,
                                'result': volume,
                                'n_integrations': n_integrations,
                                'logs': faza_integrator.logs
                            }
                        )
                        
                    except Exception as e:
                        print(
                            "WMI with mode {}, \t integrator = {}, \t failed = {}".format(
                                mode, integrator.__class__.__name__, str(e)
                            )
                        )
                        logging.exception(e)
                        results.append(
                            {
                                'time': time.time()-start_time,
                                'example': f_name,
                                'integrator': integrator.__class__.__name__,
                                'mode': mode,
                                'result': str(e),
                                'n_integrations': None,
                                'logs': faza_integrator.logs
                            }
                        )
                    
                    pd.DataFrame(results).to_csv(f"example_results_{int(start_time)}.csv", index=False)
                
                
        except Exception as e:
            logging.error(e)
            