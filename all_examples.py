import time, argparse, os, logging
from wmipa.integration import FazaIntegrator, LatteIntegrator, VolestiIntegrator
from wmipa import WMI
from importlib.machinery import SourceFileLoader
import pathlib

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

    for f_name in os.listdir(examples_path):
        try:
            example = SourceFileLoader(f_name.split(".")[0], os.path.join(examples_path, f_name)).load_module()
            phi = example.phi
            w = example.w
            chi = example.chi

            print("####################### Example:", f_name, "#######################")

            print("Formula:", phi.serialize())
            print("Weight function:", w.serialize())
            print("Support:", chi.serialize())
            
            for integrator in [
                FazaIntegrator(max_workers=args.max_workers, threshold=args.threshold),
                LatteIntegrator(),
                VolestiIntegrator()
            ]:
                for mode in [
                        WMI.MODE_SAE4WMI
                    ]:
                    try:
                        start_time = time.time()
                        wmi = WMI(chi, w, integrator=integrator)
                        result, n_integrations = wmi.computeWMI(phi, mode=mode)
                        print(
                            "WMI with mode {}, \t integrator = {}, \t result = {}, \t # integrations = {}, \t time = {:.2f}s({:.2f}h)".format(
                                mode, integrator.__class__.__name__, result, n_integrations, time.time()-start_time,(time.time()-start_time)/3600
                            )
                        )
                    except Exception as e:
                        print(
                            "WMI with mode {}, \t integrator = {}, \t failed = {}".format(
                                mode, integrator.__class__.__name__, str(e)
                            )
                        )
        except Exception as e:
            logging.error(e)
            