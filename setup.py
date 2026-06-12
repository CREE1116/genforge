import os
import sys
import platform
import subprocess
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py

class CustomBuildPy(build_py):
    def run(self):
        super().run()

        src_ext_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "oqboost", "_ext"))
        build_ext_dir = os.path.abspath(os.path.join(self.build_lib, "oqboost", "_ext"))

        system = platform.system()
        if system == "Darwin":
            lib_name = "liboqboost.dylib"
            cmd = ["clang++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17",
                   "-framework", "Accelerate"]
            try:
                omp = subprocess.check_output(["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL).decode().strip()
                cmd += ["-Xpreprocessor", "-fopenmp", f"-I{omp}/include", f"-L{omp}/lib", "-lomp"]
            except Exception:
                pass
        elif system == "Windows":
            lib_name = "oqboost.dll"
            cmd = ["cl", "/O2", "/LD", "/EHsc", "/openmp"]
        else:
            lib_name = "liboqboost.so"
            cmd = ["g++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17", "-fopenmp"]

        src_oqboost = os.path.join(src_ext_dir, "oqboost.cpp")
        src_lib      = os.path.join(src_ext_dir, lib_name)
        build_lib_path = os.path.join(build_ext_dir, lib_name)

        if os.path.exists(src_oqboost):
            compile_cmd = cmd + [src_oqboost] + ["-o", src_lib]
            print(f"Compiling C++ extension: {' '.join(compile_cmd)}")
            subprocess.run(compile_cmd, check=True)

            os.makedirs(build_ext_dir, exist_ok=True)
            print(f"Copying compiled library to build folder: {build_lib_path}")
            shutil.copy2(src_lib, build_lib_path)

            for src_name in ["oqboost.cpp", "oqboost_types.h", "oqboost_core.h"]:
                src_file_in_build = os.path.join(build_ext_dir, src_name)
                if os.path.exists(src_file_in_build):
                    print(f"Removing source file {src_file_in_build} from build folder")
                    os.remove(src_file_in_build)
        else:
            print("C++ source files not found in source directory. Skipping compilation.")

setup(
    cmdclass={"build_py": CustomBuildPy},
)
