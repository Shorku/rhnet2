import os
import subprocess

from zipfile import ZipFile, ZIP_LZMA


def orca_runner(inputs: list, work_dir: str, orca_path: str):
    cwd = os.getcwd()
    os.chdir(work_dir)
    for inp in inputs:
        with open(os.path.join(work_dir, f'{inp}.out'), 'w') as out:
            process = subprocess.Popen([orca_path, f'{inp}.inp'],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            for line in process.stdout:
                out.write(line.decode())
            process.wait()
    os.chdir(cwd)


def orca_cleanup(inputs_dft: list, inputs_dmt: list, work_dir: str):
    cwd = os.getcwd()
    os.chdir(work_dir)
    exts = ['inp',
            'prop',
            'gbw',
            'out',
            'property.txt',
            'bibtex',
            'densitiesinfo',
            'densities']
    for inp in inputs_dft:
        with ZipFile(f'{inp}.zip', 'w', compression=ZIP_LZMA) as zf:
            zf.write(f'{inp}.gbw')
            zf.write(f'{inp}.inp')
            zf.write(f'{inp}.out')
        for ext in exts:
            os.remove(os.path.join(work_dir, f'{inp}.{ext}'))
    for inp in inputs_dmt:
        with ZipFile(f'{inp}.zip', 'w', compression=ZIP_LZMA) as zf:
            zf.write(f'{inp}.out')
        for ext in exts:
            try:
                os.remove(os.path.join(work_dir, f'{inp}.{ext}'))
            except FileNotFoundError:
                continue
    os.chdir(cwd)
