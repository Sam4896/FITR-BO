* Download binary data from https://github.com/LeoIV/BenchSuite/tree/master/data/mopta08
* Put them here:
  - **Linux (x86_64)**: `mopta08_elf64.bin`
  - **Linux (i386)**: `mopta08_elf32.bin`
  - **ARM**: `mopta08_armhf.bin`
* **Windows**: BenchSuite does not provide a Windows executable. The code expects `mopta08_amd64.exe` in this directory; you must obtain it elsewhere or run MOPTA08 on Linux/WSL. On Windows, `DefineProblems("MOPTA08")` will raise `FileNotFoundError` if the exe is missing.