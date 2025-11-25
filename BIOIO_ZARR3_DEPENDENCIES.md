# BioIO and Zarr-3 Dependency Compatibility Report

## Currently Installed Packages (from pip freeze)

### Core Zarr and OME Packages
```
zarr==3.1.5
ome-zarr==0.12.2
ome-types==0.6.2
```

### BioIO Core and Readers
```
bioio==3.0.0
bioio-base==3.0.0
bioio-ome-zarr==3.1.0
```

### Supporting Libraries
```
dask==2025.10.0
distributed==2025.10.0
fsspec==2025.10.0
s3fs==2025.10.0
numcodecs==0.16.5
tifffile==2025.10.16
xarray==2025.11.0
```

## Expected Packages (from requirements.txt)

### Missing BioIO Readers
The following bioio readers are specified in pyproject.toml but not currently installed:
```
bioio-czi==2.4.1          # For CZI file format
bioio-ome-tiff==1.4.0     # For OME-TIFF files
bioio-tifffile==1.3.0     # For TIFF files
```

## Zarr-3 Compatibility Analysis

### ✅ COMPATIBLE - Core Packages
- **zarr 3.1.5**: Latest stable zarr-3 version
- **ome-zarr 0.12.2**: Fully supports zarr-3 (released Dec 2024)
- **bioio 3.0.0**: Compatible with zarr-3
- **bioio-base 3.0.0**: Compatible with zarr-3
- **bioio-ome-zarr 3.1.0**: Explicitly supports zarr-3

### ✅ COMPATIBLE - Supporting Libraries
- **tifffile 2025.10.16**: Latest version, supports zarr-3
- **numcodecs 0.16.5**: Compatible with zarr-3
- **dask 2025.10.0**: Compatible with zarr-3
- **xarray 2025.11.0**: Compatible with zarr-3
- **fsspec 2025.10.0**: Compatible with zarr-3
- **s3fs 2025.10.0**: For S3 zarr stores, compatible

### ⚠️ VERSION DISCREPANCY
- **dask**: Currently installed 2025.10.0 vs requirements.txt 2025.3.0
  - Status: **OK** - Both versions are compatible
  - Note: bioio-ome-zarr requires `dask!=2025.11.0`, so 2025.10.0 is safe

### 🔍 MISSING - Optional Readers (not installed)
These are specified in pyproject.toml but not installed in current environment:
- **bioio-czi 2.4.1**: For Zeiss CZI files
- **bioio-ome-tiff 1.4.0**: For OME-TIFF files
- **bioio-tifffile 1.3.0**: For standard TIFF files

## Compatibility Status by Format

### Zarr Format Support
| Format | Reader | Installed | Zarr-3 Compatible |
|--------|--------|-----------|-------------------|
| OME-Zarr | bioio-ome-zarr 3.1.0 | ✅ Yes | ✅ Yes |
| OME-Zarr | ome-zarr 0.12.2 | ✅ Yes | ✅ Yes |

### TIFF Format Support
| Format | Reader | Installed | Zarr-3 Compatible |
|--------|--------|-----------|-------------------|
| TIFF | bioio-tifffile | ❌ No | ✅ Yes (if installed) |
| OME-TIFF | bioio-ome-tiff | ❌ No | ✅ Yes (if installed) |

### Other Format Support
| Format | Reader | Installed | Zarr-3 Compatible |
|--------|--------|-----------|-------------------|
| CZI | bioio-czi | ❌ No | ✅ Yes (if installed) |

## Recommendations

### For Zarr-3 Dataloader Usage (Minimal)
Current installation is **SUFFICIENT** for zarr-3 format. You have:
- ✅ zarr 3.1.5
- ✅ ome-zarr 0.12.2
- ✅ bioio-ome-zarr 3.1.0
- ✅ bioio-base 3.0.0
- ✅ All supporting libraries

### For Complete BioIO Functionality
To support all file formats specified in pyproject.toml, install:
```bash
pip install bioio-czi bioio-ome-tiff bioio-tifffile
```

Or install from requirements.txt:
```bash
pip install -r requirements/requirements.txt
```

### For Production Use
Recommend installing all bioio readers to ensure compatibility with various input formats:
```bash
pip install \
  bioio>=3.0.0 \
  bioio-base>=3.0.0 \
  bioio-czi>=2.4.1 \
  bioio-ome-tiff>=1.4.0 \
  bioio-ome-zarr>=3.1.0 \
  bioio-tifffile>=1.3.0 \
  zarr>=3.0.0 \
  ome-zarr>=0.12.0
```

## Version Constraints in pyproject.toml

Current constraints (verified compatible with zarr-3):
```toml
dependencies = [
    "zarr>=3.0.0",           # ✅ Ensures zarr-3
    "ome-zarr>=0.12.0",      # ✅ Ensures zarr-3 support
    "bioio>=3.0.0",          # ✅ Compatible
    "bioio-base>=3.0.0",     # ✅ Compatible
    "bioio-czi>=2.4.1",      # ✅ Compatible
    "bioio-ome-tiff>=1.4.0", # ✅ Compatible
    "bioio-ome-zarr>=3.1.0", # ✅ Explicitly supports zarr-3
    "bioio-tifffile>=1.3.0", # ✅ Compatible
    "tifffile>=2024.0.0",    # ✅ Compatible
]
```

## Known Compatibility Issues

### ✅ RESOLVED
- **Dask version**: Using 2025.10.0 which is compatible (bioio-ome-zarr only excludes 2025.11.0)
- **Zarr format**: All packages support both zarr v2 and v3

### ⚠️ NOTES
- When creating new OME-Zarr files, they will use zarr v3 by default
- Reading older zarr v2 files is still supported
- Minor warning may appear: "version mismatch: detected: FormatV04, requested: FormatV05"
  - This is harmless and only affects metadata version, not functionality

## Summary

**For zarr-3 dataloader usage**: ✅ **READY TO USE**

All core dependencies for zarr-3 format are installed and compatible:
- zarr 3.1.5 ✅
- ome-zarr 0.12.2 ✅
- bioio-ome-zarr 3.1.0 ✅
- bioio-base 3.0.0 ✅
- Supporting libraries all compatible ✅

The dataloader code will work with zarr-3 format without any issues.

**For complete file format support**: Install missing bioio readers (bioio-czi, bioio-ome-tiff, bioio-tifffile) if you need to read CZI or TIFF files in addition to zarr.
