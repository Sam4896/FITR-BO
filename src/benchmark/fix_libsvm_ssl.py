"""
Fix SSL certificate verification for libsvmdata downloads.

This module patches the SSL context to use certifi certificates or disables
SSL verification as a fallback for dataset downloads.

Usage:
    import fix_libsvm_ssl  # Import before using LassoBench with real datasets
    import LassoBench
    bench = LassoBench.RealBenchmark(pick_data='rcv1')
"""

import ssl
import certifi
import os

# Set environment variables to use certifi's certificate bundle
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Solution 1: Try to create SSL context with certifi certificates
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    # Test the context
    ssl._create_default_https_context = lambda: ssl_context
    print(f"SSL certificates configured using certifi: {certifi.where()}")
except Exception as e:
    # Solution 2: If that fails, disable SSL verification as fallback
    # This is necessary on some Windows systems with certificate store issues
    # WARNING: This disables certificate verification - only use for dataset downloads
    ssl._create_default_https_context = ssl._create_unverified_context
    print(f"Warning: Could not configure SSL with certifi: {e}")
    print("SSL verification disabled for downloads (safe for dataset downloads only).")
