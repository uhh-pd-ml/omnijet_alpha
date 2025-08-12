#!/bin/bash
curl --output checkpoints.tar https://syncandshare.desy.de/public.php/dav/files/HX388YZbjSGPNcC/?accept=zip
tar -xvf checkpoints.tar
rm -rf checkpoints.tar
