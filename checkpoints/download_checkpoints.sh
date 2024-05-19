#!/bin/bash
curl --output checkpoints.tar https://syncandshare.desy.de/index.php/s/JYiSxBTo6FQGXqk/download\?path\=\&files\=checkpoints.tar
tar -xvf checkpoints.tar
rm -rf checkpoints.tar
