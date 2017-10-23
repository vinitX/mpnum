#!/bin/bash

convert -background 'rgba(255,255,255,0)' -density 120 mpnum_logo.svg mpnum_logo_120.png 
convert -background 'rgba(255,255,255,0)' -density 144 mpnum_logo.svg mpnum_logo_144.png 

for i in tensors_*; do
  convert -background 'rgba(255,255,255,0)' -density 135 "$i" "${i/.svg/.png}"
done

